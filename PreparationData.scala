import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

Logger.getLogger("org.apache.spark.scheduler.DAGScheduler").setLevel(Level.ERROR)

val PATH = "."
val FILE = "weatherAUS.csv"

// === 1) Lectura base ===
val weatherDF = spark.read.option("header","true").option("inferSchema","true").option("delimiter",",").csv(s"$PATH/$FILE")

// === 2) Casting único y seguro ===
val atributosDouble = Seq(
  "MinTemp","MaxTemp","Temp9am","Temp3pm",
  "Rainfall","Evaporation","Sunshine",
  "Pressure9am","Pressure3pm",
  "Humidity9am","Humidity3pm",
  "WindGustSpeed","WindSpeed9am","WindSpeed3pm"
)
val atributosInt = Seq("Cloud9am","Cloud3pm")

val dfCasted1 = atributosDouble.foldLeft(weatherDF){ (df, c) =>
  if (df.columns.contains(c)) df.withColumn(c, col(c).cast("double")) else df
}
val dfPrepared = atributosInt.foldLeft(dfCasted1){ (df, c) =>
  if (df.columns.contains(c)) df.withColumn(c, col(c).cast("int")) else df
}

// === 3) Utilidades de missing ===
// Categóricas (texto)
def isMissingStr(c: Column): Column = c.isNull || trim(c) === "" ||lower(trim(c)).isin("na","nan","null","n/a","none","missing")
  
// Numéricas
def isNullNum(c: Column): Column = c.isNull

// === 4) Normaliza etiqueta/fecha y tokens de viento (PRE-split) ===
def normalizeLabelAndBasics(df: DataFrame): DataFrame = {
  val base = if (df.schema("Date").dataType != DateType)
    df.withColumn("Date", to_date(col("Date")))
  else df

  val stdYN = Seq("RainTomorrow").foldLeft(base){ (acc,c) =>
    if (acc.columns.contains(c))
      acc.withColumn(c, when(lower(trim(col(c))) === "yes","Yes").when(lower(trim(col(c))) === "no","No").otherwise(col(c))
      )
    else acc
  }

  stdYN.withColumn("Month", month(col("Date")))
}

val missingTokens = Seq("NA","N/A","NULL","NONE","MISSING","-","?")

val df0 = normalizeLabelAndBasics(dfPrepared).transform { d =>
  Seq("WindGustDir","WindDir9am","WindDir3pm").foldLeft(d){ (acc,c) =>
    if (acc.columns.contains(c)) {
      val v = trim(upper(col(c)))
      acc.withColumn(c, when(v.isin(missingTokens: _*), lit(null:String)).otherwise(v))
    } else acc
  }
}

// === 5) Filtra etiquetas válidas ===
val dfClean = df0.filter(col("RainTomorrow").isin("Yes","No"))

// === 6) Split estratificado 80/20 (único) ===
val seed = 2025L
val Array(yesTrain, yesTest) = dfClean.filter(col("RainTomorrow")==="Yes").randomSplit(Array(0.8,0.2), seed)
val Array(noTrain ,  noTest)  = dfClean.filter(col("RainTomorrow")==="No" ).randomSplit(Array(0.8,0.2), seed)

val trainDF = yesTrain.unionByName(noTrain)
val testDF  = yesTest.unionByName(noTest)

// === 7) Normalizaciones deterministas post-split ===
def normalizeRest(df: DataFrame): DataFrame = {
  Seq("WindGustDir","WindDir9am","WindDir3pm").foldLeft(df){ (acc,c) =>
    if (acc.columns.contains(c)) acc.withColumn(c, upper(trim(col(c)))) else acc
  }
}
val trainN = normalizeRest(trainDF)
val testN  = normalizeRest(testDF)

// === 8) Columnas según distribución esperada ===
val normalNumCols = Seq("MinTemp","MaxTemp","Temp9am","Temp3pm","Pressure9am","Pressure3pm").filter(trainN.columns.contains)
val nonNormalNumCols = Seq(
  "Rainfall","Evaporation","Sunshine","Humidity9am","Humidity3pm",
  "Cloud9am","Cloud3pm","WindGustSpeed","WindSpeed9am","WindSpeed3pm"
).filter(trainN.columns.contains)
val windCatCols = Seq("WindGustDir","WindDir9am","WindDir3pm").filter(trainN.columns.contains)

// === 9) Métricas por clase para imputación ===
def p50(colName: String): Column = expr(s"percentile_approx($colName, 0.5)")

def classMean(train: DataFrame, c: String): DataFrame =
  train.groupBy("RainTomorrow").agg(avg(col(c)).alias(s"${c}_rt_mean"))
def classMedian(train: DataFrame, c: String): DataFrame =
  train.groupBy("RainTomorrow").agg(p50(c).alias(s"${c}_rt_median"))

def locClassMean(train: DataFrame, c: String): DataFrame =
  train.groupBy("Location","RainTomorrow").agg(avg(col(c)).alias(s"${c}_locrt_mean"))
def locClassMedian(train: DataFrame, c: String): DataFrame =
  train.groupBy("Location","RainTomorrow").agg(p50(c).alias(s"${c}_locrt_median"))

def modeBy(groups: Seq[String], target: String, df: DataFrame): DataFrame = {
  val counts = df.groupBy((groups :+ target).map(col): _*).count()
  val w = Window.partitionBy(groups.map(col): _*).orderBy(desc("count"), col(target))
  counts.withColumn("rn", row_number().over(w)).filter(col("rn") === 1).drop("count","rn")
}

// === 10) Imputación por clase (media/mediana) ===
def applyClassImpute(df: DataFrame, train: DataFrame): DataFrame = {
  val withNormal = normalNumCols.foldLeft(df){ (acc,c) =>
    val rtMean = classMean(train.filter(col(c).isNotNull), c)
    acc.join(rtMean, Seq("RainTomorrow"), "left")
      .withColumn(c, when(isNullNum(col(c)), col(s"${c}_rt_mean")).otherwise(col(c)))
      .drop(s"${c}_rt_mean")
  }

  val withNonNormal = nonNormalNumCols.foldLeft(withNormal){ (acc,c) =>
    val rtMedian = classMedian(train.filter(col(c).isNotNull), c)
    acc.join(rtMedian, Seq("RainTomorrow"), "left")
      .withColumn(c, when(isNullNum(col(c)), col(s"${c}_rt_median")).otherwise(col(c)))
      .drop(s"${c}_rt_median")
  }

  withNonNormal
}

val trainM1 = applyClassImpute(trainN, trainN)
val testM1  = applyClassImpute(testN , trainN)

// === 11) Utilidad para revisar distribución ===
def distributionView(df: DataFrame, name: String): Unit = {
  val total = df.count()
  println(s"\nDistribución en $name (N=$total):")
  df.groupBy("RainTomorrow").count().withColumn("pct", round(col("count") * 100.0 / lit(total), 2)).orderBy(desc("count")).show(false)
}

// === 12) Checks finales ===
println(s"Train: ${trainM1.count()}  |  Test: ${testM1.count()}")
distributionView(dfClean, "Global (antes del split, ya normalizado)")
distributionView(trainM1, "Train imputado")
distributionView(testM1 , "Test imputado")
