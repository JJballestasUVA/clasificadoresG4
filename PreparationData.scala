import org.apache.spark.ml.feature.{Imputer, StringIndexer, VectorAssembler, StandardScaler}
import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.functions.{udf, col, cos, sin, sqrt, when, lit}
import org.apache.spark.sql.types.DoubleType
import spark.implicits._
import org.apache.spark.sql.types.IntegerType

 

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


  stdYN.withColumn("Month", month(col("Date"))).withColumn("DayMonth", date_format(col("Date"), "ddMM"))  // <-- NUEVA COLUMNA ddMM


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

// === 7) Normalizaciones post-split ===
def normalizeRest(df: DataFrame): DataFrame = {
  Seq("WindGustDir","WindDir9am","WindDir3pm").foldLeft(df){ (acc,c) =>
    if (acc.columns.contains(c)) acc.withColumn(c, upper(trim(col(c)))) else acc
  }
}
val trainN = normalizeRest(trainDF)
val testN  = normalizeRest(testDF)

// === 8) Columnas según distribución esperada ===
val normalNumCols = Seq("MinTemp","MaxTemp","Temp9am","Temp3pm","Pressure9am","Pressure3pm").filter(trainN.columns.contains)

val nonNormalNumCols = Seq("Rainfall","Evaporation","Sunshine","Humidity9am","Humidity3pm",
  "Cloud9am","Cloud3pm","WindGustSpeed","WindSpeed9am","WindSpeed3pm").filter(trainN.columns.contains)

val discrete_NumCols= Seq("Cloud9am","Cloud3pm").filter(trainN.columns.contains)


val windCatCols = Seq("WindGustDir","WindDir9am","WindDir3pm").filter(trainN.columns.contains)

// === 9) Métricas por clase para imputación ===

def p50(colName: String): Column = expr(s"percentile_approx($colName, 0.5)")


//train DataFrame base donde están los datos.   trainN
//c     Nombre de la columna que quieres agregar.       "Temp3pm"
//keys  Lista de columnas por las que agrupar.  Seq("Location","RainTomorrow")
//aggCol        Expresión de agregación que aplicarás.  avg(col(c)) o p50(c)
//suffix        Texto que se añadirá al nombre de la nueva columna para identificarla.

 def aggregationBy( train: DataFrame, c: String, keys: Seq[String], aggCol: Column, suffix: String
): DataFrame = {
  train.groupBy(keys.map(col): _*).agg(aggCol.alias(s"${c}_${suffix}"))
}

// ===================
// 1) Media (mean)
// ===================

  // Por Location + DayMonth + clase para el conjunto de Train
def meanByLocDayMonthClass(train: DataFrame, c: String): DataFrame =aggregationBy(train, c, Seq("Location","DayMonth","RainTomorrow"), avg(col(c)), s"locdaycla_mean")

// Por DayMonth + Location Para el Conjunto de Test
def meanByLocDayMonth(train: DataFrame, c: String): DataFrame =aggregationBy(train, c, Seq("Location","DayMonth"), avg(col(c)), s"locday_mean")



// ====================
// 2) MEDIANAS (median)
// ====================

 // Por Location + DayMonth + clase para el conjunto de Train
def medianByLocDayMonthClass(train: DataFrame, c: String): DataFrame =aggregationBy(train, c,  Seq("Location","DayMonth","RainTomorrow"), p50(c), s"locdaycla_median")

// Por DayMonth + Location Para el Conjunto de Test
def medianByLocDayMonth(train: DataFrame, c: String): DataFrame =aggregationBy(train, c, Seq("Location","DayMonth"), p50(c), s"locday_median")



// ==========================
// 3) MODA CATEGÓRICA genérica
// ==========================
def modeBy(groups: Seq[String], target: String, df: DataFrame): DataFrame = {
  val counted = df.groupBy((groups :+ target).map(col): _*).agg(count(lit(1)).alias("cnt"))
  counted.groupBy(groups.map(col): _*).agg(expr(s"max_by($target, cnt)").alias(target))
}

val windDirCols = Seq("WindGustDir", "WindDir9am", "WindDir3pm")

 // Por Location + DayMonth + clase para el conjunto de Train
def modeByLocDayMonthClass(df: DataFrame, target: String): DataFrame = modeBy(Seq("Location","DayMonth","RainTomorrow"), target, df)

 // Por Location + DayMonth + clase para el conjunto de Test
def modeByLocDayMonth(df: DataFrame, target: String): DataFrame = modeBy(Seq("Location","DayMonth"), target, df)


//Imputación para Train
def imputeDiscreteNearestMeanByTrain(df: DataFrame, train: DataFrame, c: String): DataFrame = {
  // Calcula la media por Location, DayMonth y Clase (RainTomorrow)
  val meanDF = meanByLocDayMonthClass(train.filter(col(c).isNotNull), c)

  df.join(meanDF, Seq("Location","DayMonth","RainTomorrow"), "left")
    .withColumn(
      c,
      when(
        col(c).isNull,
        // Redondea la media al entero más cercano y hace cast a Int
        round(col(s"${c}_locdaycla_mean")).cast("int")
      ).otherwise(col(c))
    ).drop(s"${c}_locdaycla_mean")
}

def imputeDiscreteNearestMeanByTest(df: DataFrame, train: DataFrame, c: String): DataFrame = {
  // Calcula la media por Location, DayMonth
  val meanDF = meanByLocDayMonth(train.filter(col(c).isNotNull), c)

  df.join(meanDF, Seq("Location","DayMonth"), "left")
    .withColumn(
      c,
      when(
        col(c).isNull,
        // Redondea la media al entero más cercano y hace cast a Int
        round(col(s"${c}_locday_mean")).cast("int")
      ).otherwise(col(c))
    ).drop(s"${c}_locday_mean")
}


def imputeNonNormalByTrain(df: DataFrame, train: DataFrame, c: String): DataFrame = {
  val sLocMn  = s"${c}_locdaycla_median"

  val j1 = df.join(medianByLocDayMonthClass(train.filter(col(c).isNotNull), c),
                   Seq("Location","DayMonth","RainTomorrow"), "left")

  j1.withColumn(
      c,
      when(col(c).isNull, coalesce(col(sLocMn))).otherwise(col(c))
    ).drop(sLocMn)
}

def imputeNonNormalByTest(df: DataFrame, train: DataFrame, c: String): DataFrame = {
  val sLocMn  = s"${c}_locday_median"

  val j1 = df.join(medianByLocDayMonth(train.filter(col(c).isNotNull), c),
                   Seq("Location","DayMonth"), "left")

  j1.withColumn(
      c,
      when(col(c).isNull, coalesce(col(sLocMn))).otherwise(col(c))
    ).drop(sLocMn)
}





def imputeNormalByTrain(df: DataFrame, train: DataFrame, c: String): DataFrame = {
  val sLocMn  = s"${c}_locdaycla_mean"

  val j1 = df.join(meanByLocDayMonthClass(train.filter(col(c).isNotNull), c),
                   Seq("Location","DayMonth","RainTomorrow"), "left")


  j1.withColumn(
      c,
      when(col(c).isNull, coalesce(col(sLocMn))).otherwise(col(c))
    ).drop(sLocMn)
}


def imputeNormalByTest(df: DataFrame, train: DataFrame, c: String): DataFrame = {
  val sLocDm = s"${c}_locday_mean"

  val j1 = df.join(meanByLocDayMonth(train.filter(col(c).isNotNull), c),
                   Seq("Location","DayMonth"), "left")

  j1.withColumn(
      c,
      when(col(c).isNull, col(sLocDm)).otherwise(col(c))
    ).drop(sLocDm)
}








def applyWindImputeTrain(df: DataFrame, train: DataFrame): DataFrame = {
  windDirCols.foldLeft(df){ (acc, c) =>
    // Moda por Location + RainTomorrow para la columna c, renombrada para evitar ambigüedad
    val modeDF = modeByLocDayMonthClass(train.filter(col(c).isNotNull), c).withColumnRenamed(c, s"${c}_mode")

    // Join seguro y reemplazo sólo cuando falte
    acc.join(modeDF, Seq("Location","RainTomorrow"), "left")
      .withColumn(
        c,
        when(isMissingStr(col(c)), col(s"${c}_mode")).otherwise(col(c))
      ).drop(s"${c}_mode")
  }
}


def applyWindImputeTest(df: DataFrame, train: DataFrame): DataFrame = {
  windDirCols.foldLeft(df){ (acc, c) =>
    // Moda por Location   para la columna c, renombrada para evitar ambigüedad
    val modeDF = modeByLocDayMonth(train.filter(col(c).isNotNull), c).withColumnRenamed(c, s"${c}_mode")

    // Join seguro y reemplazo sólo cuando falte
    acc.join(modeDF, Seq("Location"), "left")
      .withColumn(
        c,
        when(isMissingStr(col(c)), col(s"${c}_mode")).otherwise(col(c))
      ).drop(s"${c}_mode")
  }
}


//======================================================================================================

// === 10) Imputación por clase (media/mediana y valor mas cercano a la media) ===

def applyImputeTrain(df: DataFrame, train: DataFrame): DataFrame = {

  // Imputación  para columnas normales (media)
  val withNormal = normalNumCols.foldLeft(df){ (acc, c) =>
    imputeNormalByTrain(acc, train, c)
  }

  // Imputación  para columnas no normales (mediana)
  val withNonNormal = nonNormalNumCols.foldLeft(withNormal){ (acc, c) =>
    imputeNonNormalByTrain(acc, train, c)
  }

 val withNonNormalDiscrete= discrete_NumCols.foldLeft(withNonNormal){ (acc, c) =>
    imputeDiscreteNearestMeanByTrain(acc, train, c)
  }
  withNonNormalDiscrete
}

def applyImputeTest(df: DataFrame, train: DataFrame): DataFrame = {

  // Imputación  para columnas normales (media)
  val withNormal = normalNumCols.foldLeft(df){ (acc, c) =>
    imputeNormalByTest(acc, train, c)
  }

  // Imputación  para columnas no normales (mediana)
  val withNonNormal = nonNormalNumCols.foldLeft(withNormal){ (acc, c) =>
    imputeNonNormalByTest(acc, train, c)
  }

 val withNonNormalDiscrete= discrete_NumCols.foldLeft(withNonNormal){ (acc, c) =>
    imputeDiscreteNearestMeanByTest(acc, train, c)
  }
  withNonNormalDiscrete
}



def imputeRainTodayByTrain(df: DataFrame): DataFrame = {
  // Normalizamos RainToday a mayúsculas
  val base = df.withColumn("RainToday", upper(trim(col("RainToday"))))

  // Calculamos la moda de RainToday por clase (RainTomorrow)
  val rainTodayMode = modeBy(Seq("RainTomorrow"), "RainToday", base).withColumnRenamed("RainToday", "RainToday_mode")

  // Reemplazamos los nulos o vacíos atendiendo a la clase
  base.join(rainTodayMode, Seq("RainTomorrow"), "left").withColumn("RainToday",
      when(isMissingStr(col("RainToday")), col("RainToday_mode")).otherwise(col("RainToday"))
    ).drop("RainToday_mode")
}


def imputeRainTodayByTest(df: DataFrame): DataFrame = {
  val base = df.withColumn("RainToday", upper(trim(col("RainToday"))))

  // Crea una columna auxiliar que indica lluvia según Rainfall (>0.0)
  val withRainfallFlag = base.withColumn("RainToday_pred",
    when(col("Rainfall") > 0.0, lit("Yes")).otherwise(lit("No"))
  )

  // Calcula por DayMonth la moda del valor predicho (para mantener consistencia)
  val predMode = modeBy(Seq("DayMonth"), "RainToday_pred", withRainfallFlag).withColumnRenamed("RainToday_pred", "RainToday_mode")

  // Une y sustituye RainToday vacío por la predicción o moda por DayMonth
  withRainfallFlag.join(predMode, Seq("DayMonth"), "left").withColumn("RainToday",
      when(isMissingStr(col("RainToday")), coalesce(col("RainToday_pred"), col("RainToday_mode")))
          .otherwise(col("RainToday"))).drop("RainToday_pred", "RainToday_mode")
}




//===========================================================================================================

// Imputación numérica
// Primero: numéricas
val trainM1 = applyImputeTrain(trainN, trainN)
val testM1  = applyImputeTest(testN , trainN)

// Luego: imputar RainToday basándose en Rainfall
val trainM2 = imputeRainTodayByTrain(trainM1)
val testM2  = imputeRainTodayByTest(testM1)

// Finalmente: imputar viento y demás
val trainM3 = applyWindImputeTrain(trainM2, trainM2)
val testM3  = applyWindImputeTest(testM2, trainM2)



 
// === 11) Checks finales ===
println(s"\n=== RESUMEN FINAL ===")

// Conteo simple
val trainCount = trainM1.count()
val testCount = testM1.count()
println(s"Total Train: $trainCount registros")
println(s"Total Test:  $testCount registros")

// Solo mostrar las primeras filas
println("\n--- Muestra de datos Train ---")
trainM1.select("Date", "Location", "MinTemp", "MaxTemp", "RainToday", "RainTomorrow").show(5)

println("\n--- Muestra de datos Test ---")
testM1.select("Date", "Location", "MinTemp", "MaxTemp", "RainToday", "RainTomorrow").show(5)

println("\n✓ Proceso completado exitosamente")
 
  
 
 //=====================================TRANSFORMACIONES =================================
val trainFinal = trainM3
val testFinal = testM3

 