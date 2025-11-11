// ============================================================
// VERSIÓN ULTRA-OPTIMIZADA - PreparationDatav4R5
// Reducción masiva de joins mediante batch processing
// ============================================================

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{Imputer, StringIndexer, VectorAssembler, StandardScaler}

Logger.getLogger("org.apache.spark.scheduler.DAGScheduler").setLevel(Level.ERROR)

val PATH = "."
val FILE = "weatherAUS.csv"

// === 1) Lectura y preparación ===
val weatherDF = spark.read.option("header","true").option("inferSchema","true").option("delimiter",",").csv(s"$PATH/$FILE")

// === 2) Casting (todas las columnas en un solo select) ===
val atributosDouble = Seq("MinTemp","MaxTemp","Temp9am","Temp3pm","Rainfall","Evaporation","Sunshine","Pressure9am","Pressure3pm","Humidity9am","Humidity3pm","WindGustSpeed","WindSpeed9am","WindSpeed3pm")
val atributosInt = Seq("Cloud9am","Cloud3pm")
val dfPrepared = weatherDF.select(weatherDF.columns.map { c => if (atributosDouble.contains(c)) col(c).cast("double").alias(c) else if (atributosInt.contains(c)) col(c).cast("int").alias(c) else col(c) }: _*)

// === 3) Utilidades de missing ===
def isMissingStr(c: Column): Column = c.isNull || trim(c) === "" || lower(trim(c)).isin("na","nan","null","n/a","none","missing")
def isNullNum(c: Column): Column = c.isNull

// === 4) Normalización (todo en un solo select) ===
val missingTokens = Seq("NA","N/A","NULL","NONE","MISSING","-","?")
val windCols = Seq("WindGustDir","WindDir9am","WindDir3pm")
val dfNormalized = dfPrepared.select(dfPrepared.columns.map { c => if (c == "Date") to_date(col("Date")).alias("Date") else if (c == "RainTomorrow") when(lower(trim(col(c))) === "yes","Yes").when(lower(trim(col(c))) === "no","No").otherwise(col(c)).alias(c) else if (windCols.contains(c)) when(upper(trim(col(c))).isin(missingTokens.map(_.toUpperCase): _*), lit(null)).otherwise(upper(trim(col(c)))).alias(c) else col(c) }: _*).withColumn("Month", month(col("Date"))).withColumn("DayMonth", date_format(col("Date"), "ddMM"))

// === 5) Split estratificado ===
val seed = 2025L
val dfClean = dfNormalized.filter(col("RainTomorrow").isin("Yes","No"))
val Array(yesTrain, yesTest) = dfClean.filter(col("RainTomorrow")==="Yes").randomSplit(Array(0.8,0.2), seed)
val Array(noTrain, noTest) = dfClean.filter(col("RainTomorrow")==="No").randomSplit(Array(0.8,0.2), seed)
val trainN = yesTrain.unionByName(noTrain).persist()
val testN = yesTest.unionByName(noTest).persist()

println(s"\n DataFrames cacheados: Train=${trainN.count()}, Test=${testN.count()}")

// === 6) Configuración de columnas ===
val normalNumCols = Seq("MinTemp","MaxTemp","Temp9am","Temp3pm","Pressure9am","Pressure3pm")
val nonNormalNumCols = Seq("Rainfall","Evaporation","Sunshine","Humidity9am","Humidity3pm","Cloud9am","Cloud3pm","WindGustSpeed","WindSpeed9am","WindSpeed3pm")
val discreteNumCols = Seq("Cloud9am","Cloud3pm")
val windDirCols = Seq("WindGustDir","WindDir9am","WindDir3pm")

// === 7) Funciones de agregación  ===
def p50(colName: String): Column = expr(s"percentile_approx($colName, 0.5)")

// Calcular  estadísticas 
def calculateAllStatsForTrain(train: DataFrame, numCols: Seq[String], discCols: Seq[String]): DataFrame = {
  val stats = numCols.flatMap(c => Seq(avg(col(c)).alias(s"${c}_mean"), p50(c).alias(s"${c}_median"))) ++ discCols.map(c => avg(col(c)).alias(s"${c}_mean"))
  train.groupBy("Location","DayMonth","RainTomorrow").agg(stats.head, stats.tail: _*)
}

def calculateAllStatsForTest(train: DataFrame, numCols: Seq[String], discCols: Seq[String]): DataFrame = {
  val stats = numCols.flatMap(c => Seq(avg(col(c)).alias(s"${c}_mean"), p50(c).alias(s"${c}_median"))) ++ discCols.map(c => avg(col(c)).alias(s"${c}_mean"))
  train.groupBy("Location","DayMonth").agg(stats.head, stats.tail: _*)
}

// Calcular modas de viento en un solo paso
def calculateWindModesForTrain(train: DataFrame): DataFrame = {
  windDirCols.map { c => train.filter(col(c).isNotNull && !isMissingStr(col(c))).groupBy("Location","DayMonth","RainTomorrow",c).agg(count(lit(1)).alias("cnt")).groupBy("Location","DayMonth","RainTomorrow").agg(expr(s"max_by($c, cnt)").alias(s"${c}_mode")) }.reduce((df1,df2) => df1.join(df2, Seq("Location","DayMonth","RainTomorrow"), "outer"))
}

def calculateWindModesForTest(train: DataFrame): DataFrame = {
  windDirCols.map { c => train.filter(col(c).isNotNull && !isMissingStr(col(c))).groupBy("Location","DayMonth",c).agg(count(lit(1)).alias("cnt")).groupBy("Location","DayMonth").agg(expr(s"max_by($c, cnt)").alias(s"${c}_mode")) }.reduce((df1,df2) => df1.join(df2, Seq("Location","DayMonth"), "outer"))
}

def calculateRainTodayModeForTrain(train: DataFrame): DataFrame = {
  train.withColumn("RainToday", upper(trim(col("RainToday")))).filter(!isMissingStr(col("RainToday"))).groupBy("RainTomorrow","RainToday").agg(count(lit(1)).alias("cnt")).groupBy("RainTomorrow").agg(expr("max_by(RainToday, cnt)").alias("RainToday_mode"))
}

// === 8) Imputación estadísticas ===

//====================estadisticas sin jerarquia =============================
def applyBatchImputeTrain(df: DataFrame, train: DataFrame): DataFrame = {
  println("  → Calculando todas las estadísticas...")
  val allCols = normalNumCols ++ nonNormalNumCols
  val columnsToSelect = Seq("Location","DayMonth","RainTomorrow") ++ allCols
  
  val statsDF = calculateAllStatsForTrain(
    train.select(columnsToSelect.map(col): _*).filter(allCols.map(c => col(c).isNotNull).reduce(_ || _)), 
    normalNumCols ++ nonNormalNumCols.diff(discreteNumCols), 
    discreteNumCols
  )
  
  val dfJoined = df.join(statsDF, Seq("Location","DayMonth","RainTomorrow"), "left")
  
  val imputedCols = df.columns.map { c =>
    if (normalNumCols.contains(c)) when(col(c).isNull, coalesce(col(s"${c}_mean"))).otherwise(col(c)).alias(c)
    else if (nonNormalNumCols.diff(discreteNumCols).contains(c)) when(col(c).isNull, coalesce(col(s"${c}_median"))).otherwise(col(c)).alias(c)
    else if (discreteNumCols.contains(c)) when(col(c).isNull, round(coalesce(col(s"${c}_mean"), lit(0))).cast("int")).otherwise(col(c)).alias(c)
    else col(c)
  }
  
  dfJoined.select(imputedCols: _*)
}

def applyBatchImputeTest(df: DataFrame, train: DataFrame): DataFrame = {
   val allCols = normalNumCols ++ nonNormalNumCols
  val statsDF = calculateAllStatsForTest(train.select("Location","DayMonth" +: allCols: _*).filter(allCols.map(c => col(c).isNotNull).reduce(_ || _)), normalNumCols ++ nonNormalNumCols.diff(discreteNumCols), discreteNumCols)

  val dfJoined = df.join(statsDF, Seq("Location","DayMonth"), "left")
  
  val imputedCols = df.columns.map { c =>
    if (normalNumCols.contains(c)) when(col(c).isNull, col(s"${c}_mean")).otherwise(col(c)).alias(c)
    else if (nonNormalNumCols.diff(discreteNumCols).contains(c)) when(col(c).isNull, col(s"${c}_median")).otherwise(col(c)).alias(c)
    else if (discreteNumCols.contains(c)) when(col(c).isNull, round(coalesce(col(s"${c}_mean"), lit(0))).cast("int")).otherwise(col(c)).alias(c)
    else col(c)
  }
  
  dfJoined.select(imputedCols: _*)
}

//=======================================


// ============================================================
// 10) IMPUTACIÓN JERÁRQUICA
// ============================================================

def calculateStatsTest(
    df: DataFrame,
    normalCols: Seq[String],
    nonNormalCols: Seq[String],
    discreteCols: Seq[String],
    groupCols: Seq[String]
): DataFrame = {

  import org.apache.spark.sql.functions._

  // --- Calcular medias ---
  val meanAggs = normalCols.map { c =>
    avg(col(c)).alias(s"${c}_mean_${groupCols.mkString("_")}")
  }

  // --- Calcular medianas ---
  val medianAggs = nonNormalCols.map { c =>
    expr(s"percentile_approx($c, 0.5)").alias(s"${c}_median_${groupCols.mkString("_")}")
  }

  // --- Calcular medias para discretas ---
  val discreteAggs = discreteCols.map { c =>
    avg(col(c)).alias(s"${c}_mean_${groupCols.mkString("_")}")
  }

  // --- Un solo groupBy con todas las agregaciones ---
  val aggExprs = meanAggs ++ medianAggs ++ discreteAggs
  df.groupBy(groupCols.map(col): _*).agg(aggExprs.head, aggExprs.tail: _*)
}


def calculateStatsTrain(
    df: DataFrame,
    normalCols: Seq[String],
    nonNormalCols: Seq[String],
    discreteCols: Seq[String],
    groupCols: Seq[String]
): DataFrame = {

  import org.apache.spark.sql.functions._

  // --- Calcular medias ---
  val meanAggs = normalCols.map { c =>
    avg(col(c)).alias(s"${c}_mean_${groupCols.mkString("_")}")
  }

  // --- Calcular medianas ---
  val medianAggs = nonNormalCols.map { c =>
    expr(s"percentile_approx($c, 0.5)").alias(s"${c}_median_${groupCols.mkString("_")}")
  }

  // --- Calcular medias para discretas ---
  val discreteAggs = discreteCols.map { c =>
    avg(col(c)).alias(s"${c}_mean_${groupCols.mkString("_")}")
  }

  // --- Un solo groupBy con todas las agregaciones ---
  val aggExprs = meanAggs ++ medianAggs ++ discreteAggs
  df.groupBy(groupCols.map(col): _*).agg(aggExprs.head, aggExprs.tail: _*)
}


def applyBatchImputeTrain(df: DataFrame, train: DataFrame): DataFrame = {
  // --- Precalcular estadísticas ---
  val statsDayMonthClass = calculateStatsTrain(train, normalNumCols, nonNormalNumCols, discreteNumCols, Seq("Location","DayMonth","RainTomorrow"))
  val statsLocClass      = calculateStatsTrain(train, normalNumCols, nonNormalNumCols, discreteNumCols, Seq("Location","RainTomorrow"))
  
  // --- Un solo join con ambas tablas ---
  val dfJoined = df
    .join(statsDayMonthClass, Seq("Location","DayMonth","RainTomorrow"), "left")
    .join(statsLocClass, Seq("Location","RainTomorrow"), "left")

  // --- Aplicar imputación jerárquica ---
  val imputadas = df.columns.map { c =>
    if (normalNumCols.contains(c)) {
      when(
        col(c).isNull,
        coalesce(col(s"${c}_mean_Location_DayMonth_RainTomorrow"), col(s"${c}_mean_Location_RainTomorrow"))
      ).otherwise(col(c)).alias(c)
    }
    else if (nonNormalNumCols.contains(c)) {
      when(
        col(c).isNull,
        coalesce(col(s"${c}_median_Location_DayMonth_RainTomorrow"), col(s"${c}_median_Location_RainTomorrow"))
      ).otherwise(col(c)).alias(c)
    }
    else if (discreteNumCols.contains(c)) {
      when(
        col(c).isNull,
        round(
          coalesce(col(s"${c}_mean_Location_DayMonth_RainTomorrow"), col(s"${c}_mean_Location_RainTomorrow"))
        ).cast("int")
      ).otherwise(col(c)).alias(c)
    }
    else col(c)
  }

  dfJoined.select(imputadas: _*)
}


def applyBatchImputeTest(df: DataFrame, train: DataFrame): DataFrame = {
  // --- Precalcular estadísticas ---
  val statsDayMonth = calculateStatsTest(train, normalNumCols, nonNormalNumCols, discreteNumCols, Seq("Location","DayMonth"))
  val statsLoc      = calculateStatsTest(train, normalNumCols, nonNormalNumCols, discreteNumCols, Seq("Location"))
  
  // --- Un solo join con ambas tablas ---
  val dfJoined = df
    .join(statsDayMonth, Seq("Location","DayMonth"), "left")
    .join(statsLoc, Seq("Location"), "left")

  // --- Aplicar imputación jerárquica ---
  val imputadas = df.columns.map { c =>
    if (normalNumCols.contains(c)) {
      when(
        col(c).isNull,
        coalesce(col(s"${c}_mean_Location_DayMonth"), col(s"${c}_mean_Location"))
      ).otherwise(col(c)).alias(c)
    }
    else if (nonNormalNumCols.contains(c)) {
      when(
        col(c).isNull,
        coalesce(col(s"${c}_median_Location_DayMonth"), col(s"${c}_median_Location"))
      ).otherwise(col(c)).alias(c)
    }
    else if (discreteNumCols.contains(c)) {
      when(
        col(c).isNull,
        round(
          coalesce(col(s"${c}_mean_Location_DayMonth"), col(s"${c}_mean_Location"))
        ).cast("int")
      ).otherwise(col(c)).alias(c)
    }
    else col(c)
  }

  dfJoined.select(imputadas: _*)
}








def applyRainTodayImputeTrain(df: DataFrame, train: DataFrame): DataFrame = {
  val modeDF = calculateRainTodayModeForTrain(train)
  df.withColumn("RainToday", upper(trim(col("RainToday")))).join(modeDF, Seq("RainTomorrow"), "left").withColumn("RainToday", when(isMissingStr(col("RainToday")), col("RainToday_mode")).otherwise(col("RainToday"))).drop("RainToday_mode")
}

def applyRainTodayImputeTest(df: DataFrame): DataFrame = {
  df.withColumn("RainToday", upper(trim(col("RainToday")))).withColumn("RainToday_pred", when(col("Rainfall") > 0.0, lit("Yes")).otherwise(lit("No"))).groupBy("DayMonth","RainToday_pred").agg(count(lit(1)).alias("cnt")).groupBy("DayMonth").agg(expr("max_by(RainToday_pred, cnt)").alias("RainToday_mode")).join(df.withColumn("RainToday", upper(trim(col("RainToday")))).withColumn("RainToday_pred", when(col("Rainfall") > 0.0, lit("Yes")).otherwise(lit("No"))), Seq("DayMonth"), "right").withColumn("RainToday", when(isMissingStr(col("RainToday")), coalesce(col("RainToday_pred"), col("RainToday_mode"))).otherwise(col("RainToday"))).drop("RainToday_pred","RainToday_mode","cnt")
}

def applyWindImputeTrain(df: DataFrame, train: DataFrame): DataFrame = {
  println("  → Imputando viento (UN SOLO JOIN)...")
  val modesDF = calculateWindModesForTrain(train)
  val dfJoined = df.join(modesDF, Seq("Location","DayMonth","RainTomorrow"), "left")  // ← Agregado DayMonth
  val imputedCols = df.columns.map { c => if (windDirCols.contains(c)) when(isMissingStr(col(c)), col(s"${c}_mode")).otherwise(col(c)).alias(c) else col(c) }
  dfJoined.select(imputedCols: _*)
}

def applyWindImputeTest(df: DataFrame, train: DataFrame): DataFrame = {
  println("  → Imputando viento (UN SOLO JOIN - Test)...")
  val modesDF = calculateWindModesForTest(train)
  val dfJoined = df.join(modesDF, Seq("Location","DayMonth"), "left")  // ← Agregado DayMonth
  val imputedCols = df.columns.map { c => if (windDirCols.contains(c)) when(isMissingStr(col(c)), col(s"${c}_mode")).otherwise(col(c)).alias(c) else col(c) }
  dfJoined.select(imputedCols: _*)
}


// ============================================================
// 11) Revisión de valores nulos y "NA"
// ============================================================
def revisarNulosYNA(df: DataFrame, nombre: String): Unit = {
  println(s"\n  Revisión de valores nulos y 'NA' en $nombre")
  println("=" * 60)

  // Contar nulos y tokens "NA"/"N/A"/"NULL"/"NONE"/"MISSING"/"-"/"?"
  val tokens = Seq("na","n/a","null","none","missing","-","?")
  val resumen = df.columns.map { c =>
    val nulos = df.filter(col(c).isNull).count()
    val nas   = df.filter(lower(trim(col(c))).isin(tokens: _*)).count()
    (c, nulos, nas, nulos + nas)
  }

  // Mostrar columnas con valores faltantes
  val dfResumen = resumen.toSeq.toDF("Columna","Nulos","NA_texto","Total_faltantes")
    .filter(col("Total_faltantes") > 0)
    .orderBy(desc("Total_faltantes"))

  if (dfResumen.count() == 0)
    println(s" No se encontraron valores nulos o 'NA' en $nombre.")
  else {
    println(s"  Columnas con valores faltantes en $nombre:")
    dfResumen.show(truncate = false)
  }
}

// Llamadas al control




// === 9) PROCESO PRINCIPAL ===
revisarNulosYNA(trainN, "TRAIN INICIAL")
revisarNulosYNA(testN , "TEST INICIAL")


println("\n" + "="*70)
println("PROCESO DE IMPUTACIÓN")
println("="*70)

println("\n Paso 1: Imputación numérica (1 JOIN)")
//val trainNumeric = applyBatchImputeTrain(trainN, trainN).localCheckpoint(eager=true)
//val testNumeric = applyBatchImputeTest(testN, trainN).localCheckpoint(eager=true)

val trainNumeric = applyBatchImputeTrain(trainN, trainN)
val testNumeric  = applyBatchImputeTest(testN, trainN)

println(" Completado\n")

println(" Paso 2: Imputación RainToday (1 JOIN)")
val trainRain = applyRainTodayImputeTrain(trainNumeric, trainNumeric)
val testRain = applyRainTodayImputeTest(testNumeric)
println(" Completado\n")

println(" Paso 3: Imputación viento (1 JOIN)")
val trainFinal = applyWindImputeTrain(trainRain, trainRain).persist()
val testFinal = applyWindImputeTest(testRain, trainRain).persist()
println(" Completado\n")

// === 10) Resumen ===
println("="*70)
println("RESUMEN FINAL")
println("="*70)
val trainCount = trainFinal.count()
val testCount = testFinal.count()
println(s" Train: ${trainCount} | Test: ${testCount}")
println("\n--- Train Sample ---")
trainFinal.select("Date","Location","MinTemp","MaxTemp","RainToday","RainTomorrow").show(5, false)
println("\n--- Test Sample ---")
testFinal.select("Date","Location","MinTemp","MaxTemp","RainToday","RainTomorrow").show(5, false)


revisarNulosYNA(trainFinal, "TRAIN FINAL")
revisarNulosYNA(testFinal , "TEST FINAL")



trainN.unpersist()
testN.unpersist()

println("\n" + "="*70)
println(" PROCESO COMPLETADO")
 