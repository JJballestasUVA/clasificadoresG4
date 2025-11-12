// ============================================================
// VERSIÓN FINAL CORREGIDA - PreparationDatav4R8
// Imputación secundaria aplicada a TODAS las columnas
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

// === 2) Casting ===
val atributosDouble = Seq("MinTemp","MaxTemp","Temp9am","Temp3pm","Rainfall","Evaporation","Sunshine","Pressure9am","Pressure3pm","Humidity9am","Humidity3pm","WindGustSpeed","WindSpeed9am","WindSpeed3pm")
val atributosInt = Seq("Cloud9am","Cloud3pm")
val dfPrepared = weatherDF.select(weatherDF.columns.map { c => if (atributosDouble.contains(c)) col(c).cast("double").alias(c) else if (atributosInt.contains(c)) col(c).cast("int").alias(c) else col(c) }: _*)

// === 3) Utilidades ===
def isMissingStr(c: Column): Column = c.isNull || trim(c) === "" || lower(trim(c)).isin("na","nan","null","n/a","none","missing")
def isNullNum(c: Column): Column = c.isNull

// === 4) Normalización ===
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

// === 6.1) Outliers ===
val outlierCols = Seq("Cloud9am", "Cloud3pm", "Rainfall", "Evaporation","WindGustSpeed", "WindSpeed9am", "WindSpeed3pm")

def computeIQRBounds(df: DataFrame, colName: String): (Double, Double) = {
  val quantiles = df.stat.approxQuantile(colName, Array(0.25, 0.75), 0.05)
  val q1 = quantiles(0)
  val q3 = quantiles(1)
  val iqr = q3 - q1
  (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
}

def removeOutliers(df: DataFrame, cols: Seq[String]): DataFrame = {
  var result = df
  cols.foreach { c =>
    val (lower, upper) = computeIQRBounds(df.filter(col(c).isNotNull), c)
    println(f"→ $c%-15s | IQR: [$lower%.3f , $upper%.3f]")
    result = result.withColumn(c, when(col(c) < lower || col(c) > upper, lit(null).cast(df.schema(c).dataType)).otherwise(col(c)))
  }
  result
}

// === 7) Funciones de agregación ===
def p50(colName: String): Column = expr(s"percentile_approx($colName, 0.5)")

def calculateAllStatsForTrain(train: DataFrame, numCols: Seq[String], discCols: Seq[String]): DataFrame = {
  val stats = numCols.flatMap(c => Seq(avg(col(c)).alias(s"${c}_mean"), p50(c).alias(s"${c}_median"))) ++ discCols.map(c => avg(col(c)).alias(s"${c}_mean"))
  train.groupBy("Location","DayMonth","RainTomorrow").agg(stats.head, stats.tail: _*)
}

def calculateAllStatsForTest(train: DataFrame, numCols: Seq[String], discCols: Seq[String]): DataFrame = {
  val stats = numCols.flatMap(c => Seq(avg(col(c)).alias(s"${c}_mean"), p50(c).alias(s"${c}_median"))) ++ discCols.map(c => avg(col(c)).alias(s"${c}_mean"))
  train.groupBy("Location","DayMonth").agg(stats.head, stats.tail: _*)
}

def calculateWindModesForTrain(train: DataFrame): DataFrame = {
  windDirCols.map { c => train.filter(col(c).isNotNull && !isMissingStr(col(c))).groupBy("Location","DayMonth","RainTomorrow",c).agg(count(lit(1)).alias("cnt")).groupBy("Location","DayMonth","RainTomorrow").agg(expr(s"max_by($c, cnt)").alias(s"${c}_mode")) }.reduce((df1,df2) => df1.join(df2, Seq("Location","DayMonth","RainTomorrow"), "outer"))
}

def calculateWindModesForTest(train: DataFrame): DataFrame = {
  windDirCols.map { c => train.filter(col(c).isNotNull && !isMissingStr(col(c))).groupBy("Location","DayMonth",c).agg(count(lit(1)).alias("cnt")).groupBy("Location","DayMonth").agg(expr(s"max_by($c, cnt)").alias(s"${c}_mode")) }.reduce((df1,df2) => df1.join(df2, Seq("Location","DayMonth"), "outer"))
}

def calculateRainTodayModeForTrain(train: DataFrame): DataFrame = {
  train.withColumn("RainToday", upper(trim(col("RainToday")))).filter(!isMissingStr(col("RainToday"))).groupBy("RainTomorrow","RainToday").agg(count(lit(1)).alias("cnt")).groupBy("RainTomorrow").agg(expr("max_by(RainToday, cnt)").alias("RainToday_mode"))
}

// === 8) Imputación principal ===
def applyBatchImputeTrain(df: DataFrame, train: DataFrame): DataFrame = {
  val allCols = normalNumCols ++ nonNormalNumCols
  val columnsToSelect = Seq("Location", "DayMonth", "RainTomorrow") ++ allCols
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
  val columnsToSelect = Seq("Location", "DayMonth") ++ allCols
  val statsDF = calculateAllStatsForTest(
    train.select(columnsToSelect.map(col): _*).filter(allCols.map(c => col(c).isNotNull).reduce(_ || _)),
    normalNumCols ++ nonNormalNumCols.diff(discreteNumCols),
    discreteNumCols
  )
  val dfJoined = df.join(statsDF, Seq("Location","DayMonth"), "left")
  val imputedCols = df.columns.map { c =>
    if (normalNumCols.contains(c)) when(col(c).isNull, col(s"${c}_mean")).otherwise(col(c)).alias(c)
    else if (nonNormalNumCols.diff(discreteNumCols).contains(c)) when(col(c).isNull, col(s"${c}_median")).otherwise(col(c)).alias(c)
    else if (discreteNumCols.contains(c)) when(col(c).isNull, round(coalesce(col(s"${c}_mean"), lit(0))).cast("int")).otherwise(col(c)).alias(c)
    else col(c)
  }
  dfJoined.select(imputedCols: _*)
}

def applyRainTodayImputeTrain(df: DataFrame, train: DataFrame): DataFrame = {
  val modeDF = calculateRainTodayModeForTrain(train)
  df.withColumn("RainToday", upper(trim(col("RainToday")))).join(modeDF, Seq("RainTomorrow"), "left").withColumn("RainToday", when(isMissingStr(col("RainToday")), col("RainToday_mode")).otherwise(col("RainToday"))).drop("RainToday_mode")
}

def applyRainTodayImputeTest(df: DataFrame): DataFrame = {
  df.withColumn("RainToday", upper(trim(col("RainToday")))).withColumn("RainToday_pred", when(col("Rainfall") > 0.0, lit("Yes")).otherwise(lit("No"))).groupBy("DayMonth","RainToday_pred").agg(count(lit(1)).alias("cnt")).groupBy("DayMonth").agg(expr("max_by(RainToday_pred, cnt)").alias("RainToday_mode")).join(df.withColumn("RainToday", upper(trim(col("RainToday")))).withColumn("RainToday_pred", when(col("Rainfall") > 0.0, lit("Yes")).otherwise(lit("No"))), Seq("DayMonth"), "right").withColumn("RainToday", when(isMissingStr(col("RainToday")), coalesce(col("RainToday_pred"), col("RainToday_mode"))).otherwise(col("RainToday"))).drop("RainToday_pred","RainToday_mode","cnt")
}

def applyWindImputeTrain(df: DataFrame, train: DataFrame): DataFrame = {
  val modesDF = calculateWindModesForTrain(train)
  val dfJoined = df.join(modesDF, Seq("Location","DayMonth","RainTomorrow"), "left")
  val imputedCols = df.columns.map { c => if (windDirCols.contains(c)) when(isMissingStr(col(c)), col(s"${c}_mode")).otherwise(col(c)).alias(c) else col(c) }
  dfJoined.select(imputedCols: _*)
}

def applyWindImputeTest(df: DataFrame, train: DataFrame): DataFrame = {
  val modesDF = calculateWindModesForTest(train)
  val dfJoined = df.join(modesDF, Seq("Location","DayMonth"), "left")
  val imputedCols = df.columns.map { c => if (windDirCols.contains(c)) when(isMissingStr(col(c)), col(s"${c}_mode")).otherwise(col(c)).alias(c) else col(c) }
  dfJoined.select(imputedCols: _*)
}

// === 9) Imputación secundaria (SIN Location) - TODAS LAS COLUMNAS ===

def calculateSecondaryStatsForTrain(df: DataFrame, normalCols: Seq[String], nonNormalCols: Seq[String], discCols: Seq[String]): DataFrame = {
  // Medianas para variables normales
  val normalMedianAggs = normalCols.map { c => expr(s"percentile_approx($c, 0.5)").alias(s"${c}_median_sec") }
  // Medianas para variables no normales (excepto discretas)
  val nonNormalMedianAggs = nonNormalCols.diff(discCols).map { c => expr(s"percentile_approx($c, 0.5)").alias(s"${c}_median_sec") }
  // Medias para variables discretas
  val meanAggs = discCols.map { c => avg(col(c)).alias(s"${c}_mean_sec") }
  
  val allAggs = normalMedianAggs ++ nonNormalMedianAggs ++ meanAggs
  val allNumCols = normalCols ++ nonNormalCols
  
  df.filter(allNumCols.map(c => col(c).isNotNull).reduce(_ || _))
    .groupBy("RainTomorrow")
    .agg(allAggs.head, allAggs.tail: _*)
}

def calculateSecondaryWindModeForTrain(df: DataFrame, windCols: Seq[String]): DataFrame = {
  val modeDFs = windCols.map { c =>
    df.filter(col(c).isNotNull)
      .groupBy("RainTomorrow", c)
      .agg(count(lit(1)).alias("cnt"))
      .groupBy("RainTomorrow")
      .agg(expr(s"max_by($c, cnt)").alias(s"${c}_mode_sec"))
  }
  modeDFs.reduce((df1, df2) => df1.join(df2, Seq("RainTomorrow"), "outer"))
}

def applySecondaryImputationTrain(df: DataFrame, train: DataFrame): DataFrame = {
  println("  → Imputación secundaria (RainTomorrow) - TODAS las columnas...")
  
  val statsDF = calculateSecondaryStatsForTrain(train, normalNumCols, nonNormalNumCols, discreteNumCols)
  val windModeDF = calculateSecondaryWindModeForTrain(train, windDirCols)
  
  val dfJoined = df.join(statsDF, Seq("RainTomorrow"), "left").join(windModeDF, Seq("RainTomorrow"), "left")
  
  val imputedCols = df.columns.map { c =>
    if (normalNumCols.contains(c)) {
      // Variables normales: mediana
      when(col(c).isNull, col(s"${c}_median_sec")).otherwise(col(c)).alias(c)
    }
    else if (nonNormalNumCols.diff(discreteNumCols).contains(c)) {
      // Variables no normales (excepto discretas): mediana
      when(col(c).isNull, col(s"${c}_median_sec")).otherwise(col(c)).alias(c)
    }
    else if (discreteNumCols.contains(c)) {
      // Variables discretas: media redondeada
      when(col(c).isNull, round(coalesce(col(s"${c}_mean_sec"), lit(0))).cast("int")).otherwise(col(c)).alias(c)
    }
    else if (windDirCols.contains(c)) {
      // Direcciones de viento: moda
      when(col(c).isNull, col(s"${c}_mode_sec")).otherwise(col(c)).alias(c)
    }
    else col(c)
  }
  
  dfJoined.select(imputedCols: _*)
}

def calculateSecondaryStatsForTest(df: DataFrame, normalCols: Seq[String], nonNormalCols: Seq[String], discCols: Seq[String]): DataFrame = {
  val dfWithRainToday = df.withColumn("RainToday_inferred", when(col("Rainfall") > 0.0, lit("Yes")).otherwise(lit("No")))
  
  // Medianas para variables normales
  val normalMedianAggs = normalCols.map { c => expr(s"percentile_approx($c, 0.5)").alias(s"${c}_median_sec") }
  // Medianas para variables no normales (excepto discretas)
  val nonNormalMedianAggs = nonNormalCols.diff(discCols).map { c => expr(s"percentile_approx($c, 0.5)").alias(s"${c}_median_sec") }
  // Medias para variables discretas
  val meanAggs = discCols.map { c => avg(col(c)).alias(s"${c}_mean_sec") }
  
  val allAggs = normalMedianAggs ++ nonNormalMedianAggs ++ meanAggs
  val allNumCols = normalCols ++ nonNormalCols
  
  dfWithRainToday
    .filter(allNumCols.map(c => col(c).isNotNull).reduce(_ || _))
    .groupBy("RainToday_inferred")
    .agg(allAggs.head, allAggs.tail: _*)
}

def calculateSecondaryWindModeForTest(df: DataFrame, windCols: Seq[String]): DataFrame = {
  val dfWithRainToday = df.withColumn("RainToday_inferred", when(col("Rainfall") > 0.0, lit("Yes")).otherwise(lit("No")))
  
  val modeDFs = windCols.map { c =>
    dfWithRainToday
      .filter(col(c).isNotNull)
      .groupBy("RainToday_inferred", c)
      .agg(count(lit(1)).alias("cnt"))
      .groupBy("RainToday_inferred")
      .agg(expr(s"max_by($c, cnt)").alias(s"${c}_mode_sec"))
  }
  modeDFs.reduce((df1, df2) => df1.join(df2, Seq("RainToday_inferred"), "outer"))
}

def applySecondaryImputationTest(df: DataFrame, train: DataFrame): DataFrame = {
  println("  → Imputación secundaria (RainToday inferido) - TODAS las columnas...")
  
  val statsDF = calculateSecondaryStatsForTest(train, normalNumCols, nonNormalNumCols, discreteNumCols)
  val windModeDF = calculateSecondaryWindModeForTest(train, windDirCols)
  
  val dfWithRainToday = df.withColumn("RainToday_inferred", when(col("Rainfall") > 0.0, lit("Yes")).otherwise(lit("No")))
  val dfJoined = dfWithRainToday.join(statsDF, Seq("RainToday_inferred"), "left").join(windModeDF, Seq("RainToday_inferred"), "left")
  
  val imputedCols = df.columns.map { c =>
    if (normalNumCols.contains(c)) {
      // Variables normales: mediana
      when(col(c).isNull, col(s"${c}_median_sec")).otherwise(col(c)).alias(c)
    }
    else if (nonNormalNumCols.diff(discreteNumCols).contains(c)) {
      // Variables no normales (excepto discretas): mediana
      when(col(c).isNull, col(s"${c}_median_sec")).otherwise(col(c)).alias(c)
    }
    else if (discreteNumCols.contains(c)) {
      // Variables discretas: media redondeada
      when(col(c).isNull, round(coalesce(col(s"${c}_mean_sec"), lit(0))).cast("int")).otherwise(col(c)).alias(c)
    }
    else if (windDirCols.contains(c)) {
      // Direcciones de viento: moda
      when(col(c).isNull, col(s"${c}_mode_sec")).otherwise(col(c)).alias(c)
    }
    else col(c)
  }
  
  dfJoined.select(imputedCols: _*)
}

// === 10) Manejo de nulos residuales ===
def calculateGlobalStats(df: DataFrame, numCols: Seq[String], discCols: Seq[String], windCols: Seq[String]): Map[String, Any] = {
  val medians = numCols.map { c => c -> df.filter(col(c).isNotNull).stat.approxQuantile(c, Array(0.5), 0.01).headOption.getOrElse(0.0) }.toMap
  val means = discCols.map { c => c -> math.round(df.filter(col(c).isNotNull).agg(avg(col(c))).collect()(0)(0).asInstanceOf[Number].doubleValue()).toInt }.toMap
  val modes = windCols.map { c => c -> df.filter(col(c).isNotNull).groupBy(c).count().orderBy(desc("count")).limit(1).collect().headOption.map(_.getString(0)).getOrElse("N") }.toMap
  medians ++ means ++ modes
}

def applyGlobalFallback(df: DataFrame, numCols: Seq[String], discCols: Seq[String], windCols: Seq[String]): DataFrame = {
  println("  → Aplicando imputación GLOBAL para nulos residuales...")
  val globalStats = calculateGlobalStats(df, numCols, discCols, windCols)
  val imputedCols = df.columns.map { c =>
    if (numCols.contains(c)) when(col(c).isNull, lit(globalStats(c).asInstanceOf[Double])).otherwise(col(c)).alias(c)
    else if (discCols.contains(c)) when(col(c).isNull, lit(globalStats(c).asInstanceOf[Int])).otherwise(col(c)).alias(c)
    else if (windCols.contains(c)) when(col(c).isNull, lit(globalStats(c).asInstanceOf[String])).otherwise(col(c)).alias(c)
    else col(c)
  }
  df.select(imputedCols: _*)
}

// === 11) Revisión ===
def revisarNulosYNA(df: DataFrame, nombre: String): Unit = {
  println(s"\n  $nombre")
  println("=" * 60)
  val tokens = Seq("na","n/a","null","none","missing","-","?")
  val resumen = df.columns.map { c => (c, df.filter(col(c).isNull).count(), df.filter(lower(trim(col(c))).isin(tokens: _*)).count()) }.filter(r => r._2 + r._3 > 0)
  if (resumen.isEmpty) println("  ✅ No hay valores nulos") else resumen.toSeq.toDF("Columna","Nulos","NA").show(false)
}

// === 12) PROCESO PRINCIPAL ===
println("\n=== Eliminando outliers ===")
val trainCleaned = removeOutliers(trainN, outlierCols).persist()
val testCleaned = removeOutliers(testN, outlierCols).persist()

revisarNulosYNA(trainCleaned, "TRAIN INICIAL")
revisarNulosYNA(testCleaned, "TEST INICIAL")

println("\n" + "="*70)
println("PROCESO DE IMPUTACIÓN")
println("="*70)

println("\n Paso 1: Imputación numérica")
val trainNumeric = applyBatchImputeTrain(trainCleaned, trainCleaned)
val testNumeric = applyBatchImputeTest(testCleaned, trainCleaned)
println(" ✓ Completado")

println("\n Paso 2: Imputación RainToday")
val trainRain = applyRainTodayImputeTrain(trainNumeric, trainNumeric)
val testRain = applyRainTodayImputeTest(testNumeric)
println(" ✓ Completado")

println("\n Paso 3: Imputación viento")
val trainWind = applyWindImputeTrain(trainRain, trainRain)
val testWind = applyWindImputeTest(testRain, trainRain)
println(" ✓ Completado")

println("\n Paso 4: Imputación secundaria")
val trainFinal = applySecondaryImputationTrain(trainWind, trainWind).persist()
val testFinal = applySecondaryImputationTest(testWind, trainWind).persist()
 
println(" ✓ Completado")

// === Resumen ===
println("\n" + "="*70)
println("RESUMEN FINAL")
println("="*70)
println(s" Train: ${trainFinal.count()} | Test: ${testFinal.count()}")

revisarNulosYNA(trainFinal, "TRAIN FINAL")
revisarNulosYNA(testFinal, "TEST FINAL")

trainN.unpersist()
testN.unpersist()

println("\n✅ PROCESO COMPLETADO - Dataset listo para ML")