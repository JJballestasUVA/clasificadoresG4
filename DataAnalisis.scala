import org.apache.log4j.{Level, Logger}
// SQL base
import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._     // col, when, lit, cos, sin, sqrt, udf, etc.

// ML
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{Imputer, StringIndexer, VectorAssembler, StandardScaler}
import org.apache.spark.ml.linalg.{Vector, Matrix}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.feature.QuantileDiscretizer

Logger.getLogger("org.apache.spark.scheduler.DAGScheduler").setLevel(Level.ERROR)

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import java.nio.file.{Files, Paths}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// ========================================
// CONFIGURACIÓN INICIAL
// ========================================
 
 
val PATH = "."
val FILE = "weatherAUS.csv"
 

println("\n=== CARGANDO DEL DATASET ===")
   
 val weatherDF = spark.read.option("header", "true").option("inferSchema", "true").option("delimiter", ",").csv(s"$PATH/$FILE")
 
// Verificar que se cargó correctamente
println("=== INFORMACIÓN GENERAL DEL DATASET ===")
weatherDF.printSchema()
println(s"Total de columnas: ${weatherDF.columns.length}")
weatherDF.show(5)

// Cache del DataFrame para mejorar rendimiento
weatherDF.cache()
val totalReg = weatherDF.count()
println(s"Total de registros: $totalReg")

 
// ========================================
// 1) ANÁLISIS DE VALORES VACÍOS/NULOS
// ========================================
// Se define una función para identificar valores ausentes: nulos, vacíos o equivalentes textuales como "NA", "null", etc.
// Se calcula el número de valores faltantes por columna y porcentaje respecto al total de registros.
// Se determina cuántas filas están completamente llenas y cuántas contienen al menos un valor ausente.
// Este análisis forma parte de la evaluación de calidad de datos y sirve como base para el resumen ejecutivo.

println("\n=== ANÁLISIS DE VALORES VACÍOS/NULOS ===")


def esVacioONulo(c: Column): Column =
  c.isNull || lower(trim(c.cast("string"))).isin("","na","nan","null","n/a","none","missing")


 

// Conteo de nulos por columna
val nullStats = weatherDF.columns.map { colName =>
  (colName, weatherDF.filter(esVacioONulo(col(colName))).count())
}.sortBy(-_._2)

// Mostrar estadísticas de nulos

println("\nColumnas con valores vacíos/nulos (ordenadas por cantidad):")
nullStats.foreach { case (colName, count) =>
  val pct = (count * 100.0) / totalReg
  println(f"  $colName%-20s: $count%6d (${pct}%.2f%%)")
}

// Análisis de filas completas vs incompletas
val filasCompletas = weatherDF.filter(
  weatherDF.columns.map(c => !esVacioONulo(col(c))).reduce(_ && _)
).count()
val filasIncompletas = totalReg - filasCompletas
 
 
// ========================================
// 2) VALIDACIÓN DE VALORES CATEGÓRICOS
// ========================================
// Se validan columnas categóricas con valores esperados:
//   - Columnas tipo Yes/No ("RainToday", "RainTomorrow") se comparan contra los valores válidos "Yes" y "No".
//   - Columnas de dirección del viento se verifican contra los 16 rumbos cardinales estándar.
// Se identifican valores inválidos y se reporta la cantidad de registros afectados por cada caso.



println("\n=== VALIDACIÓN DE VALORES CATEGÓRICOS ===")

// Validación Yes/No
val colsYesNo = Seq("RainToday", "RainTomorrow")
val valoresEsperadosYN = Set("Yes", "No")

println("Validación de columnas Yes/No:")
colsYesNo.foreach { colName =>
  val valoresUnicos = weatherDF
    .select(col(colName))
    .filter(col(colName).isNotNull)
    .distinct()
    .collect()
    .map(_.getString(0))
  
  val invalidos = valoresUnicos.filterNot(valoresEsperadosYN.contains)
  
  if (invalidos.nonEmpty) {
    println(s"  $colName tiene valores inválidos: ${invalidos.mkString(", ")}")
    val count = weatherDF.filter(col(colName).isin(invalidos: _*)).count()
    println(s"    Registros afectados: $count")
  } else {
    println(s"  $colName: OK (solo Yes/No)")
  }
}

// Validación direcciones del viento
val windDirCols = Seq("WindGustDir", "WindDir9am", "WindDir3pm")
val direccionesValidas = Set("N","NNE","NE","ENE","E","ESE","SE","SSE",
                             "S","SSW","SW","WSW","W","WNW","NW","NNW")

println("\nValidación de direcciones del viento:")
windDirCols.foreach { colName =>
  val valoresUnicos = weatherDF
    .select(col(colName))
    .filter(col(colName).isNotNull)
    .distinct()
    .collect()
    .map(_.getString(0))
    .toSet
  
  val invalidos = valoresUnicos -- direccionesValidas
  
  if (invalidos.nonEmpty) {
    println(s"  $colName tiene direcciones inválidas: ${invalidos.mkString(", ")}")
    val count = weatherDF.filter(col(colName).isin(invalidos.toSeq: _*)).count()
    println(s"    Registros afectados: $count")
  } else {
    println(s"  $colName: OK")
  }
}


 
// ========================================
// 3) VALIDACIÓN DE RANGOS NUMÉRICOS
// ========================================
// Se definen rangos físicos y plausibles para atributos numéricos clave del dataset.
// Se verifica si existen valores fuera de estos rangos (out-of-range) y se contabilizan por columna.
// Se reporta el número de registros afectados por cada variable, o se confirma que todos los valores están dentro de los límites esperados.
// Este análisis forma parte de la detección preliminar de outliers y validación de calidad de datos.

println("\n=== VALIDACIÓN DE RANGOS NUMÉRICOS ===")

val rangosValidos = Map(
  "Rainfall"       -> (0.0, 1000.0),
  "Evaporation"    -> (0.0, 500.0),
  "Sunshine"       -> (0.0, 24.0),
  "WindGustSpeed"  -> (0.0, 200.0),
  "WindSpeed9am"   -> (0.0, 200.0),
  "WindSpeed3pm"   -> (0.0, 200.0),
  "Humidity9am"    -> (0.0, 100.0),
  "Humidity3pm"    -> (0.0, 100.0),
  "Cloud9am"       -> (0.0, 9.0),
  "Cloud3pm"       -> (0.0, 9.0),
  "Pressure9am"    -> (800.0, 1100.0),
  "Pressure3pm"    -> (800.0, 1100.0),
  "Temp9am"        -> (-50.0, 60.0),
  "Temp3pm"        -> (-50.0, 60.0),
  "MinTemp"        -> (-50.0, 60.0),
  "MaxTemp"        -> (-50.0, 60.0)
)

println("Columnas con valores fuera de rango:")
var totalFueraRango = 0L

rangosValidos.foreach { case (colName, (min, max)) =>
  
val cD = col(colName).cast(DoubleType)
val fueraRango = weatherDF.filter(cD.isNotNull && (cD < lit(min) || cD > lit(max))).count()
  
  
  if (fueraRango > 0) {
    println(f"  $colName%-20s: $fueraRango%6d registros fuera de [$min%.1f, $max%.1f]")
    totalFueraRango += fueraRango
  }
}

if (totalFueraRango == 0) {
  println("  Todos los valores numéricos están dentro de rangos válidos")
}

println("\n=== VALIDACIÓN DE FECHAS ===")
// ========================================
// 4) VALIDACIÓN DE FECHAS
// ========================================
// Se verifica si la columna "Date" puede ser correctamente parseada al formato yyyy-MM-dd.
// Se identifican registros con fechas inválidas o no parseables y se muestran ejemplos.
// Se calcula el rango temporal del dataset (fecha mínima y máxima).
// Este análisis ayuda a detectar errores de formato y evaluar la cobertura temporal de los datos.

val dateIsString = weatherDF.schema("Date").dataType == StringType


val dateDF = if (dateIsString) {
  weatherDF.withColumn("Date_tmp", to_date(col("Date"), "yyyy-MM-dd"))
} else {
  weatherDF.withColumn("Date_tmp", col("Date").cast("date"))
}

// Validar fechas usando dateDF
val invalidDate = if (dateIsString) {
  dateDF.filter(col("Date").isNotNull && col("Date_tmp").isNull)
} else {
  dateDF.filter(col("Date").isNull)
}



val fechasInvalidas = invalidDate.count()
println(s"Registros con fecha no parseable (formato yyyy-MM-dd): $fechasInvalidas")

// Mostrar ejemplos de fechas inválidas si existen
if (fechasInvalidas > 0) {
  println("Ejemplos de fechas inválidas:")
  invalidDate.select("Date", "Location").show(20, false)
}

// Análisis de rango temporal  
 

val rangoStats = dateDF.agg(
  min("Date_tmp").as("primera_fecha"),
  max("Date_tmp").as("ultima_fecha")
).collect()(0)


val primeraFecha = rangoStats.getAs[java.sql.Date]("primera_fecha")
val ultimaFecha = rangoStats.getAs[java.sql.Date]("ultima_fecha")



println(s"Rango de fechas: $primeraFecha hasta $ultimaFecha")



// ========================================
// 5) ANÁLISIS DE DATOS BOOLEANOS
// ========================================
// Se identifican columnas booleanas, ya sean tipadas como BooleanType o con valores equivalentes (Yes/No, true/false, 0/1).
// Para cada columna detectada:
//   - Se listan los valores únicos encontrados junto con su frecuencia.
//   - Se calcula el porcentaje de registros clasificados como True/Yes, False/No y otros valores atípicos.
// Este análisis permite evaluar la consistencia de las variables binarias y su distribución por clases,
// incluyendo las columnas "RainToday" y "RainTomorrow".
 


println("\n=== ANÁLISIS DE DATOS BOOLEANOS ===")

// Detectar columnas booleanas (tipadas como BooleanType o equivalentes 0/1, true/false, Yes/No)
val booleanCols = weatherDF.schema.fields.collect {
  case f if f.dataType == BooleanType => f.name
}.toSeq ++ Seq("RainToday", "RainTomorrow") // añadimos manualmente si Spark las infirió como string

if (booleanCols.nonEmpty) {
  println(s"Columnas booleanas detectadas: ${booleanCols.mkString(", ")}")
  booleanCols.foreach { colName =>
    println(s"\n--- Análisis de columna: $colName ---")
    
    // Contar valores únicos (true/false/otros)
    val resumen = weatherDF
      .groupBy(col(colName))
      .agg(count("*").alias("frecuencia"))
      .orderBy(desc("frecuencia"))
    
    resumen.show(false)
    
    // Porcentaje de True vs False
    val total = weatherDF.filter(col(colName).isNotNull).count()
	val trueCount  = weatherDF.filter(lower(col(colName).cast("string")) === "true" || lower(col(colName).cast("string")) === "yes" || col(colName) === lit(1)).count()
	val falseCount = weatherDF.filter(lower(col(colName).cast("string")) === "false" || lower(col(colName).cast("string")) === "no"  || col(colName) === lit(0)).count()

	
	
    val otros = total - trueCount - falseCount
    
    println(f"True/Yes: ${trueCount * 100.0 / total}%.2f%%  |  False/No: ${falseCount * 100.0 / total}%.2f%%  |  Otros: ${otros * 100.0 / total}%.2f%%")
  }
} else {
  println("No se detectaron columnas booleanas en el dataset.")
}








// ========================================
// 6) ANÁLISIS DE DUPLICADOS
// ========================================
// Se calcula el número total de registros duplicados exactos en el DataFrame.
// Se determina el número de registros únicos eliminando duplicados.
// Se imprime el total de registros, los únicos, los duplicados y el porcentaje de duplicación.
// Si existen duplicados, se muestran los 5 ejemplos más frecuentes junto con su recuento.


println("\n=== ANÁLISIS DE DUPLICADOS ===")

val registrosUnicos = weatherDF.dropDuplicates().count()
val duplicadosExactos = totalReg - registrosUnicos

 
println(s"Registros únicos: $registrosUnicos")
println(s"Registros duplicados (copias adicionales): $duplicadosExactos")
println(f"Porcentaje de duplicación: ${duplicadosExactos * 100.0 / totalReg}%.2f%%")

// Mostrar ejemplos de duplicados si existen
if (duplicadosExactos > 0) {
  println("\nEjemplos de registros duplicados (top 5):")
  weatherDF
    .groupBy(weatherDF.columns.map(col): _*)
    .count()
    .filter(col("count") > 1)
    .orderBy(desc("count"))
    .select("count", "Date", "Location", "MinTemp", "MaxTemp", "Rainfall", "RainToday", "RainTomorrow")
    .show(5, false)
}



// ========================================
// 7) ESTADÍSTICOS BÁSICOS: MÍNIMO, MÁXIMO, MEDIA Y DESVIACIÓN ESTÁNDAR
// ========================================
// Se seleccionan los atributos numéricos del DataFrame y se convierten a tipo Double para evitar errores.
// Se calculan los estadísticos básicos (mínimo, máximo, media y desviación estándar) ignorando valores nulos.
// Los resultados se muestran formateados por consola y también como un DataFrame ordenado por nombre de atributo.
 


println("\n=== ESTADÍSTICOS BÁSICOS: MÍNIMO, MÁXIMO, MEDIA Y DESVIACIÓN ESTÁNDAR ===")


val atributosNumericos = Seq(
  "MinTemp", "MaxTemp", "Temp9am", "Temp3pm",
  "Rainfall", "Evaporation", "Sunshine",
  "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
  "Humidity9am", "Humidity3pm",
  "Pressure9am", "Pressure3pm",
  "Cloud9am", "Cloud3pm"
)

// Cast seguro a Double para evitar errores de tipo
val weatherNumeric = atributosNumericos.foldLeft(weatherDF) { (df, c) =>
  if (df.columns.contains(c)) df.withColumn(c, col(c).cast(DoubleType)) else df
}

// Calcular estadísticos básicos ignorando nulos
val exprs = atributosNumericos.flatMap { c =>
  Seq(
    min(col(c)).alias(s"${c}_min"),
    max(col(c)).alias(s"${c}_max"),
    mean(col(c)).alias(s"${c}_mean"),
    stddev(col(c)).alias(s"${c}_stddev")
  )
}

val estadisticos = weatherNumeric.select(exprs: _*).collect()(0)

// Mostrar resultados formateados
println("Resumen estadístico de los atributos numéricos:\n")
atributosNumericos.foreach { c =>
  val minVal = Option(estadisticos.getAs[Any](s"${c}_min")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val maxVal = Option(estadisticos.getAs[Any](s"${c}_max")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val media  = Option(estadisticos.getAs[Any](s"${c}_mean")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val desv   = Option(estadisticos.getAs[Any](s"${c}_stddev")).map(_.toString.toDouble).getOrElse(Double.NaN)
  println(f"  $c%-15s  Mín: $minVal%8.3f  Máx: $maxVal%8.3f  Media: $media%8.3f  Desv.Est.: $desv%8.3f")
}

// Versión DataFrame ordenada
val dfEstadisticos = atributosNumericos.map { c =>
  val minVal = Option(estadisticos.getAs[Any](s"${c}_min")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val maxVal = Option(estadisticos.getAs[Any](s"${c}_max")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val media  = Option(estadisticos.getAs[Any](s"${c}_mean")).map(_.toString.toDouble).getOrElse(Double.NaN)
  val desv   = Option(estadisticos.getAs[Any](s"${c}_stddev")).map(_.toString.toDouble).getOrElse(Double.NaN)
  (c, minVal, maxVal, media, desv)
}.toDF("Atributo", "Minimo", "Maximo", "Media", "DesviacionEstandar")

dfEstadisticos.orderBy("Atributo").show(truncate = false)



// ========================================
// 8) RESUMEN EJECUTIVO
// ========================================
// Se calcula un score de calidad general basado en el porcentaje de filas completas y duplicados exactos.
// Se resumen los principales problemas de calidad detectados: incompletitud, duplicados, fechas inválidas y valores fuera de rango.
// Se identifican las 5 columnas con mayor proporción de valores ausentes.
// Finalmente, se libera la caché del DataFrame y se marca el fin del análisis.
// [Memoria] resume problemas detectados (ausentes, etc.).


println("\n=== RESUMEN EJECUTIVO DE CALIDAD DE DATOS ===")



 
 
// ========================================
// 9) CORRELACIONES Y VISTA RÁPIDA DE VARIABLES
// ========================================


 




println("\n=== CORRELACIONES BÁSICAS (PEARSON) ===")

// 1) Filtramos filas con etiqueta válida
val corrBase = weatherDF.filter(col("RainTomorrow").isin("Yes","No"))

// 2) Creamos versión numérica de la etiqueta (0/1)
val corrDF = corrBase.withColumn("RainTomorrow_num", when(col("RainTomorrow") === "Yes", 1.0).otherwise(0.0))

// 3) Casteamos a double las columnas que vamos a usar
val corrDFnum = atributosNumericos.foldLeft(corrDF) { (df, c) =>
  df.withColumn(c, col(c).cast("double"))
}

// 4) Quitamos cualquier fila con null en las columnas a correlacionar
val corrInputDF = corrDFnum.na.drop(atributosNumericos)

// 5) VectorAssembler para la matriz de correlación
val corrAssembler = new VectorAssembler()
  .setInputCols(atributosNumericos.toArray)
  .setOutputCol("corrFeatures")

val corrVecDF = corrAssembler.transform(corrInputDF).select("corrFeatures")

// 6) Matriz de correlación de Pearson entre las variables numéricas
val corrRow = Correlation.corr(corrVecDF, "corrFeatures", "pearson").head
val corrMatrix = corrRow.getAs[Matrix](0)

def printCorrMatrix(
    cols: Seq[String],
    m: Matrix,
    decimals: Int = 3,
    minWidth: Int = 7,
    maxName: Int = 9   // recorta nombres largos
): Unit = {
  // recortamos nombres para que no se hagan eternos
  val shortCols = cols.map { n =>
    if (n.length > maxName) n.take(maxName - 1) + "…" else n
  }

  // ancho fijo: segun minWidth
  val cellWidth = minWidth

  def pad(s: String): String =
    s.padTo(cellWidth, ' ').mkString

  val fmt = "%1." + decimals + "f"

  println()
  // cabecera
  println(pad("") + shortCols.map(c => pad(c)).mkString)

  for (i <- shortCols.indices) {
    val rowVals = shortCols.indices.map { j =>
      val v = m(i, j)
      val vs = String.format(fmt, Double.box(v))
      pad(vs)
    }.mkString
    println(pad(shortCols(i)) + rowVals)
  }
  println()
}

println(s"\nMatriz de correlación calculada sobre ${corrVecDF.count()} filas válidas.")
println(s"Tamaño de la matriz: ${corrMatrix.numRows} x ${corrMatrix.numCols}")

printCorrMatrix(atributosNumericos, corrMatrix, decimals = 3, minWidth = 7, maxName = 7)


// === Pearson corr(feature, RainTomorrow_num) | tabla feature↔label ===
val withLabel = corrDF  // ya contiene RainTomorrow_num = 0/1

val corrFeatLabel: Seq[(String, Double)] =
  atributosNumericos.flatMap { c =>
    val df = withLabel.select(col(c).cast("double").as(c), col("RainTomorrow_num")).na.drop(Seq(c, "RainTomorrow_num"))
    if (df.head(1).nonEmpty) Some(c -> df.stat.corr(c, "RainTomorrow_num"))
    else None
  }.sortBy { case (_, r) => -math.abs(r) }

println("\n=== Pearson corr(feature, RainTomorrow_num) | ordenado por |corr| desc ===")
corrFeatLabel.foreach { case (feat, r) =>
  println(f"$feat%-15s  corr = $r%+1.4f")
}


 

val porcentajeCompleto = (filasCompletas * 100.0 / totalReg)
val calidadScore = porcentajeCompleto * (1 - (duplicadosExactos.toDouble / totalReg))

println(f"Score de calidad general: $calidadScore%.2f / 100")
println("\nPrincipales problemas encontrados:")
println(f"  • Filas incompletas: $filasIncompletas (${filasIncompletas * 100.0 / totalReg}%.2f%%)")
println(f"  • Duplicados exactos: $duplicadosExactos (${duplicadosExactos * 100.0 / totalReg}%.2f%%)")

if (fechasInvalidas > 0) {
  println(s"  • Fechas inválidas: $fechasInvalidas")
}
if (totalFueraRango > 0) {
  println(s"  • Valores numéricos fuera de rango: $totalFueraRango casos")
}

// Columnas más problemáticas (top 15 con más nulos)
println("\nColumnas con más valores faltantes (top 15):")
nullStats.take(15).foreach { case (colName, count) =>
  val pct = (count * 100.0) / totalReg
  println(f"  • $colName%-20s: $pct%.2f%% faltante")
}
 








// === (C) Percentiles por columna numérica (approxQuantile) ===
val probs = Array(0.01, 0.05, 0.50, 0.95, 0.99)
val relErr = 0.01
println("\n=== Percentiles (P1,P5,P50,P95,P99) ===")
atributosNumericos.foreach { c =>
  val colD = weatherDF.select(col(c).cast("double").as(c)).na.drop(Seq(c))
  if (colD.head(1).nonEmpty) {
    val qs = colD.stat.approxQuantile(c, probs, relErr)
    println(f"$c%-12s -> P1=${qs(0)}%8.3f  P5=${qs(1)}%8.3f  P50=${qs(2)}%8.3f  P95=${qs(3)}%8.3f  P99=${qs(4)}%8.3f")
  }
}





// === (D) Co-ocurrencia de nulos entre columnas problemáticas ===
val topNullCols = nullStats.take(5).map(_._1).toSeq
 
println("\n=== Co-ocurrencia de nulos (pares) en top-5 columnas con más faltantes ===")
for {
  i <- topNullCols.indices
  j <- (i+1) until topNullCols.length
} {
  val a = topNullCols(i); val b = topNullCols(j)
  val both = weatherDF.filter(esVacioONulo (col(a)) && esVacioONulo (col(b))).count()
  val pct  = both * 100.0 / totalReg
  println(f"$a%-15s & $b%-15s : $both%8d (${pct}%.2f%%)")
}






// === (E) Balance de clases y baseline ===
 
val cls = corrBase.groupBy("RainTomorrow").count().orderBy(desc("count")).collect()

if (cls.nonEmpty) {
  val totalYN = cls.map(_.getLong(1)).sum
  val (majLabel, majCount) = (cls(0).getString(0), cls(0).getLong(1))
  val majAcc = majCount.toDouble / totalYN
  println("\n=== Balance de clases RainTomorrow ===")
  cls.foreach(r => println(f"  ${r.getString(0)}: ${r.getLong(1)} (${"%1.2f".format(r.getLong(1)*100.0/totalYN)}%%)"))
  println(f"Baseline (siempre '${majLabel}'): ${majAcc*100}%.2f%% accuracy")
}



// === (F) Cardinalidad y Top-k categorías ===
val catCols = Seq("Location","WindGustDir","WindDir9am","WindDir3pm").filter(weatherDF.columns.contains)
val k = 10
catCols.foreach { c =>
  val nonNull = weatherDF.filter(col(c).isNotNull)
  val card = nonNull.select(c).distinct().count()
  println(s"\n=== $c | cardinalidad: $card ===")
  nonNull.groupBy(c).count().orderBy(desc("count")).show(k, truncate=false)
}

//===  categorías con baja frecuencia === 

val rareThresh = 50  
Seq("Location","WindGustDir","WindDir9am","WindDir3pm").filter(weatherDF.columns.contains).foreach { c =>
  val rares = weatherDF.groupBy(col(c)).count().filter(col("count") < rareThresh).orderBy(asc("count"))
  val nRares = rares.count()
  println(s"\n$c – categorías raras (<$rareThresh): $nRares")
  if (nRares > 0) rares.show(10, false)
}



// === (G) Tasa de lluvia por mes y por Location (exploratorio) === 
 
  
val dfDate = dateDF.withColumn("Month", month(col("Date_tmp"))).filter(col("RainTomorrow").isin("Yes","No"))
dfDate.printSchema()
dfDate.select("Date", "Date_tmp", "Month", "RainTomorrow").show(5, false)


println("\n=== Tasa de RainTomorrow=Yes por mes ===")
dfDate.groupBy("Month").agg(avg(when(col("RainTomorrow")==="Yes",1).otherwise(0)).alias("rain_rate")).orderBy("Month").show(12, false)

if (dfDate.columns.contains("Location")) {
  println("\n=== Tasa de RainTomorrow=Yes por Location (top-10 por nº de registros) ===")
  val topLoc = dfDate.groupBy("Location").count().orderBy(desc("count")).limit(10).select("Location").as[String].collect().toSet
  
  dfDate.filter(col("Location").isin(topLoc.toSeq:_*)).groupBy("Location").agg(avg(when(col("RainTomorrow")==="Yes",1).otherwise(0)).alias("rain_rate"), count("*").alias("n")).orderBy(desc("n")).show(10, false)
}
 
// === (B) Pares con colinealidad alta (|r| ≥ 0.90) ===
val umbral = 0.90
val altos = for {
  i <- atributosNumericos.indices
  j <- (i+1) until atributosNumericos.length
  r = corrMatrix(i, j)
  if math.abs(r) >= umbral
} yield (atributosNumericos(i), atributosNumericos(j), r)

println(s"\n=== Pares con |r| ≥ $umbral ===")
if (altos.isEmpty) {
  println("No se detectaron pares con colinealidad alta.")
} else {
  altos
    .sortBy { case (_, _, r) => -math.abs(r) }
    .foreach { case (a, b, r) =>
      println(f"$a%-12s ~ $b%-12s : r = $r%+1.4f")
      
    }
}



println(f"\n• Filas usadas en Pearson (matriz): ${corrVecDF.count()} / $totalReg = ${corrVecDF.count()*100.0/totalReg}%.2f%%")
println("• Top-10 corr(feature, RainTomorrow_num): " + corrFeatLabel.take(10).map{ case(k,v) => f"$k=$v%+.3f" }.mkString(", "))
// Liberar cache

 weatherDF.unpersist()