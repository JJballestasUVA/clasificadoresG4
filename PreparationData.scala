import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
Logger.getLogger("org.apache.spark.scheduler.DAGScheduler").setLevel(Level.ERROR)

val PATH = "."
val FILE = "weatherAUS.csv"

val weatherDF = spark.read.option("header", "true").option("inferSchema", "true").option("delimiter", ",").csv(s"$PATH/$FILE")

weatherDF.cache()
 

//esta parte se quita al unir los archivos


 
// ===== PARTICIÓN ESTRATIFICADA POR RainTomorrow (80/20) =====
// Se realiza un particionado estratificado de los datos en función de la variable objetivo "RainTomorrow":
//   1) Se filtran únicamente los registros con etiquetas válidas ("Yes"/"No").
//   2) Se separan los datos en dos subconjuntos según la clase (Yes y No).
//   3) Cada subconjunto se divide aleatoriamente en 80% entrenamiento y 20% test, usando una semilla fija para reproducibilidad.
//   4) Se recomponen los conjuntos de entrenamiento y test uniendo las particiones de cada clase, garantizando la misma proporción de etiquetas.
 
// Finalmente, se muestran conteos y distribuciones de clases en los conjuntos global, train y test para verificar la estratificación.
// [Memoria] “particionado estratificado 80/20 por RainTomorrow” con semilla. 
 

val seed = 2025L
val split = Array(0.8, 0.2)

// 1) Mantener solo registros con etiqueta válida ("Yes"/"No"), ignorando nulos/NA
val dfLabel = weatherDF.filter(
  lower(col("RainTomorrow")).isin("yes", "no")
)

weatherDF.unpersist()

// 2) Separación por clase
val dfYes = dfLabel.filter(lower(col("RainTomorrow")) === "yes")
val dfNo  = dfLabel.filter(lower(col("RainTomorrow")) === "no")

// 3) División aleatoria controlada dentro de cada clase (80/20)
val Array(yesTrain, yesTest) = dfYes.randomSplit(split, seed)
val Array(noTrain,  noTest)  = dfNo.randomSplit(split,  seed)

// 4) Recomposición estratificada
val trainDF = yesTrain.unionByName(noTrain)
val testDF  = yesTest.unionByName(noTest)

 

// --------- CHECKS DE VERIFICACIÓN ----------
// Se define una función auxiliar dist que muestra la distribución de la variable objetivo "RainTomorrow"
// en un DataFrame dado, incluyendo el número de registros y el porcentaje por clase.
// Se imprimen los tamaños de los conjuntos de entrenamiento y test.
// Finalmente, se muestran las distribuciones de clases en:
//   - El dataset global (antes del split).
//   - El conjunto de entrenamiento (Train).
//   - El conjunto de prueba (Test).
// Esto permite comprobar que la partición estratificada mantiene las proporciones originales de las clases.



// Función auxiliar para ver distribución de clases con porcentajes
def distributionView(df: DataFrame, name: String): Unit = {
  val total = df.count()
  println(s"\nDistribución en $name (N=$total):")
  df.groupBy("RainTomorrow").count().withColumn("pct", round(col("count") * 100.0 / lit(total), 2)).orderBy(desc("count")).show(false)
}

// Conteos globales y distribución
println(s" Conteos globales y distribución Train: ${trainDF.count()}  |  Test: ${testDF.count()}")
distributionView(dfLabel, "Global (antes del split)")
distributionView(trainDF, "Train")
distributionView(testDF, "Test")
  
    
  
  //(NTAIE, Número Total de Ausentes/Inconsistentes Eliminados
  //(NTOE, Número Total de Outliers Eliminados). 
  //NTAIE+NTOE (que es la tasa de error que proporcionan las métricas) 
  //tasa de no clasificados: (NTAIE+NTOE)/(tamaño conjunto de prueba). 
  //La tasa de no clasificados debe ser al menos un orden de magnitud menor que la tasa de error.
  
  
  
def normalizeLabelAndBasics(df: DataFrame): DataFrame = {
  val base =
    if (df.schema("Date").dataType != DateType) df.withColumn("Date", to_date(col("Date")))
    else df

  val stdYN = Seq("RainTomorrow").foldLeft(base){ (acc,c) =>
    if (acc.columns.contains(c))
      acc.withColumn(c,
        when(lower(trim(col(c))) === "yes","Yes")
        .when(lower(trim(col(c))) === "no","No")
        .otherwise(col(c)))
    else acc
  }

  stdYN.withColumn("Month", month(col("Date")))
}

// (A) Aplica limpieza  PRE-split
val df0 = normalizeLabelAndBasics(dfLabel)
  .transform{ d =>
    val missingTokens = Seq("NA","N/A","NULL","NONE","MISSING","-","?")
    Seq("WindGustDir","WindDir9am","WindDir3pm").foldLeft(d){ (acc,c) =>
      if (acc.columns.contains(c)) {
        val v = trim(upper(col(c)))
        acc.withColumn(c, when(v.isin(missingTokens: _*), lit(null:String)).otherwise(v))
      } else acc
    }
  }
 
// (B) Filtra filas con etiqueta válida
val dfClean = df0.filter(col("RainTomorrow").isin("Yes","No"))
 
// (C) Estratificación por clase
val dfYes = dfClean.filter(col("RainTomorrow") === "Yes")
val dfNo  = dfClean.filter(col("RainTomorrow") === "No")

val Array(yesTrain, yesTest) = dfYes.randomSplit(Array(0.8, 0.2), seed)
val Array(noTrain,  noTest)  = dfNo.randomSplit (Array(0.8, 0.2), seed)

val trainDF = yesTrain.unionByName(noTrain)
val testDF  = yesTest.unionByName(noTest)

// (D) Normalizaciones restantes deterministas (si faltó algo de formateo)

def normalizeRest(df: DataFrame): DataFrame = {
  val upWind = Seq("WindGustDir","WindDir9am","WindDir3pm").foldLeft(df){ (acc,c) =>
    if (acc.columns.contains(c)) acc.withColumn(c, upper(trim(col(c)))) else acc
  }
  // Aquí NO se hace nada que “aprenda” del conjunto
  upWind
}

val trainN = normalizeRest(trainDF)
val testN  = normalizeRest(testDF)

  

println(s"Train: ${trainN.count()}  |  Test: ${testN.count()}")
distributionView(dfClean, "Global (antes del split, ya normalizado)")
distributionView(trainN, "Train  normalizado")
distributionView(testN,  "Test  normalizado")

  println(s"Se clasifican las columnas numéricas en dos grupos según su distribución esperada:")
  println(s" - normalNumCols: variables con distribución aproximadamente normal (ej. temperaturas, presión)")
  println(s"  - nonNormalNumCols: variables con distribución no normal o sesgada (ej. lluvia, humedad, viento, nubes)")
  println(s" Además, se identifican columnas categóricas de viento (direcciones)")
 
  val normalNumCols = Seq("MinTemp","MaxTemp","Temp9am","Temp3pm","Pressure9am","Pressure3pm")
  .filter(trainN.columns.contains)
val nonNormalNumCols = Seq(
  "Rainfall","Evaporation","Sunshine","Humidity9am","Humidity3pm",
  "Cloud9am","Cloud3pm","WindGustSpeed","WindSpeed9am","WindSpeed3pm"
).filter(trainN.columns.contains)
val windCatCols = Seq("WindGustDir","WindDir9am","WindDir3pm").filter(trainN.columns.contains)


  println(s" Se Definen Funciones Para Calcular Moda Promedios Mediana Basados en la Clase")
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



def applyClassImpute(df: DataFrame, train: DataFrame): DataFrame = {
  // 1. Imputación para columnas con distribución "Normal" (usando la media por clase)
  val withNormal = normalNumCols.foldLeft(df){ (acc,c) =>
    // Calcula la media de la columna 'c' agrupada por 'RainTomorrow'
    val rtMean = classMean(train.filter(col(c).isNotNull), c)

    // Junta las medias calculadas por 'RainTomorrow' al DataFrame actual
    acc.join(rtMean, Seq("RainTomorrow"), "left")
        // Reemplaza los nulos en 'c' con la media de su clase
        .withColumn(c, when(col(c).isNull, col(s"${c}_rt_mean")).otherwise(col(c)))
        // Elimina la columna de la media auxiliar después de la imputación
        .drop(s"${c}_rt_mean")
  }

  // 2. Imputación para columnas con distribución "No Normal" (usando la mediana por clase)
  val withNonNormal = nonNormalNumCols.foldLeft(withNormal){ (acc,c) =>
    // Calcula la mediana de la columna 'c' agrupada por 'RainTomorrow'
    val rtMedian = classMedian(train.filter(col(c).isNotNull), c)

    // Junta las medianas calculadas por 'RainTomorrow' al DataFrame actual
    acc.join(rtMedian, Seq("RainTomorrow"), "left")
        // Reemplaza los nulos en 'c' con la mediana de su clase
        .withColumn(c, when(col(c).isNull, col(s"${c}_rt_median")).otherwise(col(c)))
        // Elimina la columna de la mediana auxiliar
        .drop(s"${c}_rt_median")
  }

  withNonNormal
}

val trainM1 = applyClassImpute(trainN, trainN)
val testM1  = applyClassImpute(testN,  trainN)

val trainM1 = applyClassImpute(trainN, trainN)
val testM1  = applyClassImpute(testN,  trainN)
