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
def dist(df: DataFrame, name: String): Unit = {
  val total = df.count()
  println(s"\nDistribución en $name (N=$total):")
  df.groupBy("RainTomorrow").count().withColumn("pct", round(col("count") * 100.0 / lit(total), 2)).orderBy(desc("count")).show(false)
}

// Conteos globales y distribución
println(s"Train: ${trainDF.count()}  |  Test: ${testDF.count()}")
dist(dfLabel, "Global (antes del split)")
dist(trainDF, "Train")
dist(testDF, "Test")
  
  