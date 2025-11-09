import org.apache.spark.ml.feature.{Imputer, StringIndexer, VectorAssembler, StandardScaler}
import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.functions.{udf, col, cos, sin, sqrt, when, lit}
import org.apache.spark.sql.types.DoubleType
import spark.implicits._
import org.apache.spark.sql.types.IntegerType

:load ./PreparationData.scala

// Quitar cuando PreparationData esté completo
val trainFinal = trainM1
val testFinal = testM1

val trainBase = trainFinal.withColumn("Rainfall",col("Rainfall").cast("double"))
val testBase  = testFinal.withColumn("Rainfall",col("Rainfall").cast("double"))




val temperatura = Seq("MaxTemp", "MinTemp", "Temp9am", "Temp3pm")
val presion = Seq("Pressure9am", "Pressure3pm")
val humedad = Seq("Humidity9am", "Humidity3pm")
val velViento = Seq("WindGustspeed", "WindSpeed9am", "WindSpeed3pm")

val windVecCols =
  Seq("WindGustDir_ux","WindGustDir_uy","WindDir9am_ux","WindDir9am_uy","WindDir3pm_ux","WindDir3pm_uy")


val categorias = Seq(temperatura, presion, humedad,velViento)


// Crea en el Dataframe las nuevas columnas que se usarán pasa 
def crearMediasMedidas(df: DataFrame): DataFrame = {

  val withTemps = if (df.columns.contains("Temp9am") && df.columns.contains("Temp3pm"))
    df.withColumn("Temp_avg", (col("MaxTemp") + col("Temp9am") + col("Temp3pm") + col("MinTemp") ) / 4.0)
  else df

  val withPress = if (df.columns.contains("Pressure9am") && df.columns.contains("Pressure3pm"))
    withTemps.withColumn("Pressure_avg", (col("Pressure9am") + col("Pressure3pm")) / 2.0)
  else withTemps

  val withHum = if (df.columns.contains("Humidity9am") && df.columns.contains("Humidity3pm"))
    withPress.withColumn("Humidity_avg", (col("Humidity9am") + col("Humidity3pm")) / 2.0)
  else withPress

  val withWindS = if (df.columns.contains("WindSpeed9am") && df.columns.contains("WindSpeed3pm"))
    withHum.withColumn("WindSpeed_avg", (col("WindGustSpeed") + col("WindSpeed9am") + col("WindSpeed3pm")) / 3.0)
  else withHum

  withWindS
}


val trainAgg = crearMediasMedidas(trainBase)
val testAgg  = crearMediasMedidas(testBase)

// Eliminar atributos no necesarios
val dropCols = Seq(
  "RainToday", "Location", "Date", "Sunshine",
  "MinTemp","MaxTemp","Temp9am","Temp3pm",      // Agrupado en WindSpeed_avg
  "Pressure9am","Pressure3pm",                  // Agrupado en Pressure_avg
  "Humidity9am","Humidity3pm",                  // Agrupado en Humidity_avg
  "WindGustSpeed","WindSpeed9am","WindSpeed3pm", // Agrupado en WindSpeed_avg
  "Cloud9am","Cloud3pm",                        // Deamiados nulos
  "WindGustDir","WindDir9am","WindDir3pm",      // TODO: Agrupar
  "Month","Day", "Year",
  "Evaporation"                                 // Demasiados nulos,
)

val trainSel = trainAgg.drop(dropCols:_*)
val testSel  = testAgg.drop(dropCols:_*)



val va = new VectorAssembler().setOutputCol("features").setInputCols(trainSel.columns.diff(Array("RainTomorrow")))

val rainFeaturesDF = va.transform(trainSel).select("features","RainTomorrow")

// Cambiar la label
val indiceClase= new StringIndexer().setInputCol("RainTomorrow").setOutputCol("label").setStringOrderType("alphabetDesc")

val rainFeaturesLabelDF = indiceClase.fit(rainFeaturesDF).transform(rainFeaturesDF).drop("RainTomorrow")

// Partición de los datos (para tener conjunto de prueba para evaluar los árboles)
val Array(trainRainDF, testRainDF) = rainFeaturesLabelDF.randomSplit(Array(0.66, 0.34), seed=0)


trainRainDF.show(5)