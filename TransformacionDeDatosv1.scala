:load ./PreparationData.scala
 
 // ============================================================
// TRANSFORMACIÓN DE DATOS
// ============================================================

def esVacioONulo(c: Column): Column =
  c.isNull || lower(trim(c.cast("string"))).isin("","na","nan","null","n/a","none","missing")


// 1️Crear medias combinadas
def crearMediasMedidas(df: DataFrame): DataFrame = {
  df.withColumn("Temp_avg", (col("MaxTemp") + col("MinTemp") + col("Temp9am") + col("Temp3pm")) / 4.0)
    .withColumn("Pressure_avg", (col("Pressure9am") + col("Pressure3pm")) / 2.0)
    .withColumn("Humidity_avg", (col("Humidity9am") + col("Humidity3pm")) / 2.0)
    .withColumn("WindSpeed_avg", (col("WindGustSpeed") + col("WindSpeed9am") + col("WindSpeed3pm")) / 3.0)
}

// 2️Aplicar medias
val trainAgg = crearMediasMedidas(trainFinal)
val testAgg  = crearMediasMedidas(testFinal)

// 3️Eliminar ambigüedad en DayMonth
val trainAggClean = trainAgg.withColumnRenamed("DayMonth", "DayMonth_tmp").drop("DayMonth").withColumnRenamed("DayMonth_tmp", "DayMonth")

val testAggClean = testAgg.withColumnRenamed("DayMonth", "DayMonth_tmp").drop("DayMonth").withColumnRenamed("DayMonth_tmp", "DayMonth")

// 4️Seleccionar columnas relevantes
val dropCols = Seq(
  "RainToday", "Location", "Date", "Sunshine",
  "MinTemp","MaxTemp","Temp9am","Temp3pm",
  "Pressure9am","Pressure3pm",
  "Humidity9am","Humidity3pm",
  "WindGustSpeed","WindSpeed9am","WindSpeed3pm",
  "Cloud9am","Cloud3pm",
  "WindGustDir","WindDir9am","WindDir3pm",
  "Month","Day","Year", "DayMonth",
  "Evaporation"
)

val trainSel = trainAggClean.drop(dropCols: _*)
val testSel  = testAggClean.drop(dropCols: _*)

// 6️VectorAssembler
val va = new VectorAssembler()
  .setOutputCol("features")
  .setInputCols(trainSel.columns.diff(Array("RainTomorrow")))

val rainFeaturesDF = va.transform(trainSel).select("features", "RainTomorrow")

// 7️Indexar clase
val indiceClase = new StringIndexer().setInputCol("RainTomorrow").setOutputCol("label").setStringOrderType("alphabetDesc")

val rainDF = indiceClase.fit(rainFeaturesDF).transform(rainFeaturesDF).drop("RainTomorrow")

rainDF.show(5)
/* // No es necesario ya que CrossValidation hace la particion
val Array(trainRainDF, testRainDF) = rainDF.randomSplit(Array(0.66, 0.34), seed=0)
val validationDF = testRainDF
*/
 
 
 