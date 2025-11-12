import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, Column}

val RFModel = RandomForestClassificationModel.load("./ModeloRF")


def prepararTestDF(testDF:DataFrame): DataFrame ={
    val va = new VectorAssembler().setOutputCol("features").setInputCols(testSel.columns.diff(Array("RainTomorrow")))
    // Cambiar la label
    val indiceClase= new StringIndexer().setInputCol("RainTomorrow").setOutputCol("label").setStringOrderType("alphabetDesc")

    val rainFeaturesDF = va.transform(testSel).select("features","RainTomorrow")

    return indiceClase.fit(rainFeaturesDF).transform(rainFeaturesDF).drop("RainTomorrow")
}



val rainTestDF = prepararTestDF(testSel)

val predicciones = RFModel.transform(rainTestDF)

val predictionsAndLabelsDF_RF = predicciones.select("prediction", "label")


import org.apache.spark.ml.linalg.Vector


// Tasa de error, des. estándar e intervalo de confianza
val multiEvaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
val tasa_acierto = multiEvaluator.evaluate(predicciones)
val tasa_error = 1.0 - tasa_acierto

val tasaError = 1- accuracy.evaluate(predicciones)
println("Tasa de error: " + tasaError)

val metrics = new MulticlassMetrics(predictionsAndLabelsDF_RT)
val accuracy = metrics.accuracy
val tasaError = accuracy // Es accuracy ya que la clase dominante es la negativa
println("Tasa de error: " + tasaError)

val N =predicciones.count()
val errorStd =  math.sqrt(tasaError * (1 - tasaError) / N)
println("Desviación típica del error: " + errorStd)

val z = 1.96                // z para 95%

val errorStd = math.sqrt(p * (1 - p) / n)
val icInf = tasaError - z * errorStd
val icSup = tasaError + z * errorStd

println(f"Intervalo de confianza 95%%: [${icInf}%.4f , ${icSup}%.4f]")

val errores = predictionsAndLabelsDF_RF.map(x => if (x(0) == x(1)) 0 else 1).collect.sum

val tasaDeError = errores.toDouble/predictionsAndLabels.count

// Matriz de confusión

// Usamos RRDs para construir la matriz de confusión
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionsAndLabels = predicciones.select("prediction","label").rdd.map{row => (row.getDouble(0), row.getDouble(1))}
val metrics = new MulticlassMetrics(predictionsAndLabels)
// Matriz de confusión

val confusionMatrix = metrics.confusionMatrix

println("Matriz de confusión:\n" + confusionMatrix)

val tn = confusionMatrix(0, 0)
val fp = confusionMatrix(0, 1)
val fn = confusionMatrix(1, 0)
val tp = confusionMatrix(1, 1)
// PAra los siguientes apartados, neceistamos las equitquetas orginales ( 0 == "Yes", 1 == "No"))
import org.apache.spark.ml.feature.IndexToString

val labelsDF= predicciones.select("label").distinct
val converter = new IndexToString().setInputCol("label").setOutputCol("Clase original")
val clasesDF = converter.transform(labelsDF)
println(f"%nAsociación índices-etiquetas:")
clasesDF.show
// 0 == Yes  1 == No
//Tasa de aciertos positivos 
//
val labels = Array(0.0,1.0)

// Tasa de ciertos positivos
val multiEvaluator = new MulticlassClassificationEvaluator().setMetricName("truePositiveRateByLabel")

println(f"%nTasa de ciertos positivos por etiqueta:")
labels.foreach {l =>
              multiEvaluator.setMetricLabel(l)
              val tp = multiEvaluator.evaluate(predicciones)
              println(f"truePositiveRateByLabel($l) = $tp%1.4f")}

// Ciertos positivos ponderados
multiEvaluator.setMetricName("weightedTruePositiveRate")

// Tasa de falsos positivos
val multiEvaluator = new MulticlassClassificationEvaluator().setMetricName("falsePositiveRateByLabel")

println(f"%nTasa de falsos positivos por etiqueta:")
labels.foreach {l =>
              multiEvaluator.setMetricLabel(l)
              val fp = multiEvaluator.evaluate(predicciones)
              println(f"falsePositiveRateByLabel($l) = $fp%1.4f")}

// Falsos positivos ponderados
multiEvaluator.setMetricName("weightedFalsePositiveRate")


// AUC-ROC
val binaryMetrics = new BinaryClassificationEvaluator().setRawPredictionCol("probability")
val ML_aucROC = binaryMetrics.evaluate(predicciones)
println(f"%nAUC de la curva ROC (con ML):  $ML_aucROC%1.4f%n")


val prediccionesRDD = predicciones.rdd.map(row => (row.getDouble(4), row.getDouble(0)))


/* Y los puntos de la curva ROC */
// Curva ROC conMLlib
val MLlib_curvaROC =MLlib_binarymetrics.roc

val probabilitiesAndLabelsRDD = predicciones.select("label", "probability").rdd.map{row => (row.getAs[Vector](1).toArray, row.getDouble(0))}.map{r => ( r._1(1), r._2)}
println(f"%nRDD probabilitiesAndLabels:")
probabilitiesAndLabelsRDD.take(5).foreach(x => println(x))

val MLlib_binarymetrics = new BinaryClassificationMetrics(probabilitiesAndLabelsRDD,17)

/* Calculamos Área bajo la curva ROC, auROC      */
//  AUC=0.83
val MLlib_auROC = MLlib_binarymetrics.areaUnderROC
// val MLlib_aucROC = binaryMetrics.areaUnderROC
println(f"%nAUC de la curva ROC (17 bins):  $MLlib_aucROC%1.4f%n")

// Curva ROC
val MLlib_curvaROC =MLlib_binarymetrics.roc
println("Puntos para construir curva ROC con con 17 bins:")
MLlib_curvaROC.take(17).foreach(x => println(x))



// Area bajo curva PR
val MLlib_auPR = MLlib_binarymetrics.areaUnderPR
// val MLlib_aucROC = binaryMetrics.areaUnderROC
println(f"%nAUC de la curva PR (17 bins):  $MLlib_auPR%1.4f%n")










