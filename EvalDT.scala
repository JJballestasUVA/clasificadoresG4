import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics 
import org.apache.spark.sql.{DataFrame, Column}

import org.apache.spark.mllib.evaluation.MulticlassMetrics

val DTModel = DecisionTreeClassificationModel.load("./ModeloDT")


def prepararTestDF(testDF:DataFrame): DataFrame ={
    val va = new VectorAssembler().setOutputCol("features").setInputCols(testSel.columns.diff(Array("RainTomorrow")))
    // Cambiar la label
    val indiceClase= new StringIndexer().setInputCol("RainTomorrow").setOutputCol("label").setStringOrderType("alphabetDesc")

    val rainFeaturesDF = va.transform(testSel).select("features","RainTomorrow")

    return indiceClase.fit(rainFeaturesDF).transform(rainFeaturesDF).drop("RainTomorrow")
}



val rainTestDF = prepararTestDF(testSel)

val predicciones = DTModel.transform(rainTestDF)
val predictionsAndLabelsDF_DT = predicciones.select("prediction", "label")


import org.apache.spark.ml.linalg.Vector


// Tasa de error, des. estándar e intervalo de confianza
val multiEvaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val tasaError = 1- multiEvaluator.evaluate(predicciones)
println("Tasa de error: " + tasaError)

val N =predicciones.count()
val errorStd =  math.sqrt(tasaError * (1 - tasaError) / N)
println("Desviación típica del error: " + errorStd)

val icInf = tasaError - z * errorStd
val icSup = tasaError + z * errorStd

println(f"Intervalo de confianza 95%%: [${icInf}%.4f , ${icSup}%.4f]")


// Matriz de confusión

// Usamos RRDs para construir la matriz de confusión
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
multiEvaluator.setMetricName("truePositiveRateByLabel")

println(f"%nTasa de ciertos positivos por etiqueta:")
labels.foreach {l =>
              multiEvaluator.setMetricLabel(l)
              val tp = multiEvaluator.evaluate(predicciones)
              println(f"truePositiveRateByLabel($l) = $tp%1.4f")}

//
// Calculamos truePositiveRate ponderada
//
//
multiEvaluator.setMetricName("weightedTruePositiveRate")

val ponderada = multiEvaluator.evaluate(predicciones)
println(f"%nTasa de ciertos positivos ponderada: $ponderada%1.4f")

// Tasa de falsos positivos
//
// Calculamos falsePositiveRateByLabel  para cada etiqueta
//
//
multiEvaluator.setMetricName("falsePositiveRateByLabel")

println(f"%nTasa de falsos positivos por etiqueta:")
labels.foreach {l =>
              multiEvaluator.setMetricLabel(l)
              val fp = multiEvaluator.evaluate(predicciones)
              println(f"falsePositiveRateByLabel($l) = $fp%1.4f")}

//
// Calculamos falsePositiveRate ponderada
//
//
multiEvaluator.setMetricName("weightedFalsePositiveRate")

val ponderada =multiEvaluator.evaluate(predicciones)
println(f"%nTasa de falsos positivos ponderada: $ponderada%1.4f")


// AUC-ROC
val binaryMetrics = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
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
println(f"%nAUC de la curva PR:  $MLlib_auPR%1.4f%n")










