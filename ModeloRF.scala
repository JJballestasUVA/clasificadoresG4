import org.apache.spark.ml.classification.{RandomForestClassifier,RandomForestClassificationModel}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator}
import org.apache.spark.sql.Row
/// Modelo avanzado Gradient Boosted Tree
:load ./TransformacionDeDatos.scala

val RF = new RandomForestClassifier().
setFeaturesCol("features").
setLabelCol("label").
setNumTrees(10).
setMaxDepth(7).
setMaxBins(20).
setMinInstancesPerNode(1).
setMinInfoGain(0.0).
setCacheNodeIds(false).
setCheckpointInterval(10)

// Esto sugiere que hacamos el cross validation con profundidades bajas,
// y manejando el número de iteraciones
val RFRain = new RandomForestClassifier().setNumTrees(10)

val evaluatorAUC = new BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC")

// Construcción del Grid
val paramGrid = new ParamGridBuilder().addGrid(RF.maxBins, Array(28,32,36)).addGrid(RF.maxDepth, Array(5,7,9)).build()


val cvRF = new CrossValidator()
  .setEstimator(RF)
  .setEvaluator(evaluatorAUC)
  .setEstimatorParamMaps(paramGrid)
  .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel


val cvRFModel = cvRF.fit(rainDF)


val bestModel = cvRFModel.bestModel.asInstanceOf[RandomForestClassificationModel]

bestModel.extractParamMap().toSeq.foreach(println) 

// El mejor arbol tien maxDepth=9, veamos con un grid mayor. Lo mismo con maxBins, pero con un grid menor

val paramGrid = new ParamGridBuilder().addGrid(RF.maxBins, Array(24,28,30)).addGrid(RF.maxDepth, Array(9,10,11)).build()

val cvRF = new CrossValidator()
  .setEstimator(RF)
  .setEvaluator(evaluatorAUC)
  .setEstimatorParamMaps(paramGrid)
  .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel


val cvRFModel = cvRF.fit(rainDF)


val bestModel = cvRFModel.bestModel.asInstanceOf[RandomForestClassificationModel]

bestModel.extractParamMap().toSeq.foreach(println) 

bestModel.write.overwrite().save("./ModeloRF")

