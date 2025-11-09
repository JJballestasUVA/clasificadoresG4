
import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator}


// .....................................
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row




:load ./TransformacionDeDatos.scala


// Partición de los datos (para tener conjunto de prueba para evaluar los árboles)

/*
// Del ejercicio individual, sabemos que el atributo maxBins no es muy relevante en el modelo,
// y que maxDepth lo es, teniendo los mejores arboles entre 5 y 9.
*/
  

// Construccion del modelo
val dtSimple = new DecisionTreeClassifier()
/* Por si se prefiere usar F1
val evaluatorF1 = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("f1")
*/
// AUC 

// Construccion del Evaluator
val evaluatorAUC = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("probability")
    .setMetricName("areaUnderROC")

// Construcción del Grid
val paramGrid = new ParamGridBuilder().addGrid(dtSimple.impurity, Array("gini", "entropy")).addGrid(dtSimple.maxDepth, Array(3,5,7,9)).build()

// Construccion del CrossValidator
val cv = new CrossValidator()
  .setEstimator(dtSimple)
  .setEvaluator(evaluatorAUC)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)  // Use 3+ in practice (o 5)
  .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel


val cvModel = cv.fit(rainDF)

//  Nos quedamos con el mejor modelo
//
val bestModel = cvModel.bestModel.asInstanceOf[DecisionTreeClassificationModel]

bestModel.extractParamMap().toSeq.foreach(println) // Observamos que el arbol escgido tiene maxDepth = 9 y impurity = gini

// Volvemos a repetir con otro grid. Esta vez con maxBins y maxDepth alrededor de 5

val newParamGrid = new ParamGridBuilder().addGrid(dtSimple.maxBins, Array(28,32,36)).addGrid(dtSimple.maxDepth, Array(bestModel.getMaxDepth-1,bestModel.getMaxDepth,bestModel.getMaxDepth+1)).build()
val cv = new CrossValidator()
  .setEstimator(dtSimple)
  .setEvaluator(evaluatorAUC)
  .setEstimatorParamMaps(newParamGrid)
  .setNumFolds(5)  // Use 3+ in practice (o 5)
  .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

val cvModel = cv.fit(rainDF)

val bestModel = cvModel.bestModel.asInstanceOf[DecisionTreeClassificationModel]

bestModel.extractParamMap().toSeq.foreach(println) // Observamos que el arbol escgido tiene maxDepth = 5 y maxBins =32
