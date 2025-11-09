import org.apache.spark.ml.classification.DecisionTreeClassifier


:load ./TransformacionDeDatos.scala


val decisionTree= new DecisionTreeClassifier().setLabelCol("label")
    .setFeaturesCol("features")
    .setCacheNodeIds(false)

val DTRainModel1 = decisionTree.fit(trainRainDF)