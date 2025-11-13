// GuardarDataFrames.scala
// Este script guarda los DataFrames finales preparados en formato Parquet

import org.apache.spark.sql.{DataFrame, SaveMode}

// Asume que los DataFrames trainFinal y testFinal ya están en memoria
// y que se han generado previamente en PreparationData.scala

:load PreparationData.scala
val outputPathTrain = "./datos_preparados/trainFinal"
val outputPathTest = "./datos_preparados/testFinal"

// Guardado en formato Parquet (eficiente y conserva tipos de datos)
println(s"💾 Guardando trainFinal en: $outputPathTrain")
trainFinal.write.mode(SaveMode.Overwrite).parquet(outputPathTrain)

println(s"💾 Guardando testFinal en: $outputPathTest")
testFinal.write.mode(SaveMode.Overwrite).parquet(outputPathTest)

println("\n✅ Guardado completado. Los DataFrames están listos para ser reutilizados.")
