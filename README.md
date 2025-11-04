# clasificadoresG4
Weather in Australia — Limpieza, validación y feature engineering con Spark

Script en Scala (Spark 3.5.x) para auditar calidad de datos meteorológicos de Australia, imputar ausentes de forma robusta (incluida imputación circular para direcciones de viento), realizar un split estratificado 80/20 por RainTomorrow, y generar conjuntos de features continuas estandarizadas y discretizadas listos para modelado. 

RainInAustralia

Qué hace

Carga y cacheo del dataset weatherAUS.csv desde una ruta configurable. 

RainInAustralia

Auditoría de calidad: nulos, vacíos, valores fuera de rango y fechas inválidas; duplicados, estadísticos básicos y resumen ejecutivo con score de calidad. 

RainInAustralia

Validaciones de categóricas: Yes/No y direcciones de viento (16 rumbos válidos). 

RainInAustralia

Split estratificado por RainTomorrow (80/20) manteniendo distribución de clases. 

RainInAustralia

Imputación numérica: por mes (media/mediana según distribución), vecinos lag/lead por Location, y fallback Location+Month. 

RainInAustralia

Imputación circular de viento: pasa rumbos a ángulos → vectores unitarios (ux, uy), promedia con vecinos y fallback media circular mensual; descarta vectores (0,0). 

RainInAustralia

Limpieza final: filtra nubes (=9) y outliers de Evaporation; reporta métricas NTAIE/NTOE y tasa de no clasificados en test. 

RainInAustralia

Feature engineering: agregados diarios (temperatura, presión, humedad, viento, nubes), codificación cíclica de Month, discretización por cuantiles (≤10 bins), estandarización y assemblers para pipelines lineales y NB/árboles. 

RainInAustralia

Entradas y salidas

Entrada: weatherAUS.csv (con columnas como Date, Location, temperaturas, viento, humedad, etc.). Ruta configurable con PATH/FILE. 

RainInAustralia

Salidas intermedias: trainDF, testDF (estratificados) → trainFinal, testFinal (limpios) → datasets transformados para modelos:

train_for_linear / test_for_linear (features continuas estandarizadas)

train_for_nb_or_trees / test_for_nb_or_trees (features discretizadas) 

RainInAustralia

Requisitos

Apache Spark 3.5.x y Scala 2.12 (código estilo spark-shell/notebook; no objeto main).
