package spark.anomalydetection

import com.linkedin.relevance.isolationforest.IsolationForest
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.functions._

object AnomalyDetectionIForest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("lambda")
      .getOrCreate()

    import spark.implicits._

    val schema = StructType(
      StructField("duration", DoubleType, nullable = true) ::
        StructField("network", DoubleType, nullable = true) ::
        StructField("storage", DoubleType, nullable = true) ::
        StructField("anomaly", IntegerType, nullable = true) ::
        Nil
    )

    val inputData = spark.read.option("header", "true").schema(schema).csv("src/main/resources/data/training.csv")
    val featureCols = Array("duration", "network", "storage")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("Rawfeatures")
    val assembledDF = assembler.transform(inputData)
    val scaler = new StandardScaler()
      .setInputCol("Rawfeatures")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    val transformedDF = scaler.fit(assembledDF).transform(assembledDF)

    transformedDF.show(false)

    val trainingDF = transformedDF.filter(col("anomaly") === 0)

    val testingDF = transformedDF.filter(col("anomaly") =!= 0)

    val isf = new IsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(41)
      .setMaxFeatures(3)
      .setFeaturesCol("features")
      .setPredictionCol("predicted_label")
      .setScoreCol("outlier_score")
      .setContamination(0.1)
      .setRandomSeed(1)

    val model = isf.fit(trainingDF)

    // test the model with test data set
    val pDf = model.transform(testingDF)
    pDf.printSchema()
    pDf.show(truncate = false)
  }
}
