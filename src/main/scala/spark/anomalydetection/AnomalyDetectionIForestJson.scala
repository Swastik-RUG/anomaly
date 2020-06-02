package spark.anomalydetection

import com.linkedin.relevance.isolationforest.IsolationForest
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DateType, DoubleType, IntegerType, StructField, StructType}

object AnomalyDetectionIForestJson {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("lambdaint")
      .getOrCreate()

    import spark.implicits._

    val schema = StructType(
      StructField("timestamp", DateType, nullable = true) ::
        StructField("Feature1", DoubleType, nullable = true) ::
        StructField("Feature2", DoubleType, nullable = true) ::
        Nil
    )

    val inputData = spark.read.option("multiline", "true").json("src/main/resources/data/sample.json")
    println(inputData.schema.toString())
    inputData.show(false)
    val featureCols = Array( "Feature1", "Feature2")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("Rawfeatures")
    val assembledDF = assembler.transform(inputData)
    val scaler = new StandardScaler()
      .setInputCol("Rawfeatures")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    val transformedDF = scaler.fit(assembledDF).transform(assembledDF)

    transformedDF.show(false)

    val trainingDF = transformedDF.filter(col("Feature1") <= 110)

    val testingDF = transformedDF.filter(col("Feature1") > 110)

    val isf = new IsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      //.setMaxSamples(41)
      .setMaxFeatures(2)
      .setFeaturesCol("features")
      .setPredictionCol("predicted_label")
      .setScoreCol("outlier_score")
      .setContamination(0.1)
      .setRandomSeed(1)

    val model = isf.fit(trainingDF)

    // test the model with test data set
    val pDf = model.transform(testingDF)
    pDf.printSchema()
    pDf.filter($"predicted_label" =!= 1.0).show(100, truncate = false)
  }
}
