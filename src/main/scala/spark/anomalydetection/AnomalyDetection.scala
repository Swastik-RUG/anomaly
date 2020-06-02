package spark.anomalydetection

import com.linkedin.relevance.isolationforest.{IsolationForest, IsolationForestModel}
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.ml.Model
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import spark.anomalydetection.AnomalyDetector.{predict, training}

object AnomalyDetection {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("lambda")
      .getOrCreate()

    import spark.implicits._
    val config = ConfigFactory.load("application.conf")
    val inputDF = read(spark, config)

    //    val trainingDF = inputDF.filter(col("Feature1") <= 110)
    //    val testingDF = inputDF.filter(col("Feature1") > 110)

    detectAnomaly(spark, config, inputDF, inputDF)
  }

  def read(sparkSession: SparkSession, config: Config): DataFrame = {
    val source = scala.io.Source.fromFile(config.getString("schemafile"))
    val lines = try source.mkString finally source.close()
    val schema = DataType.fromJson(lines).asInstanceOf[StructType]
    val inputData = Utility.readDataFrame(config.getString("filetype"), sparkSession, config.getString("input_data_path"), schema)
    inputData.show(false)
    val featureCols = config.getString("features").split(";")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("Rawfeatures")
    val assembledDF = assembler.transform(inputData)
    val scaler = new StandardScaler()
      .setInputCol("Rawfeatures")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    scaler.fit(assembledDF).transform(assembledDF)
  }

  def detectAnomaly(sparkSession: SparkSession, config: Config, trainingDF: DataFrame, testingDF: DataFrame): Unit = {
    import sparkSession.implicits._

    val model: IsolationForestModel = if (config.getBoolean("training")) {
      val isf = new IsolationForest()
        .setNumEstimators(100)
        .setBootstrap(false)
        .setMaxSamples(0.8)
        .setMaxFeatures(2)
        .setFeaturesCol("features")
        .setPredictionCol("predicted_label")
        .setScoreCol("outlier_score")
        .setContamination(0.1)
        .setRandomSeed(1).fit(trainingDF)
      isf.save("src/main/resources/model/isolationTree")
      isf
    } else {
      IsolationForestModel.load("src/main/resources/model/isolationTree")
    }

    val pDf = model.transform(testingDF)
    pDf.printSchema()
    pDf.show(100, truncate = false)
  }
}
