package spark.anomalydetection

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Model
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}

object AnomalyDetector extends AnomalyInterface {

  def run(sparkSession: SparkSession, conf: Config): Unit = {
    conf.getBoolean("training") match {
      case true => val model = training(sparkSession, conf); predict(sparkSession, model, conf)
      case false => val model = training(sparkSession, conf); predict(sparkSession, model, conf)
    }
  }

  private def training(sparkSession: SparkSession, conf: Config) = {
    val schema = StructType(
      StructField("duration", DoubleType, nullable = true) ::
        StructField("network", DoubleType, nullable = true) ::
        StructField("storage", DoubleType, nullable = true) ::
        StructField("anomaly", IntegerType, nullable = true) ::
        Nil
    )
    val inputData = sparkSession.read.option("header", "true").schema(schema).csv("src/main/resources/data/training.csv")
    val featureCols = Array("duration", "network", "storage")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features").transform(inputData)
    val kmeans = new KMeans().setK(4).setSeed(1L)
    kmeans.fit(assembler)
  }

  private def predict(sparkSession: SparkSession, model: KMeansModel, config: Config): DataFrame = {
    val schema = StructType(
      StructField("duration", DoubleType, nullable = true) ::
        StructField("network", DoubleType, nullable = true) ::
        StructField("storage", DoubleType, nullable = true) ::
        StructField("anomaly", IntegerType, nullable = true) ::
        Nil
    )
    val inputData = sparkSession.read.option("header", "true").schema(schema).csv("src/main/resources/data/training.csv")
    val featureCols = Array("duration", "network", "storage")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features").transform(inputData)
    val predictions = model.transform(assembler)
    predictions.show(false)
    predictions
  }

  override def apply(sparkSession: SparkSession, conf: Config): Unit = {
    if (conf.getBoolean("training")) {
      val model = training(sparkSession, conf)
      model.save("/home/swastik/workspace/models/Kmeans")
      predict(sparkSession, model, conf)
    } else {
      val model = KMeansModel.load("/home/swastik/workspace/models/Kmeans")
      predict(sparkSession, model, conf)
    }
  }
}
