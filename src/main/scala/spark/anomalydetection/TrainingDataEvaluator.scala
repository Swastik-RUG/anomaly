package spark.anomalydetection

import java.lang.InterruptedException

import com.linkedin.relevance.isolationforest.IsolationForest
import com.typesafe.config.ConfigFactory
import org.apache.hadoop.hdfs.server.balancer.ExitStatus
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession
import spark.anomalydetection.Constants.{FEATURES, OUTLIER_SCORE, PREDICTED_LABEL}
import spark.example.DataSim
import org.apache.spark.sql.functions._

object TrainingDataEvaluator {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("anomalyDetection")
      .config("spark.es.index.auto.create", "true")
      .config("spark.es.nodes", "172.17.0.2")
      .config("spark.es.port", "9200")
      .config("es.index.auto.create", "true")
      .config("spark.es.net.http.auth.user", "elastic")
      .config("spark.es.net.http.auth.pass", "jAeydYNxUx2uUTE2")
      .config("spark.es.resource", "json3")
      .config("spark.es.nodes.wan.only", "true")
      .getOrCreate()

    val config = ConfigFactory.parseResources("datasim.conf").withFallback(ConfigFactory.parseResources(Constants.APPLICATION_CONF)).resolve()
    var continue = true

    ModelEvaluation.read(spark)

    while (continue) {

      val data = DataSim.getData(spark, config)
      val outliers = data.filter(col("anomaly") === 1)
      val non_outlier = data.filter(col("anomaly") === 0)
      val splitData = non_outlier.randomSplit(Array(0.7, 0.3))
      val trainingDF = splitData.head
      val testingRawDF = splitData.last.union(outliers)
      testingRawDF.write.mode("append").json("src/main/resources/model/testing")
      val testingDF = spark.read.json("src/main/resources/model/testing")
      val adultrationDF = outliers.sample(0.1)
      adultrationDF.show(false)
      trainingDF
        //.union(adultrationDF)
        .write.mode("append").json("src/main/resources/model/training")
      val trainingDataDF = spark.read.json("src/main/resources/model/training")

      val featureCols = config.getString(FEATURES).split(";")
      val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol(FEATURES)
      val assembledTrainDF = assembler.transform(trainingDataDF)
      val assembledTestDF = assembler.transform(testingDF)

      val model = new IsolationForest()
        .setNumEstimators(100)
        .setBootstrap(false)
        .setMaxSamples(0.6)
        .setMaxFeatures(2)
        .setFeaturesCol(FEATURES)
        .setPredictionCol(PREDICTED_LABEL)
        .setScoreCol(OUTLIER_SCORE)
        .setContamination(0.1)
        .fit(assembledTrainDF)

      val predicted = model.transform(assembledTestDF)

      continue = ModelEvaluation.evaluator(spark, config, predicted, "anomaly", PREDICTED_LABEL)
    }
  }
}
