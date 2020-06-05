package spark.anomalydetection

import com.linkedin.relevance.isolationforest.{IsolationForest, IsolationForestModel}
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.elasticsearch.spark.sql._
import spark.anomalydetection.Constants._

object AnomalyDetectionES {
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
    val config = ConfigFactory.load(APPLICATION_CONF)

    val inputDF = read(spark, config)

    val anomaliesDF = detectAnomaly(spark, config, inputDF, inputDF)

    write(spark, anomaliesDF, config)
  }

  def read(sparkSession: SparkSession, config: Config): DataFrame = {
    val source = scala.io.Source.fromFile(config.getString(SCHEMA_FILE))
    val lines = try source.mkString finally source.close()
    val schema = DataType.fromJson(lines).asInstanceOf[StructType]
    val inputData = sparkSession.read.format(ELASTIC_FORMAT).load(config.getString(SOURCE_ES_INDEX))

    inputData.show(false)
    val featureCols = config.getString(FEATURES).split(";")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol(RAW_FEATURES)
    val assembledDF = assembler.transform(inputData)
    val scaler = new StandardScaler()
      .setInputCol(RAW_FEATURES)
      .setOutputCol(FEATURES)
      .setWithStd(true)
      .setWithMean(true)

    scaler.fit(assembledDF).transform(assembledDF)
  }

  def detectAnomaly(sparkSession: SparkSession, config: Config, trainingDF: DataFrame, testingDF: DataFrame): DataFrame = {

    val model: IsolationForestModel = if (config.getBoolean(TRAINING_MODE)) {
      val isf = new IsolationForest()
        .setNumEstimators(100)
        .setBootstrap(false)
        .setMaxSamples(0.8)
        .setMaxFeatures(2)
        .setFeaturesCol(FEATURES)
        .setPredictionCol(PREDICTED_LABEL)
        .setScoreCol(OUTLIER_SCORE)
        .setContamination(0.1)
        .setRandomSeed(1).fit(trainingDF.sample(0.6))
      // Reduced sample to 60% for testing

      isf.save(config.getString(MODEL_CHECKPOINT))
      isf
    } else {
      IsolationForestModel.load(config.getString(MODEL_CHECKPOINT))
    }

    val pDf = model.transform(testingDF)
    pDf.printSchema()
    pDf.show(100, truncate = false)
    pDf.drop(RAW_FEATURES, FEATURES).filter(col(PREDICTED_LABEL) =!= 0.0)
  }

  def write(session: SparkSession, anomalyDF: DataFrame, config: Config): Unit = {
    anomalyDF.saveToEs(config.getString(DESTINATION_ES_INDEX))
  }
}
