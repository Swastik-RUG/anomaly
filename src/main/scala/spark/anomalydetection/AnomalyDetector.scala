package spark.anomalydetection

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}

object AnomalyDetector {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().appName("example").master("local").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val conf = ConfigFactory.load("application.conf")
    run(spark, conf)
  }

  def run(sparkSession: SparkSession, conf: Config): Unit = {
    conf.getBoolean("training") match {
      case true => training(sparkSession, conf); predict(sparkSession, conf)
      case false => predict(sparkSession, conf)
    }
    val inputData = sparkSession.read.option("header", "true").csv("src/main/resources/data/training.csv")
  }

  def training(sparkSession: SparkSession, conf: Config): DataFrame = {
    ???
  }

  def predict(session: SparkSession, config: Config): DataFrame = {
    ???
  }
}
