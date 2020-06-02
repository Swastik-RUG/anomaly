package spark.anomalydetection

import com.typesafe.config.ConfigFactory
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object Run {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("example").master("local").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val conf = ConfigFactory.load("application.conf")
    AnomalyDetector(spark, conf)
  }
}
