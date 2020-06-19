package spark.example

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.sql.{DataFrame, SparkSession}
import spark.anomalydetection.Constants.APPLICATION_CONF
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import scala.util.Random

object DataSim {

  case class FeatureConf(featureName: String, minValue: Double, maxValue: Double) {
    def genRandomValue(): (Double, Int) = {
      val prob = Random.nextDouble()
      if (prob > 0.3) {
        val randvalue = minValue + Random.nextInt((maxValue.toInt - minValue.toInt) + 1)
        (randvalue, 0)
      } else {
        if (Random.nextDouble() <= 0.5)
          (minValue - (1 + Random.nextInt((500 - 1) + 1)), 1)
        else
          (maxValue + (1 + Random.nextInt((500 - 1) + 1)), 1)
      }
    }
  }

  case class Feature(featureName: String, value: Double, anomaly: Int)

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
    val config = ConfigFactory.load("datasim.conf")
    // getData(spark, config).filter(col("anomaly") === 1).show(false)
  }

  def getData(sparkSession: SparkSession, config: Config): DataFrame = {
    val recPerCall = config.getInt("recordsPerCall")
    val featureConfigs = config.getString("featureConfigs").split(";").map(featureConf => {
      val confs = featureConf.split(",")
      FeatureConf(confs(0), confs(1).toDouble, confs(2).toDouble)
    })

    // THis will give (feature1, feature2, feature3, anomaly) for a single record.
    featureConfigs.map(feature => feature.genRandomValue())
    val f = featureConfigs.map(feature => feature.genRandomValue())
    val x = f.map(_._1)
    val y = f.map(_._2).sum

    //    val genData = featureConfigs.foldLeft(List.empty[Feature])({
    //      (acc, feature) =>
    //        (0 until recPerCall).foldLeft(acc) {
    //          val randomData = feature.genRandomValue()
    //          (acc, indx) => acc :+ Feature(feature.featureName, randomData._1, randomData._2)
    //        }
    //    })


    val genData = (0 until recPerCall).foldLeft(List.empty[List[Double]]) {
      (list, row) => {
        val randFeatureData = featureConfigs.map(feature => feature.genRandomValue())
        val features = randFeatureData.map(_._1).toList
        val anomalyFlag = if (randFeatureData.map(_._2).sum > 0) 1.0 else 0.0
        list :+ (features :+ anomalyFlag)
      }
    }

    val structFields = featureConfigs.map(feature => StructField(feature.featureName, DoubleType, true))
    val schema = StructType(
      structFields
    )

    val requiredColumns = featureConfigs.map(_.featureName) :+ "anomaly"
    import sparkSession.implicits._
    val mins = (featureConfigs.map(_.minValue) :+ 0.0).toList
    val maxs = (featureConfigs.map(_.maxValue) :+ 0.0).toList
    // genData :+ mins :+ maxs
    val gendataDF = sparkSession.sparkContext.parallelize(genData).toDF() //.toDF((featureConfigs.map(_.featureName) :+ "anomaly"): _*)
    val res = requiredColumns.foldLeft(gendataDF) {
      (df, reqCols) => df.withColumn(reqCols, col("value")(requiredColumns.indexOf(reqCols)))
    }
    res.drop("value")
    res
  }


}
