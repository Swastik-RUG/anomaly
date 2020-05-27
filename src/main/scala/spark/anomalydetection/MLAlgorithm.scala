package spark.anomalydetection

import com.typesafe.config.Config
import org.apache.spark.sql.{DataFrame, SparkSession}

trait MLAlgorithm {
  def training[T](sparkSession: SparkSession, conf: Config): T

  def predict[T](session: SparkSession, model: T, config: Config): DataFrame
}
