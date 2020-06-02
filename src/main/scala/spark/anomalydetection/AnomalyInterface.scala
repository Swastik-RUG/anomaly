package spark.anomalydetection

import com.typesafe.config.Config
import org.apache.spark.sql.{DataFrame, SparkSession}

trait AnomalyInterface {
//  def training(sparkSession: SparkSession, conf: Config)
//
//  def predict[T](session: SparkSession, model: T, config: Config): DataFrame

  def apply(sparkSession: SparkSession, conf: Config): Unit
}
