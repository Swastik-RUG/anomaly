package spark.anomalydetection

import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, SparkSession}

object Utility {

  def readDataFrame(fileType: String, sparkSession: SparkSession, inputPath: String, schema: StructType): DataFrame = {
    fileType match {
      case "json" => sparkSession.read.option("multiline", "true").schema(schema).json(inputPath)
      case _ => sparkSession.read.format("csv")
        .option("header", value = true)
        .option("delimiter", ",")
        .option("mode", "DROPMALFORMED")
        .option("timestampFormat", "MMM dd yyyy HH:mm:ss")
        .schema(schema)
        .load(inputPath)
        .cache()
    }
  }

}
