package spark.anomalydetection

import java.io.FileWriter

import com.linkedin.relevance.isolationforest.IsolationForest
import com.typesafe.config.ConfigFactory
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import spark.anomalydetection.Constants.{FEATURES, OUTLIER_SCORE, PREDICTED_LABEL}
import spark.example.DataSim

// TODO: WORK IN PROGRESS
object FeatureFocusedEvaluator {
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

    //    def detectAndFilterOutliers(df: DataFrame): DataFrame = {
    //      val statsDF = df.stat.approxQuantile(Array(FEATURES), Array(0.25, 0.75), 0.0)
    //    }

    val modelSets = config.getString("ModelSets").split("[|]").map(models => {
      val modelConfig = models.split(":")
      (modelConfig(0), modelConfig(1).split(";"))
    })

    val scenarios = config.getString("contaminationScenarios").split(";")
    var itr = config.getInt("minItr")
    while (continue && itr >= 0) {
      scenarios.foreach { x =>
        if (x.equalsIgnoreCase(scenarios.head)) {
          if (itr == 0) {
            println("Continue the evaulation.....? Insert requried itr")
            val tmp = scala.io.StdIn.readInt()
            if (tmp == 0 && tmp != -1) itr = 5 else itr = tmp
          }
          ModelEvalationV3.refresh()
        }

        val data = DataSim.getData(spark, config)
        val outliers = data.filter(col("anomaly") === 1)
        val non_outlier = data.filter(col("anomaly") === 0)
        val splitData = non_outlier.randomSplit(Array(0.7, 0.3))
        val trainingDF = splitData.head
        val testingRawDF = splitData.last.union(outliers)
        testingRawDF.write.mode("append").json(s"src/main/resources/model/testing$x")
        val testingDF = spark.read.json(s"src/main/resources/model/testing$x")
        val adultrationDF = outliers.sample(0.1)
        adultrationDF.show(false)
        trainingDF
          //.union(adultrationDF)
          .write.mode("append").json(s"src/main/resources/model/training$x")
        val trainingDataDF = spark.read.json(s"src/main/resources/model/training$x").drop("value").transform(filterOutliers)

        val featureCols = config.getString(FEATURES).split(";")
        val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol(FEATURES)
        val assembledTrainDF = assembler.transform(trainingDataDF)
        val assembledTestDF = assembler.transform(testingDF)


        val model = new IsolationForest()
          .setNumEstimators(100)
          .setBootstrap(false)
          .setMaxSamples(0.8)
          .setMaxFeatures(1.0)
          .setFeaturesCol(FEATURES)
          .setPredictionCol(PREDICTED_LABEL)
          .setScoreCol(OUTLIER_SCORE)
          .setContamination(x.toDouble)
          .fit(assembledTrainDF)

        val predicted = model.transform(assembledTestDF)

        continue = ModelEvalationV3.evaluator(spark, config, predicted, "anomaly", PREDICTED_LABEL, x)

      }

      itr = itr - 1

      def filterOutliers(df: DataFrame): DataFrame = {
        val columns = df.drop("anomaly").columns
        columns.foldLeft(df) {
          (df, column) =>
            val Q1_3 = df.stat.approxQuantile(Array(column), Array(0.25, 0.75), 0.0).head
            val Q1 = Q1_3(0)
            val Q3 = Q1_3(1)
            val IRQ = Q3 - Q1
            val low = Q1 - 1.5 * IRQ
            val high = Q3 + 1.5 * IRQ
            val filteredDF = df.filter(col(column) > low).filter(col(column) < high)

            val filterCount = df.count() - filteredDF.count()
            val fw = new FileWriter("test.txt", true)
            try {
              fw.write("Filtered " + filterCount + "Outliers \n")
            }
            finally fw.close()

            filteredDF
        }
      }
    }

  }
}
