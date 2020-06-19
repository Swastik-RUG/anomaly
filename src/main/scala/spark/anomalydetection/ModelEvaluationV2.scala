//package spark.anomalydetection
//
//import java.io.FileWriter
//import java.nio.file.{Files, Paths}
//
//import breeze.linalg.DenseVector
//import breeze.plot.{Figure, plot}
//import com.typesafe.config.Config
//import org.apache.spark.sql.{DataFrame, SparkSession}
//import org.apache.spark.sql.functions.col
//
//import scala.collection.mutable
//
//object ModelEvaluationV2 {
//  var epochs = 0.0
//  val accuracyFig = Figure("Accuracy Eval")
//  val TNfig = Figure("TrueNegative Eval")
//  val FPfig = Figure("FalsePositive Eval")
//  val FNfig = Figure("FalseNegative Eval")
//  val Pfig = Figure("precision")
//  val Rfig = Figure("Recall")
//  var accuracies: List[(Double, Double)] = List.empty
//  var trueNegatives: List[(Double, Double)] = List.empty
//  var falsePositives: List[(Double, Double)] = List.empty
//  var falseNegatives: List[(Double, Double)] = List.empty
//  var precisions: List[(Double, Double)] = List.empty
//  var recalls: List[(Double, Double)] = List.empty
//  var buckets: mutable.ListBuffer[(List[(Double, Double)], List[(Double, Double)], List[(Double, Double)])] = List.empty
//
//  def evaluator(sparkSession: SparkSession,
//                config: Config,
//                df: DataFrame,
//                refLabel: String,
//                predictedLabel: String,
//                contamination: String,
//                display: Boolean = true): Boolean = {
//
//    try {
//      val total = df.count().toDouble
//      val records = config.getInt("recordsPerCall")
//      val outlierPerc = "0%"
//      // val miscalssification = df.filter(col("anomaly") =!= col(Constants.PREDICTED_LABEL)).count().toDouble
//      val truePositive = df.filter(col("anomaly") === 1.0).filter(col(Constants.PREDICTED_LABEL) === 1.0).count().toDouble
//      val trueNegative = df.filter(col("anomaly") === 0.0).filter(col(Constants.PREDICTED_LABEL) === 0.0).count().toDouble
//      val falsePositive = df.filter(col("anomaly") === 0.0).filter(col(Constants.PREDICTED_LABEL) === 1.0).count().toDouble
//      val falseNegative = df.filter(col("anomaly") === 1.0).filter(col(Constants.PREDICTED_LABEL) === 0.0).count().toDouble
//
//      val CTDF = df.filter(col("anomaly") === 0.0).filter(col(Constants.PREDICTED_LABEL) === 1.0)
//        .drop("features", "outlier_score", "predicted_label")
//      CTDF.write.mode("append").json("src/main/resources/model/training")
//
//      val filterCount = CTDF.count()
//      val fw = new FileWriter("test.txt", true)
//      try {
//        fw.write("False Positive records " + filterCount + " Added to Continuous Training \n")
//      }
//      finally fw.close()
//
//
//      println((truePositive + trueNegative).toDouble / total)
//
//      val accuracy = (truePositive + trueNegative) / total
//      val truePositiveRate = truePositive / (falseNegative + truePositive)
//      val trueNegativeRate = trueNegative / (trueNegative + falsePositive)
//      val falsePositiveRate = falsePositive / (trueNegative + falsePositive)
//
//      val precision = truePositive / (truePositive + falsePositive)
//      val recall = truePositive / (truePositive + falseNegative)
//      val f1Score = 2 * (1 / ((1 / precision) + (1 / recall)))
//
//      epochs = epochs + 1
//      accuracies = accuracies :+ (epochs, accuracy)
//      precisions = precisions :+ (epochs, precision)
//      recalls = recalls :+ (epochs, recall)
//      //      val ind = scenarios.indexOf(contamination)
//      //    buckets(ind) = (buckets(ind)._1 :+ accuracies, buckets(ind)._2 :+ precisions, buckets(ind)._3 :+ recalls)
//
//
//      trueNegatives = trueNegatives :+ (epochs, trueNegativeRate)
//      falsePositives = falsePositives :+ (epochs, falsePositiveRate)
//      falseNegatives = falseNegatives :+ (epochs, falseNegative)
//
//      if (display) {
//
//        accuracyFig.clear()
//        val p = accuracyFig.subplot(0)
//        val x = DenseVector(accuracies.map(_._1.toDouble): _*)
//        val y = DenseVector(accuracies.map(_._2): _*)
//        // p += plot(x, x)
//        p += plot(x, y)
//        p.ylim(0.0, 1.0)
//        p.setYAxisDecimalTickUnits()
//        p.ylabel = "Eval Percentage %"
//        p.xlabel = s"Epochs (value * $records = TrainingSize)"
//        p.title = s"Accuracy with $outlierPerc outliers"
//        accuracyFig.refresh()
//
//        Pfig.clear()
//        val pp = Pfig.subplot(0)
//        val xp = DenseVector(precisions.map(_._1.toDouble): _*)
//        val yp = DenseVector(precisions.map(_._2): _*)
//        // p += plot(x, x)
//        pp += plot(xp, yp, colorcode = "r", name = "Contamination: 0.3")
//        pp.legend = true
//        pp.ylim(0.0, 1.0)
//        pp.setYAxisDecimalTickUnits()
//        pp.ylabel = "Eval Percentage %"
//        pp.xlabel = s"Epochs (value * $records = TrainingSize)"
//        pp.title = s"precisions with $outlierPerc outliers"
//        Pfig.refresh()
//
//        Rfig.clear()
//        val pr = Rfig.subplot(0)
//        val xr = DenseVector(recalls.map(_._1.toDouble): _*)
//        val yr = DenseVector(recalls.map(_._2): _*)
//        // p += plot(x, x)
//        pr += plot(xr, yr)
//        pr.ylim(0.0, 1.0)
//        pr.setYAxisDecimalTickUnits()
//        pr.ylabel = "Eval Percentage %"
//        pr.xlabel = s"Epochs (value * $records = TrainingSize)"
//        pr.title = s"recalls with $outlierPerc outliers"
//        Rfig.refresh()
//
//      }
//
//      save(sparkSession)
//
//      if (accuracy >= 0.95)
//        false
//      else
//        true
//    }
//    catch {
//      case e: InterruptedException => save(sparkSession)
//        false
//    }
//  }
//
//  def save(sparkSession: SparkSession, path: String = "src/main/resources/model/eval_metrics"): Unit = {
//    import sparkSession.implicits._
//    val metricsToSave = List((epochs, accuracies(epochs.toInt - 1)._2, precisions(epochs.toInt - 1)._2, recalls(epochs.toInt - 1)._2))
//    val r = sparkSession.sparkContext.parallelize(metricsToSave).toDF("epochs", "accuracies", "precisions", "recalls")
//    r.show(false)
//    r.write.mode("append").json(path)
//  }
//
//  def read(sparkSession: SparkSession, path: String = "src/main/resources/model/eval_metrics"): Unit = {
//    if (Files.exists(Paths.get(path))) {
//      val data = sparkSession.read.json(path).orderBy("epochs").collect().foreach(row => {
//        val epoc = row.getAs[Double]("epochs")
//        val accuracy = row.getAs[Double]("accuracies")
//        val precision = row.getAs[Double]("precisions")
//        val recall = row.getAs[Double]("recalls")
//        accuracies = accuracies :+ (epoc, accuracy)
//        precisions = precisions :+ (epoc, precision)
//        recalls = recalls :+ (epoc, recall)
//        epochs = epoc
//      })
//    }
//  }
//
//  def initBuckets(size: Int): Unit = {
//    (0 until size).foreach(indx => {
//      buckets = buckets :+ (accuracies, precisions, recalls)
//    })
//  }
//
//}