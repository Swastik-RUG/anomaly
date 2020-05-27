package spark.example

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

object SparkDfExamples {

  def main(args: Array[String]): Unit = {
    /**
     * Read the sensor data
     * master -> What is the mode to run this application on. local -> psuedodistributed mode, yarn -> cluster mode using yarn
     * client -> similar to yarn but the driver is in the local network while the cluster is in remote.
     * In yarn mode the driver and cluster are remote servers.
     * Declare it as private if declared outside a function since, SparkSessions should always be unique and should be not be directly called externally.
     *
     * The reason -> Spark stores every transformation we make as a logical query within this object, if there was function
     * call from outside to this sparksession object, the logical query can be modified leading to loss of the original logical query, errors, wrong results
     * and potentially leading to errors that are extremely hard to debug.
     */
    val spark = SparkSession.builder().appName("example").master("local").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    /**
     * Read the CSV input data file
     * option -> used to specify if the file contains header, or what is the delimiter etc.
     */
    val sensorData = spark.read.option("header", "true").csv("src/main/resources/data/KAG_energydata_complete.csv")

    /**
     * Printing and exploring the input dataframe
     */
    println("##### df.show() => Displays the top 20 truncated rows of the dataframe")
    sensorData.show(3, true)

    println("##### df.printShema() => Displays the schema/structure of the dataframe")
    sensorData.printSchema()

    println("##### Selecting only required columns and creating a new dataframe")

    println("""Normal selection df.select("column1", "column2")""")
    sensorData.select("date", "Appliances").show(3, false)

    println("""Using df.columns to select columns""")

    /**
     * sensorData.columns.slice(0, 4) -> select the first 4 columns of the dataframe and save it as an array.
     * :_* -> take the array and explode it to multiple arguments (like string*, int*), this is a scala sugar syntax.
     */
    sensorData.selectExpr(sensorData.columns.slice(0, 4): _*).show(3, false)

    println("Using list to select columns, df.selectExpr(list:_*)")
    sensorData.selectExpr(List("date", "Appliances"): _*).show(3, false)

    println("Reducing the columns of the dataframe for simplicity")
    val columns = List("date", "Appliances")

    /**
     * Saving it as a new dataframe, but it is still logical dataframe and has not created a new object
     * The data is actually moved or processed when there is an action call.
     **/
    val sensorDataReq = sensorData.selectExpr(columns: _*)

    println("""Adding a new column to the dataframe, df.withcolumn("columnName", value)""")
    println("With columns can be chained")
    sensorDataReq.withColumn("NewColumn", current_date())
      .withColumn("SecondColumn", lit("ANY VALUE")).show(3, false)

    println("""To Drop a column use df.drop("column1", "column2")""")
    sensorDataReq.drop("date").show(3, false)

    println("When and otherwise AKA IF and ELSE")

    /**
     * trim(col(*)) <- removes whitespaces
     * case(sql.dataType) <- type casting
     * when() <- IF
     * otherwise() <- else
     */
    sensorDataReq.withColumn("Appliances>50", when(trim(col("Appliances")).cast(IntegerType) > 50, true).otherwise(false)).show(3, false)

    /**
     * Performing Join
     */
    /**
     * List(1, 2, 3):+ 4 => List(1, 2, 3, 4) <- :+ it is an directional append operator ( 4 +: List(1,2 3) > List(4, 1, 2 ,3))
     * List(1, 2, 3) ++ List(4, 5) => List(1, 2, 3, 4 ,5)
     */
    val data1 = sensorData.selectExpr((sensorData.columns.slice(4, 6) ++ List("Appliances", "date")): _*)
    val data2 = sensorDataReq

    println("Left table to join")
    data1.show(3, false)
    println("Right table to join")
    data2.show(3, false)
    println("Join Key: Appliances and date")
    data1.join(data2, List("Appliances", "date"), joinType = "left").show(3, false)

    /**
     * Aggregation
     */
    sensorDataReq.groupBy("Appliances").count().show(5, false)



    /**
     * Difference between groupBy And partitionBy
     */
    println("Difference between groupBy And partitionBy")
    println("In PartitionBy date column is preserved, no column is dropped, whereas " +
      "in groupBy columns that are not in either groupBy or agg() will be dropped")
    println("PartitionBy will not drop rows, as you can see the rows are duplicated across, distinct can make it unique")
    /**
     * Using window for aggregation
     * Windows has a limit, so use it on smaller partitions, if you have allocated a ram of 8Gb and 9GB of data is moved to a single partition it will fail
     * A bit over 8GB will still work as spark has a nature of reserving a little more space that allocated.
     */
    val win = Window.partitionBy("Appliances")
    val senParittion = sensorDataReq.withColumn("countOfAppliances", count("Appliances").over(win))
    senParittion.show(5, false)
    println("Distinct and PartitionBy")
    senParittion.select("Appliances", "countOfAppliances").distinct().show(5,false)
  }

}
