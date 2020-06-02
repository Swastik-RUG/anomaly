import sbt._

object Dependencies {
  lazy val sparkCore = "org.apache.spark" %% "spark-core" % "2.4.0"
  lazy val sparkMLlib = "org.apache.spark" %% "spark-mllib" % "2.4.0"
  lazy val sparkSql = "org.apache.spark" %% "spark-sql" % "2.4.0"
  lazy val sparkStream = "org.apache.spark" %% "spark-streaming" % "2.4.0" % "provided"
  lazy val jodaTime = "joda-time" % "joda-time" % "2.8.1"
  lazy val typesafeConf = "com.typesafe" % "config" % "1.4.0"
  lazy val cassandraDatastax = "com.datastax.cassandra" % "cassandra-driver-core" % "3.8.0"
  lazy val cassandraCore = "com.datastax.cassandra" % "cassandra-driver-core" % "3.8.0"
  lazy val cassandraMapping = "com.datastax.cassandra" % "cassandra-driver-mapping" % "3.8.0"
  lazy val cassandraExtras = "com.datastax.cassandra" % "cassandra-driver-extras" % "3.8.0"
  lazy val cassandraConnector = "com.datastax.spark" %% "spark-cassandra-connector" % "2.4.0"
  lazy val jacksonDep = "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.10.2"
  lazy val kafkaApache = "org.apache.kafka" %% "kafka" % "2.3.0"
  lazy val kafkaStream = "org.apache.kafka" % "kafka-streams" % "2.3.0"
  lazy val kafkaScala = "org.apache.kafka" %% "kafka-streams-scala" % "2.3.0"
  lazy val kafkaSpark = "org.apache.spark" %% "spark-streaming-kafka-0-10" % "2.4.0"
  lazy val kafkaStreamSpark = "org.apache.spark" %% "spark-streaming-kafka" % "1.6.3"
  lazy val isolationForest = "com.linkedin.isolation-forest" % "isolation-forest_2.4.3_2.11" % "1.0.0"
}