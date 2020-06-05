import Dependencies._

name := "anomaly"

version := "0.1"

scalaVersion := "2.11.7"


scalaVersion := "2.11.7"

resolvers ++= Seq(
  Resolver.mavenCentral
)


libraryDependencies += sparkCore
libraryDependencies += sparkMLlib
libraryDependencies += sparkSql
libraryDependencies += jodaTime
libraryDependencies += typesafeConf
//libraryDependencies += cassandraDatastax
//libraryDependencies += cassandraCore
//libraryDependencies += cassandraMapping
//libraryDependencies += cassandraExtras
libraryDependencies += cassandraConnector
libraryDependencies += kafkaApache
libraryDependencies += kafkaStream
libraryDependencies += kafkaScala
libraryDependencies += kafkaSpark
libraryDependencies += kafkaStreamSpark
libraryDependencies += "org.apache.spark" %% "spark-streaming-kafka" % "1.6.3"
libraryDependencies += "org.apache.spark" % "spark-streaming_2.11" % "2.4.5" % "provided"
libraryDependencies += isolationForest
libraryDependencies += elastic