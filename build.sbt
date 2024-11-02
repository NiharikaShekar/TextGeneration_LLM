val scala2Version = "2.12.18"

lazy val sparkVersion = "3.5.3"

lazy val root = project
  .in(file("."))
  .settings(
    name := "SparkApp",
    version := "0.1.0-SNAPSHOT",
    scalaVersion := scala2Version,
    //    assembly / mainClass := Some("main.scala.MyMapReduceJob"),

    scalaVersion := scala2Version,
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "org.apache.spark" %% "spark-streaming" % sparkVersion,
      "org.apache.mrunit" % "mrunit" % "1.1.0" % Test classifier "hadoop2",
      "com.knuddels" % "jtokkit" % "0.6.1",
      "com.typesafe" % "config" % "1.4.3",
      "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1", // Latest version as of now
      "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1", // NLP support
      "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
      "org.slf4j" % "slf4j-simple" % "2.0.13", // Optional logging
      "org.scalameta" %% "munit" % "1.0.0" % Test
    ),
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) =>
        xs match {
          case "MANIFEST.MF" :: Nil =>   MergeStrategy.discard
          case "services" ::_       =>   MergeStrategy.concat
          case _                    =>   MergeStrategy.discard
        }
      case "reference.conf"  => MergeStrategy.concat
      case x if x.endsWith(".proto") => MergeStrategy.rename
      case x if x.contains("hadoop") => MergeStrategy.first
      case  _ => MergeStrategy.first
    }
  )


resolvers += "Conjars Repo" at "https://conjars.org/repo"