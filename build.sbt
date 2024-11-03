val scala2Version = "2.13.10"  // Updated to match your Scala version
val sparkVersion = "3.5.3"

lazy val root = project
  .in(file("."))
  .settings(
    name := "LLM-hw2",
    version := "0.1.0-SNAPSHOT",
    scalaVersion := scala2Version,

    libraryDependencies ++= Seq(
      // Spark dependencies
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "org.apache.spark" %% "spark-streaming" % sparkVersion,
      // Test dependencies
      "org.apache.mrunit" % "mrunit" % "1.1.0" % Test classifier "hadoop2",
      "org.scalameta" %% "munit" % "1.0.0" % Test,

      // Other dependencies
      "com.knuddels" % "jtokkit" % "0.6.1",
      "com.typesafe" % "config" % "1.4.3",

      // DL4J dependencies
      "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
      "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1",
      "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
      "commons-io" % "commons-io" % "2.11.0",
      "com.typesafe" % "config" % "1.4.2",


      // Logging
      "org.slf4j" % "slf4j-api" % "2.0.13",
      "org.slf4j" % "slf4j-simple" % "2.0.13",
      "org.scalatest" %% "scalatest" % "3.2.9" % Test
    ),

    // Assembly settings
    assembly / assemblyMergeStrategy := {
      case x if x.contains("META-INF/services") => MergeStrategy.concat
      case x if x.contains("META-INF") => MergeStrategy.discard
      case x if x.contains("reference.conf") => MergeStrategy.concat
      case x if x.endsWith(".txt") => MergeStrategy.concat
      case x if x.endsWith(".proto") => MergeStrategy.first
      case x if x.contains("hadoop") => MergeStrategy.first
      case x if x.contains("properties") => MergeStrategy.concat
      case x if x.contains("xml") => MergeStrategy.first
      case x if x.contains("class") => MergeStrategy.first
      case PathList(ps @ _*) if ps.last endsWith ".html" => MergeStrategy.first
      case "application.conf" => MergeStrategy.concat
      case "unwanted.txt" => MergeStrategy.discard
      case x => MergeStrategy.first
    },

    // Exclude Scala library from assembly
    assembly / assemblyExcludedJars := {
      val cp = (assembly / fullClasspath).value
      cp filter { _.data.getName.contains("scala-library") }
    }
  )

// Add resolvers
resolvers ++= Seq(
  "Conjars Repo" at "https://conjars.org/repo",
  "Maven Central" at "https://repo1.maven.org/maven2/",
  Resolver.sonatypeRepo("releases"),
  Resolver.sonatypeRepo("snapshots")
)