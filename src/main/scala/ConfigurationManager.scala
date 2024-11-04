// Importing necessary libraries
import com.typesafe.config.{Config, ConfigFactory}
import java.io._
import org.slf4j.LoggerFactory
import org.apache.spark.{SparkConf, SparkContext}

// This is used to manage configuration settings for a specified environment
class ConfigurationManager(val environment: String) extends Serializable {
  private val logger = LoggerFactory.getLogger(getClass)

  logger.info(s"Initializing ConfigurationManager for environment: $environment")
  private val config: Config = ConfigFactory.load().getConfig(environment)
  logger.debug("Configuration loaded successfully")

  // Configuration settings for Spark
  val sparkAppName: String = config.getString("spark.app.name")
  val sparkMaster: String = config.getString("spark.master")
  val executorMemory: String = config.getString("spark.executor.memory")
  val driverMemory: String = config.getString("spark.driver.memory")
  val serializer: String = config.getString("spark.serializer")

  logger.info(s"Spark configuration loaded - AppName: $sparkAppName, Master: $sparkMaster")
  logger.debug(s"Executor Memory: $executorMemory, Driver Memory: $driverMemory")

  // Configuration settings for model
  val vocabularySize: Int = config.getInt("model.vocabulary.size")
  val embeddingSize: Int = config.getInt("model.embedding.size")
  val windowSize: Int = config.getInt("model.window.size")
  val batchSize: Int = config.getInt("model.batch.size")
  val epochs: Int = config.getInt("model.epochs")
  val seed: Int = config.getInt("model.seed")
  val learningRate: Double = config.getDouble("model.learning.rate")
  val decayRate: Double = config.getDouble("model.decay.rate")

  logger.info(s"Model configuration loaded - Vocabulary Size: $vocabularySize, Embedding Size: $embeddingSize")
  logger.debug(s"Window Size: $windowSize, Batch Size: $batchSize, Epochs: $epochs")
  logger.trace(s"Seed: $seed, Learning Rate: $learningRate, Decay Rate: $decayRate")

  // Configuration settings for paths
  val inputPath: String = config.getString("paths.input")
  val samplePath: String = config.getString("paths.sample")
  val outputPath: String = config.getString("paths.output")
  val metricsPath: String = config.getString("paths.metrics")

  // Helper method to check if a path is S3 path
  def isS3Path(path: String): Boolean = path.startsWith("s3://")

  // Helper method to get base path for creating directories
  def getBasePath(path: String): String = {
    val lastSlashIndex = path.lastIndexOf("/")
    if (lastSlashIndex > 0) path.substring(0, lastSlashIndex)
    else path
  }

  logger.info("Path configurations loaded successfully")
  logger.debug(s"Input Path: $inputPath, Sample Path: $samplePath")
  logger.debug(s"Output Path: $outputPath, Metrics Path: $metricsPath")
}

// This handles file operations such as reading, writing, and metrics tracking
class FileHandler(config: ConfigurationManager, sc: SparkContext) extends Serializable {
  private val logger = LoggerFactory.getLogger(getClass)
  private val environment = config.environment

  // Helper method to create S3 directory structure
  private def createS3Directory(path: String): Unit = {
    logger.info(s"Creating S3 directory structure for path: $path")
    val rdd = sc.emptyRDD[String]
    try {
      rdd.coalesce(1).saveAsTextFile(path)
      logger.debug(s"Successfully created S3 directory: $path")
    } catch {
      case e: Exception =>
        logger.error(s"Error creating S3 directory: ${e.getMessage}", e)
    }
  }

  // Helper method to write data to S3
  private def writeToS3(path: String, data: String): Unit = {
    logger.info(s"Writing data to S3 path: $path")
    try {
      val dataRdd = sc.parallelize(Seq(data))
      dataRdd.coalesce(1).saveAsTextFile(path)
      logger.debug(s"Successfully wrote data to S3: $path")
    } catch {
      case e: Exception =>
        logger.error(s"Error writing to S3: ${e.getMessage}", e)
    }
  }

  // Reading the seed text with support for both local and S3
  def readSampleWord(): String = {
    logger.info(s"Reading sample word from ${config.samplePath}")

    try {
      if (environment == "emr") {
        // S3 reading using Spark
        sc.textFile(config.samplePath).collect().mkString(" ").trim
      } else {
        // Local file reading
        val source = scala.io.Source.fromFile(config.samplePath)
        val word = source.mkString.trim
        source.close()
        word
      }
    } catch {
      case e: Exception =>
        logger.error(s"Error reading sample word: ${e.getMessage}", e)
        logger.warn("Using default word 'Hello' due to read failure")
        "Hello" // Default word if reading fails
    }
  }

  // Writing generated text with support for both local and S3
  def writeGeneratedText(text: String): Unit = {
    logger.info(s"Writing generated text to ${config.outputPath}")

    try {
      if (environment == "emr") {
        // S3 writing
        writeToS3(config.outputPath, text)
      } else {
        // Local file writing
        val writer = new PrintWriter(config.outputPath)
        writer.write(text)
        writer.close()
      }
      logger.debug(s"Successfully wrote ${text.length} characters to output file")
    } catch {
      case e: Exception =>
        logger.error(s"Error writing generated text: ${e.getMessage}", e)
    }
  }

  // Creating metrics writer with support for both local and S3
  def createMetricsWriter(): MetricsWriter = {
    logger.info(s"Creating metrics writer for ${config.metricsPath}")

    if (environment == "emr") {
      new S3MetricsWriter(config.metricsPath, sc)
    } else {
      new LocalMetricsWriter(config.metricsPath)
    }
  }
}

// Abstract trait for metrics writing
trait MetricsWriter {
  def write(line: String): Unit
  def close(): Unit
}

// Implementation for local file system
class LocalMetricsWriter(path: String) extends MetricsWriter {
  private val writer = {
    val file = new File(path)
    file.getParentFile.mkdirs()
    val w = new BufferedWriter(new FileWriter(file))
    w.write("Epoch,LearningRate,Loss,Accuracy,BatchesProcessed,PredictionsMade,EpochDuration,NumberOfPartitions,NumberOfLines,MemoryUsed\n")
    w
  }

  def write(line: String): Unit = {
    writer.write(line)
    writer.flush()
  }

  def close(): Unit = {
    writer.close()
  }
}

// Implementation for S3
class S3MetricsWriter(path: String, sc: SparkContext) extends MetricsWriter {
  private val buffer = new StringBuilder()
  buffer.append("Epoch,LearningRate,Loss,Accuracy,BatchesProcessed,PredictionsMade,EpochDuration,NumberOfPartitions,NumberOfLines,MemoryUsed\n")

  def write(line: String): Unit = {
    buffer.append(line)
  }

  def close(): Unit = {
    val dataRdd = sc.parallelize(Seq(buffer.toString()))
    dataRdd.coalesce(1).saveAsTextFile(path)
  }
}