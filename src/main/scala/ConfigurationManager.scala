// Importing necessary libraries
import com.typesafe.config.{Config, ConfigFactory}
import java.io._
import org.slf4j.LoggerFactory

// This is used to manage configuration settings for a specified environment
class ConfigurationManager(environment: String) extends Serializable {
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

  logger.info("Path configurations loaded successfully")
  logger.debug(s"Input Path: $inputPath, Sample Path: $samplePath")
  logger.debug(s"Output Path: $outputPath, Metrics Path: $metricsPath")
}

// This handles file operations such as reading, writing, and metrics tracking
class FileHandler(config: ConfigurationManager) extends Serializable {
  private val logger = LoggerFactory.getLogger(getClass)

  // Reading the seed text which is given by the user and is used to generate further text
  def readSampleWord(): String = {
    logger.info(s"Attempting to read sample word from ${config.samplePath}")
    try {
      val source = scala.io.Source.fromFile(config.samplePath)
      val word = source.mkString.trim
      source.close()
      logger.debug(s"Successfully read sample word: $word")
      word
    } catch {
      case e: Exception =>
        logger.error(s"Error reading sample word: ${e.getMessage}", e)
        logger.warn("Using default word 'Hello' due to read failure")
        "Hello" // Default word if reading fails
    }
  }

  // This writes the generated text to an output file
  def writeGeneratedText(text: String): Unit = {
    logger.info(s"Attempting to write generated text to ${config.outputPath}")
    try {
      val writer = new PrintWriter(config.outputPath)
      writer.write(text)
      writer.close()
      logger.debug(s"Successfully wrote ${text.length} characters to output file")
    } catch {
      case e: Exception =>
        logger.error(s"Error writing generated text: ${e.getMessage}", e)
    }
  }

  // This creates a writer for metrics logging with a pre-defined header
  def createMetricsWriter(): BufferedWriter = {
    logger.info(s"Creating metrics writer for ${config.metricsPath}")
    val file = new File(config.metricsPath)
    file.getParentFile.mkdirs() // Ensure directories exist
    logger.debug("Created parent directories for metrics file")

    val writer = new BufferedWriter(new FileWriter(file))
    writer.write("Epoch,\tLearningRate,\tLoss,\tAccuracy,\tBatchesProcessed,\tPredictionsMade,\tEpochDuration,\tNumber of partitions,\tNumber Of Lines,\tMemoryUsed\n")
    logger.debug("Initialized metrics file with header")
    writer
  }
}