import com.typesafe.config.{Config, ConfigFactory}
import java.io._

class ConfigurationManager(environment: String) extends Serializable {
  private val config: Config = ConfigFactory.load().getConfig(environment)

  val sparkAppName: String = config.getString("spark.app.name")
  val sparkMaster: String = config.getString("spark.master")
  val executorMemory: String = config.getString("spark.executor.memory")
  val driverMemory: String = config.getString("spark.driver.memory")
  val serializer: String = config.getString("spark.serializer")
  val vocabularySize: Int = config.getInt("model.vocabulary.size")
  val embeddingSize: Int = config.getInt("model.embedding.size")
  val windowSize: Int = config.getInt("model.window.size")
  val batchSize: Int = config.getInt("model.batch.size")
  val epochs: Int = config.getInt("model.epochs")
  val seed: Int = config.getInt("model.seed")
  val learningRate: Double = config.getDouble("model.learning.rate")
  val decayRate: Double = config.getDouble("model.decay.rate")
  val inputPath: String = config.getString("paths.input")
  val samplePath: String = config.getString("paths.sample")
  val outputPath: String = config.getString("paths.output")
  val metricsPath: String = config.getString("paths.metrics")
}

class FileHandler(config: ConfigurationManager) extends Serializable {
  def readSampleWord(): String = {
    try {
      val source = scala.io.Source.fromFile(config.samplePath)
      val word = source.mkString.trim
      source.close()
      word
    } catch {
      case e: Exception =>
        println(s"Error reading sample word: ${e.getMessage}")
        "scientist"
    }
  }

  def writeGeneratedText(text: String): Unit = {
    try {
      val writer = new PrintWriter(config.outputPath)
      writer.write(text)
      writer.close()
    } catch {
      case e: Exception =>
        println(s"Error writing generated text: ${e.getMessage}")
    }
  }

  def createMetricsWriter(): BufferedWriter = {
    val file = new File(config.metricsPath)
    file.getParentFile.mkdirs()
    val writer = new BufferedWriter(new FileWriter(file))
    writer.write("Epoch,\tLearningRate,\tLoss,\tAccuracy,\tBatchesProcessed,\tPredictionsMade,\tEpochDuration,\tNumber of partitions,\tNumber Of Lines,\tMemoryUsed\n")
    writer
  }
}
