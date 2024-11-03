// Importing necessary libraries
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.optimize.api._
import org.slf4j.LoggerFactory

// This is a tokenizer that converts words to indices and vice versa
class SimpleTokenizer extends Serializable {
  private val logger = LoggerFactory.getLogger(getClass)

  private var wordToIndex = Map[String, Int]() // This is mapping of words to indices
  private var indexToWord = Map[Int, String]() // This is mapping of indices to words
  private var currentIdx = 0 // Tracks the next available index

  // This creates mappings for unique words in the input texts
  def fit(texts: Seq[String]): Unit = {
    logger.info("Starting tokenizer fitting process")
    val startTime = System.currentTimeMillis()

    texts.flatMap(_.split("\\s+")).distinct.foreach { word =>
      if (!wordToIndex.contains(word)) {
        wordToIndex += (word -> currentIdx)
        indexToWord += (currentIdx -> word)
        currentIdx += 1
      }
    }

    logger.info(s"Tokenizer fitting completed. Vocabulary size: $currentIdx")
    logger.debug(s"Fitting process took ${System.currentTimeMillis() - startTime}ms")
    logger.trace(s"First 10 words in vocabulary: ${wordToIndex.take(10).mkString(", ")}")
  }

  // This encodes a sentence into a sequence of indices based on wordToIndex
  def encode(text: String): Seq[Int] = {
    logger.debug(s"Encoding text of length ${text.length}")
    val encoded = text.split("\\s+").map(word => wordToIndex.getOrElse(word, 0))
    logger.trace(s"Encoded sequence length: ${encoded.length}")
    encoded
  }

  // This decodes a sequence of indices back into a sentence
  def decode(indices: Seq[Int]): String = {
    logger.debug(s"Decoding sequence of length ${indices.length}")
    val decoded = indices.map(idx => indexToWord.getOrElse(idx, "")).mkString(" ")
    logger.trace(s"Decoded text length: ${decoded.length}")
    decoded
  }
}

// This is a listener to monitor model score every N iterations
class CustomTrainingListener extends ScoreIterationListener(10) {
  private val logger = LoggerFactory.getLogger(getClass)
  private var currentScore: Double = 0.0 // Stores the latest model score

  // This is used to update currentScore each time an iteration completes
  override def iterationDone(model: Model, iteration: Int, epochNum: Int): Unit = {
    super.iterationDone(model, iteration, epochNum)
    currentScore = model.score()
    logger.info(s"Iteration $iteration completed - Score: $currentScore")
    logger.debug(s"Epoch: $epochNum, Current Score: $currentScore")
  }

  // This returns the most recent model score
  def getLastScore: Double = {
    logger.trace(s"Retrieved last score: $currentScore")
    currentScore
  }
}

// This is a listener to log gradient statistics (mean, max, min) periodically
class GradientNormListener(logFrequency: Int) extends IterationListener {
  private val logger = LoggerFactory.getLogger(getClass)

  override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
    if (iteration % logFrequency == 0) {
      val gradients: INDArray = model.gradient().gradient()
      val gradientMean = gradients.meanNumber().doubleValue()
      val gradientMax = gradients.maxNumber().doubleValue()
      val gradientMin = gradients.minNumber().doubleValue()

      logger.info(s"Iteration $iteration - Gradient Statistics")
      logger.debug(s"Mean: $gradientMean, Max: $gradientMax, Min: $gradientMin")
      logger.trace(s"Epoch: $epoch, Iteration: $iteration")
    }
  }
}