import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import org.deeplearning4j.optimize.api.IterationListener

class SimpleTokenizer extends Serializable {
  private var wordToIndex = Map[String, Int]()
  private var indexToWord = Map[Int, String]()
  private var currentIdx = 0

  def fit(texts: Seq[String]): Unit = {
    texts.flatMap(_.split("\\s+")).distinct.foreach { word =>
      if (!wordToIndex.contains(word)) {
        wordToIndex += (word -> currentIdx)
        indexToWord += (currentIdx -> word)
        currentIdx += 1
      }
    }
  }

  def encode(text: String): Seq[Int] = text.split("\\s+").map(word => wordToIndex.getOrElse(word, 0))
  def decode(indices: Seq[Int]): String = indices.map(idx => indexToWord.getOrElse(idx, "")).mkString(" ")
}

class CustomTrainingListener extends ScoreIterationListener(10) {
  private var currentScore: Double = 0.0
  override def iterationDone(model: Model, iteration: Int, epochNum: Int): Unit = {
    super.iterationDone(model, iteration, epochNum)
    currentScore = model.score()
  }
  def getLastScore: Double = currentScore
}

class GradientNormListener(logFrequency: Int) extends IterationListener {
  override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
    if (iteration % logFrequency == 0) {
      val gradients: INDArray = model.gradient().gradient()
      println(s"Iteration $iteration: Gradient Mean = ${gradients.meanNumber().doubleValue()}, Max = ${gradients.maxNumber().doubleValue()}, Min = ${gradients.minNumber().doubleValue()}")
    }
  }
}
