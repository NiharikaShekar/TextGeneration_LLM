import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.util.ModelSerializer
import scala.collection.mutable.{ArrayBuffer, Set => MutableSet}
import scala.util.Random
import scala.util.Try
import java.io.File

object SlidingWindow {
  // Constants
  val embeddingSize = 512
  val windowSize = 3
  val batchSize = 32
  val hiddenSize = 128
  val numEpochs = 10

  // Hardcoded paths and input text
  val embeddingsPath = "/Users/niharikabelavadishekar/Documents/Cloud_Assignment/Homework2_LLM/src/main/resources/input/part-r-00000"
  val vocabularyPath = "/Users/niharikabelavadishekar/Documents/Cloud_Assignment/Homework2_LLM/src/main/resources/input/vocabulary.txt"
  val inputText = "the city lights"

  def loadVocabulary(path: String): Seq[String] = {
    Try {
      scala.io.Source.fromFile(path).getLines()
        .map(_.trim)
        .filter(_.nonEmpty)
        .toSeq
    }.getOrElse {
      println(s"Error loading vocabulary from $path. Using default vocabulary.")
      Seq("the", "city", "lights")
    }
  }

  def computePositionalEmbedding(windowSize: Int, embeddingDim: Int): Array[Vector] = {
    val positionalEncodings = new Array[Vector](windowSize)
    for (pos <- 0 until windowSize) {
      val embedding = new Array[Double](embeddingDim)
      for (i <- 0 until embeddingDim by 2) {
        val angle = pos / math.pow(10000, (2.0 * i) / embeddingDim)
        embedding(i) = math.sin(angle)
        if (i + 1 < embeddingDim) {
          embedding(i + 1) = math.cos(angle)
        }
      }
      positionalEncodings(pos) = Vectors.dense(embedding)
    }
    positionalEncodings
  }

  def selfAttention(input: INDArray): INDArray = {
    val shape = input.shape()
    val batchSize = shape(0).toInt
    val sequenceLength = shape(1).toInt
    val embedSize = shape(2).toInt

    val query = input.dup()
    val key = input.dup()
    val value = input.dup()

    val keyTransposed = key.permute(0, 2, 1)
    val scores = Nd4j.matmul(query, keyTransposed)
      .div(math.sqrt(embedSize))

    val attentionWeights = Transforms.softmax(scores)
    Nd4j.matmul(attentionWeights, value)
  }

  def createModel(inputSize: Int, hiddenSize: Int, outputSize: Int): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .updater(new Adam(0.001))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(inputSize)
        .nOut(hiddenSize)
        .activation(Activation.RELU)
        .dropOut(0.2)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(hiddenSize)
        .nOut(hiddenSize / 2)
        .activation(Activation.RELU)
        .dropOut(0.2)
        .build())
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .nIn(hiddenSize / 2)
        .nOut(outputSize)
        .activation(Activation.IDENTITY)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(10))
    model
  }

  def preprocessInput(input: String, df: DataFrame): Array[Row] = {
    val words = input.toLowerCase.split("\\s+")
    words.map { word =>
      try {
        df.filter(df("word") === word).head
      } catch {
        case e: Exception =>
          println(s"Warning: Word '$word' not found in vocabulary, using random word instead")
          df.take(1).head
      }
    }
  }

  def findTopKNeighbors(prediction: INDArray, df: DataFrame, k: Int): Array[Row] = {
    val predictionArray = new Array[Double](prediction.length().toInt)
    for (i <- 0 until prediction.length().toInt) {
      predictionArray(i) = prediction.getDouble(i.toLong)
    }
    val predVector = Vectors.dense(predictionArray)

    val rows = df.collect()
    rows.map { row =>
        val embedding = row.getAs[Vector]("embedding")
        (row, Vectors.sqdist(predVector, embedding))
      }
      .sortBy(_._2)
      .take(k)
      .map(_._1)
  }

  def generateText(model: MultiLayerNetwork,
                   df: DataFrame,
                   inputSentence: String,
                   vocabulary: Seq[String],
                   numWords: Int,
                   temperature: Double = 0.8): String = {

    val embeddingDim = df.first().getAs[Vector]("embedding").size
    val positionalEmbeddings = computePositionalEmbedding(windowSize, embeddingDim)
    val generated = ArrayBuffer[String]()

    var currentWindow = preprocessInput(inputSentence, df).takeRight(windowSize).toBuffer
    val usedWords = MutableSet[String]()
    inputSentence.toLowerCase.split("\\s+").foreach(usedWords.add)

    val random = new Random()

    for (_ <- 1 to numWords) {
      val inputEmbeddings = currentWindow.map(_.getAs[Vector]("embedding"))
      val positionAwareEmbeddings = inputEmbeddings.zip(positionalEmbeddings).map {
        case (wordEmb, posEmb) =>
          val combined = (wordEmb.toArray, posEmb.toArray).zipped.map(_ + _)
          Vectors.dense(combined)
      }

      val inputArray = Nd4j.create(positionAwareEmbeddings.flatMap(_.toArray).toArray)
      val reshapedInput = inputArray.reshape(1, windowSize, embeddingDim)

      val attentionOutput = selfAttention(reshapedInput)
      val flattenedInput = attentionOutput.reshape(1, windowSize * embeddingDim)
      val prediction = model.output(flattenedInput)

      val K = 10
      val candidates = findTopKNeighbors(prediction, df, K)
        .map(_.getAs[String]("word"))
        .filter(word => !usedWords.contains(word))

      if (candidates.nonEmpty) {
        val selectedWord = candidates(random.nextInt(candidates.length))
        generated += selectedWord
        usedWords.add(selectedWord)
        currentWindow = currentWindow.tail :+ df.filter(df("word") === selectedWord).head
      } else {
        val randomWord = vocabulary(random.nextInt(vocabulary.size))
        generated += randomWord
        currentWindow = currentWindow.tail :+ df.filter(df("word") === randomWord).head
      }
    }

    generated.mkString(" ")
  }

  def trainModel(df: DataFrame): (MultiLayerNetwork, Int) = {
    val embeddingDim = df.first().getAs[Vector]("embedding").size
    val positionalEmbeddings = computePositionalEmbedding(windowSize, embeddingDim)

    val rows = df.collect()
    val slidingWindows = rows.sliding(windowSize + 1).flatMap { window =>
      if (window.length == windowSize + 1) {
        val inputEmbeddings = window.take(windowSize).map(_.getAs[Vector]("embedding"))
        val positionAwareEmbeddings = inputEmbeddings.zip(positionalEmbeddings).map {
          case (wordEmb, posEmb) =>
            val combined = (wordEmb.toArray, posEmb.toArray).zipped.map(_ + _)
            Vectors.dense(combined)
        }

        val targetEmbedding = window.last.getAs[Vector]("embedding")
        val input = Nd4j.create(positionAwareEmbeddings.flatMap(_.toArray).toArray)
        val target = Nd4j.create(targetEmbedding.toArray)

        Some(new DataSet(input, target))
      } else None
    }.toList

    val inputSize = windowSize * embeddingDim
    val outputSize = embeddingDim
    val model = createModel(inputSize, hiddenSize, outputSize)

    println("Starting model training...")
    for (epoch <- 0 until numEpochs) {
      slidingWindows.grouped(batchSize).foreach { batch =>
        val features = Nd4j.vstack(batch.map(_.getFeatures): _*)
        val labels = Nd4j.vstack(batch.map(_.getLabels): _*)
        model.fit(features, labels)
      }
      println(s"Completed epoch $epoch")
    }

    (model, embeddingDim)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("TextGenerator")
      .master("local[*]")
      .getOrCreate()

    try {
      spark.sparkContext.setLogLevel("ERROR")

      println("Loading vocabulary...")
      val vocabulary = loadVocabulary(vocabularyPath)

      println("Loading embeddings...")
      val data = spark.sparkContext.textFile(embeddingsPath)
        .map(_.trim)
        .filter(_.nonEmpty)

      val parsedRows = data.map { line =>
        try {
          val parts = line.split("\\s+", 3)
          if (parts.length == 3) {
            val word = parts(0)
            val tokenId = parts(1).toInt
            val embeddingStr = parts(2).stripPrefix("[").stripSuffix("]")
            val embedding = Vectors.dense(
              embeddingStr.split(",").map(_.trim).filter(_.nonEmpty).map(_.toDouble)
            )
            Some(Row(word, tokenId, embedding))
          } else None
        } catch {
          case e: Exception =>
            println(s"Error parsing line: $line - ${e.getMessage}")
            None
        }
      }.filter(_.isDefined).map(_.get)

      val schema = StructType(Seq(
        StructField("word", StringType, true),
        StructField("tokenId", IntegerType, true),
        StructField("embedding", VectorType, true)
      ))

      val df = spark.createDataFrame(parsedRows, schema).cache()

      println("Training model...")
      val (model, embeddingDim) = trainModel(df)

      println("\nGenerating text...")
      val generatedText = generateText(model, df, inputText, vocabulary, 10)

      println(s"\nInput: $inputText")
      println(s"Generated continuation: $generatedText")

      // Save model
      val modelPath = "trained_model.zip"
      ModelSerializer.writeModel(model, new File(modelPath), true)
      println(s"\nModel saved to $modelPath")

    } catch {
      case e: Exception =>
        println(s"Application error: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      spark.stop()
    }
  }
}