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
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.api.ndarray.INDArray  // Added this import
import org.nd4j.linalg.indexing.NDArrayIndex
import org.deeplearning4j.util.ModelSerializer
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import java.io.File
import org.nd4j.linalg.ops.transforms.Transforms

object SlidingWindow {
  // Constants
  val embeddingSize = 64
  val windowSize = 3
  val batchSize = 32

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
    // Get dimensions
    val shape = input.shape()
    val batchSize = shape(0).toInt
    val sequenceLength = shape(1).toInt
    val embedSize = shape(2).toInt

    // Create query, key, and value matrices
    val query = input.dup()
    val key = input.dup()
    val value = input.dup()

    // Transpose key for matrix multiplication
    val keyTransposed = key.permute(0, 2, 1)

    // Compute attention scores
    val scores = Nd4j.matmul(query, keyTransposed)
      .div(math.sqrt(embedSize))

    // Apply softmax
    val attentionWeights = Transforms.softmax(scores)

    // Compute weighted sum
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

  def generateText(model: MultiLayerNetwork, df: DataFrame, seedTokens: Array[Row], numWords: Int, temperature: Double = 0.7): Seq[String] = {
    val embeddingDim = seedTokens.head.getAs[Vector]("embedding").size
    val positionalEmbeddings = computePositionalEmbedding(windowSize, embeddingDim)
    val generated = ArrayBuffer[String]()
    var currentWindow = seedTokens.toBuffer

    for (_ <- 1 to numWords) {
      // Prepare input
      val inputEmbeddings = currentWindow.map(_.getAs[Vector]("embedding"))
      val positionAwareEmbeddings = inputEmbeddings.zip(positionalEmbeddings).map {
        case (wordEmb, posEmb) =>
          val combined = (wordEmb.toArray, posEmb.toArray).zipped.map(_ + _)
          Vectors.dense(combined)
      }

      // Convert to Nd4j array and reshape
      val inputArray = Nd4j.create(positionAwareEmbeddings.flatMap(_.toArray).toArray)
      val reshapedInput = inputArray.reshape(1, windowSize, embeddingDim)

      // Apply self-attention
      val attentionOutput = selfAttention(reshapedInput)
      val flattenedInput = attentionOutput.reshape(1, windowSize * embeddingDim)

      // Get prediction
      val prediction = model.output(flattenedInput)

      // Find nearest neighbor in embedding space
      val nextToken = findNearestNeighbor(prediction, df)
      generated += nextToken.getAs[String]("word")

      // Update window
      currentWindow = currentWindow.tail :+ nextToken
    }

    generated
  }

  def findNearestNeighbor(prediction: INDArray, df: DataFrame): Row = {
    // Convert INDArray to Vector
    val predictionArray = new Array[Double](prediction.length().toInt)
    for (i <- 0 until prediction.length().toInt) {
      // Fix: Use getDouble with a long parameter
      predictionArray(i) = prediction.getDouble(i.toLong)  // Convert i to Long
      // Alternative fix: Use varargs syntax
      // predictionArray(i) = prediction.getDouble(i: Int*)
    }
    val predVector = Vectors.dense(predictionArray)

    // Find closest embedding
    val rows = df.collect()
    rows.minBy { row =>
      val embedding = row.getAs[Vector]("embedding")
      Vectors.sqdist(predVector, embedding)
    }
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SlidingWindow")
      .master("local[*]")
      .getOrCreate()

    try {
      spark.sparkContext.setLogLevel("ERROR")

      // Read embeddings data
      val inputPath = "/Users/niharikabelavadishekar/Documents/Cloud_Assignment/Homework2_LLM/src/main/resources/input/part-r-00000"
      val data = spark.sparkContext.textFile(inputPath)

      // Parse input data with error handling
      val parsedRows = data.map { line =>
        try {
          val parts = line.trim.split("\\s+", 3)
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

      // Create and process sliding windows
      val embeddingDim = df.first().getAs[Vector]("embedding").size
      val positionalEmbeddings = computePositionalEmbedding(windowSize, embeddingDim)

      // Create training datasets
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

          // Create input array
          val input = Nd4j.create(positionAwareEmbeddings.flatMap(_.toArray).toArray)
          val target = Nd4j.create(targetEmbedding.toArray)

          Some(new DataSet(input, target))
        } else None
      }.toList

      // Train model
      val inputSize = windowSize * embeddingDim
      val hiddenSize = 128
      val outputSize = embeddingDim
      val model = createModel(inputSize, hiddenSize, outputSize)

      println("Starting model training...")
      val numEpochs = 10
      for (epoch <- 0 until numEpochs) {
        slidingWindows.grouped(batchSize).foreach { batch =>
          val features = Nd4j.vstack(batch.map(_.getFeatures): _*)
          val labels = Nd4j.vstack(batch.map(_.getLabels): _*)
          model.fit(features, labels)
        }
        println(s"Completed epoch $epoch")
      }

      // Generate sample text
      val seedTokens = rows.take(windowSize)
      val generatedText = generateText(model, df, seedTokens, 20)
      println("Generated text: " + generatedText.mkString(" "))

      // Save model
      val modelPath = "src/main/resources/output/trained_model.zip"
      ModelSerializer.writeModel(model, new File(modelPath), true)
      println(s"Model saved to $modelPath")

    } catch {
      case e: Exception =>
        println(s"Application error: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      spark.stop()
    }
  }
}