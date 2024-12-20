// Importing necessary libraries
import org.apache.commons.io.output.ByteArrayOutputStream
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.{ExponentialSchedule, ScheduleType}
import org.nd4j.linalg.ops.transforms.Transforms


import java.io.{BufferedWriter, ByteArrayInputStream, ObjectInputStream, ObjectOutputStream}
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.collection.JavaConverters._
import org.slf4j.LoggerFactory

// This is the main class for text generation and this requires a configuration manager
class TextGeneration(config: ConfigurationManager) extends Serializable {
  private val logger = LoggerFactory.getLogger(getClass)
  // Serializing model parameters and configurations into a byte array
  private def serializeModel(model: MultiLayerNetwork): Array[Byte] = {
    logger.debug("Starting model serialization")
    val baos = new ByteArrayOutputStream()
    try {
      val oos = new ObjectOutputStream(baos)
      oos.writeObject(model.params())
      oos.writeObject(model.getLayerWiseConfigurations)
      oos.close()
      baos.toByteArray
    } finally {
      baos.close()
      logger.info("Model serialization completed successfully")
    }
  }

  // Deserializing byte array back to a MultiLayerNetwork model
  private def deserializeModel(bytes: Array[Byte]): MultiLayerNetwork = {
    logger.debug("Starting model deserialization")
    val bais = new ByteArrayInputStream(bytes)
    try {
      val ois = new ObjectInputStream(bais)
      val params = ois.readObject().asInstanceOf[INDArray]
      val conf = ois.readObject().asInstanceOf[org.deeplearning4j.nn.conf.MultiLayerConfiguration]
      val model = new MultiLayerNetwork(conf)
      model.init()
      model.setParams(params)
      model
    } finally {
      bais.close()
      logger.info("Model deserialization completed successfully")
    }

  }

  // This is used to create sliding windows over token sequence for training text generation
  def createSlidingWindows(tokens: Seq[Int]): Seq[(Seq[Int], Int)] = {
    tokens.sliding(config.windowSize + 1).map { window =>
      (window.init, window.last)
    }.toSeq
  }

  // Generating an embedding matrix with positional encoding for a sequence of tokens
  def createEmbeddingMatrix(sequence: Seq[Int]): INDArray = {
    val embedding = Nd4j.zeros(1, config.embeddingSize, sequence.length)

    sequence.zipWithIndex.foreach { case (token, pos) =>
      val tokenEmbedding = Nd4j.randn(1, config.embeddingSize).mul(0.1)
      embedding.putSlice(pos, tokenEmbedding)
    }

    // Adding positional encodings to the embedding matrix
    for (pos <- sequence.indices) {
      for (i <- 0 until config.embeddingSize) {
        val angle = pos / math.pow(10000, (2 * i).toFloat / config.embeddingSize)
        if (i % 2 == 0) {
          embedding.putScalar(Array(0, i, pos), embedding.getDouble(0, i, pos) + math.sin(angle))
        } else {
          embedding.putScalar(Array(0, i, pos), embedding.getDouble(0, i, pos) + math.cos(angle))
        }
      }
    }

    embedding
  }

  // Now we are implementing the self-attention mechanism using query, key and value matrices
  def selfAttention(input: INDArray): INDArray = {
    val Array(batchSize, sequenceLength, embedSize) = input.shape()

    val query = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)
    val key = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)
    val value = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)

    if (query.isEmpty || key.isEmpty || value.isEmpty) {
      return Nd4j.empty()
    }

    val scores = query
      .tensorAlongDimension(0, 1, 2)
      .mmul(key.tensorAlongDimension(0, 1, 2).transpose())
      .div(math.sqrt(embedSize))

    val attentionWeights = Transforms.softmax(scores)
    val attendedOutput = attentionWeights
      .tensorAlongDimension(0, 1, 2)
      .mmul(value.tensorAlongDimension(0, 1, 2))

    attendedOutput.reshape(batchSize, sequenceLength, embedSize)
  }

  // Building a neural network model configuration
  def buildModel(validationIterator: DataSetIterator): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .seed(config.seed)
      .updater(new Adam(new ExponentialSchedule(ScheduleType.EPOCH, config.learningRate, config.decayRate)))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(config.embeddingSize * config.windowSize)
        .nOut(128)
        .activation(Activation.RELU)
        .dropOut(0.2)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(512)
        .nOut(128)
        .activation(Activation.RELU)
        .dropOut(0.2)
        .build())
      .layer(2, new OutputLayer.Builder()
        .nIn(128)
        .nOut(config.vocabularySize)
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()

    // Setting listeners for logging and evaluation
    val listener = new CustomTrainingListener
    model.setListeners(listener, new GradientNormListener(10), new EvaluativeListener(validationIterator, 1))
    model
  }

  // Creating validation dataset iterator
  def createValidationDataSetIterator(validationDataRDD: RDD[String], tokenizer: SimpleTokenizer): DataSetIterator = {
    val validationData = validationDataRDD.flatMap { text =>
      val tokens = tokenizer.encode(text)
      createSlidingWindows(tokens).map { case (inputSeq, label) =>
        val inputArray = Nd4j.zeros(1, config.embeddingSize * config.windowSize)
        val labelArray = Nd4j.zeros(1, config.vocabularySize)

        val embedding = createEmbeddingMatrix(inputSeq)
        val attentionOutput = selfAttention(embedding)
        if (!attentionOutput.isEmpty) {
          val flattenedAttention = attentionOutput.reshape(1, config.embeddingSize * config.windowSize)
          inputArray.putRow(0, flattenedAttention)
          labelArray.putScalar(Array(0, label), 1.0)
          new DataSet(inputArray, labelArray)
        }
        new DataSet()
      }
    }.collect().toList.asJava

    new ListDataSetIterator(validationData, config.batchSize)
  }

  // Processing a batch of data and then fitting it into the model
  private def processBatch(model: MultiLayerNetwork, batch: Seq[(Seq[Int], Int)]): (Double, Long, Long) = {
    val inputArray = Nd4j.zeros(batch.size, config.embeddingSize * config.windowSize)
    val labelsArray = Nd4j.zeros(batch.size, config.vocabularySize)

    batch.zipWithIndex.foreach { case ((sequence, label), idx) =>
      val embedding = createEmbeddingMatrix(sequence)
      val attentionOutput = selfAttention(embedding)
      if (!attentionOutput.isEmpty) {
        val flattenedAttention = attentionOutput.reshape(1, config.embeddingSize * config.windowSize)
        inputArray.putRow(idx, flattenedAttention)
        labelsArray.putScalar(Array(idx, label), 1.0)
      }
    }

    model.fit(inputArray, labelsArray)
    val output = model.output(inputArray)
    val predictions = Nd4j.argMax(output, 1)
    val labels = Nd4j.argMax(labelsArray, 1)
    val correct = predictions.eq(labels).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
      .sumNumber().longValue()

    (model.score(), correct, batch.size)
  }

  // Here we are averaging multiple models by their parameters
  private def averageModels(models: Array[MultiLayerNetwork]): MultiLayerNetwork = {
    val firstModel = models(0)
    if (models.length == 1) return firstModel

    val params = models.map(_.params())
    val avgParams = params.reduce((a, b) => a.add(b)).div(models.length)

    val result = new MultiLayerNetwork(firstModel.getLayerWiseConfigurations)
    result.init()
    result.setParams(avgParams)
    result
  }

  // Training the model for text generation
  def train(sc: SparkContext, textRDD: RDD[String], metricsWriter: MetricsWriter): MultiLayerNetwork = {
    logger.info(s"Starting training for ${config.epochs} epochs")
    val tokenizer = new SimpleTokenizer()
    val allTexts = textRDD.collect()
    tokenizer.fit(allTexts)
    val broadcastTokenizer = sc.broadcast(tokenizer)

    // Splitting the dataset into training (80%) and validation (20%) data
    val Array(trainingDataRDD, validationDataRDD) = textRDD.randomSplit(Array(0.8, 0.2))
    val validationDataSetIterator = createValidationDataSetIterator(validationDataRDD, tokenizer)

    // Building the initial model and serializing it for broadcasting
    val model = buildModel(validationDataSetIterator)
    var currentModelBytes = serializeModel(model)
    var broadcastModel = sc.broadcast(currentModelBytes)

    // These are the accumulators to track training metrics across partitions
    val batchProcessedAcc = sc.longAccumulator("batchesProcessed")
    val totalLossAcc = sc.doubleAccumulator("totalLoss")
    val correctPredictionsAcc = sc.longAccumulator("correctPredictions")
    val totalPredictionsAcc = sc.longAccumulator("totalPredictions")

    // Looping through epochs for training
    for (epoch <- 1 to config.epochs) {
      val epochStartTime = System.currentTimeMillis()
      println(s"Starting epoch $epoch")

      // Getting learning rate for the current epoch
      val learningRate = model.getLayerWiseConfigurations.getConf(0).getLayer
        .asInstanceOf[org.deeplearning4j.nn.conf.layers.BaseLayer]
        .getIUpdater.asInstanceOf[Adam].getLearningRate(epoch, config.epochs)

      println(s"Effective learning rate for epoch $epoch: $learningRate")

      // We are resetting the accumulators at the beginning of each epoch
      batchProcessedAcc.reset()
      totalLossAcc.reset()
      correctPredictionsAcc.reset()
      totalPredictionsAcc.reset()

      // Generating sliding windows over the token sequences in training data
      val samplesRDD = trainingDataRDD.flatMap { text =>
        val tokens = broadcastTokenizer.value.encode(text)
        createSlidingWindows(tokens)
      }.persist()

      // Processing the training data in distributed batches
      val processedRDD = samplesRDD.mapPartitions { partition =>
        val localModel = deserializeModel(broadcastModel.value)
        val batchBuffer = new scala.collection.mutable.ArrayBuffer[(Seq[Int], Int)]()
        var localLoss = 0.0
        var localCorrect = 0L
        var localTotal = 0L

        // Processing samples in batches within each partition
        partition.foreach { sample =>
          batchBuffer += sample
          if (batchBuffer.size >= config.batchSize) {
            val (loss, correct, total) = processBatch(localModel, batchBuffer.toSeq)
            localLoss += loss
            localCorrect += correct
            localTotal += total
            batchBuffer.clear()
            batchProcessedAcc.add(1)
          }
        }

        if (batchBuffer.nonEmpty) {
          val (loss, correct, total) = processBatch(localModel, batchBuffer.toSeq)
          localLoss += loss
          localCorrect += correct
          localTotal += total
          batchProcessedAcc.add(1)
        }

        // Accumulating the metrics across all partitions
        totalLossAcc.add(localLoss)
        correctPredictionsAcc.add(localCorrect)
        totalPredictionsAcc.add(localTotal)

        Iterator.single(serializeModel(localModel))
      }

      // Collecting the updated models from all partitions
      val updatedModels = processedRDD.collect()
      if (updatedModels.nonEmpty) {
        val averagedModel = if (updatedModels.length > 1) {
          val models = updatedModels.map(deserializeModel)
          averageModels(models)
        } else {
          deserializeModel(updatedModels(0))
        }

        broadcastModel.destroy()
        currentModelBytes = serializeModel(averagedModel)
        broadcastModel = sc.broadcast(currentModelBytes)

        // Calculate metrics for the current epoch
        val epochDuration = System.currentTimeMillis() - epochStartTime
        val avgLoss = totalLossAcc.value / batchProcessedAcc.value
        val accuracy = if (totalPredictionsAcc.value > 0) {
          correctPredictionsAcc.value.toDouble / totalPredictionsAcc.value
        } else 0.0

        // Logging the metrics
        println(f"""
                   |Epoch $epoch Statistics:
                   |Duration: ${epochDuration}ms
                   |Average Loss: $avgLoss%.4f
                   |Accuracy: ${accuracy * 100}%.2f%%
                   |Batches Processed: ${batchProcessedAcc.value}
                   |Predictions Made: ${totalPredictionsAcc.value}
                   |Memory Used: ${Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()}B
      """.stripMargin)

        val executorMemoryStatus = sc.getExecutorMemoryStatus.map { case (executor, (maxMemory, remainingMemory)) =>
          s"$executor: Max Memory = $maxMemory, Remaining Memory = $remainingMemory"
        }

        // Now, we write this metrics to a CSV file
        metricsWriter.write(
          f"$epoch,$learningRate%.6f,$avgLoss%.4f,${accuracy * 100}%.2f,${batchProcessedAcc.value},${totalPredictionsAcc.value},$epochDuration,${textRDD.getNumPartitions},${textRDD.count()},,${executorMemoryStatus.mkString("\n")}\n"
        )}
      logger.info(s"Epoch $epoch completed")
      samplesRDD.unpersist()
    }

    deserializeModel(broadcastModel.value)
  }

  // Generating text using the trained model
  def generateText(model: MultiLayerNetwork, tokenizer: SimpleTokenizer, seedText: String, length: Int, temperature: Double = 0.7): String = {
    logger.info(s"Starting text generation with seed text: $seedText")
    var currentSequence = tokenizer.encode(seedText).takeRight(config.windowSize)
    val generated = new ArrayBuffer[Int]()
    val rand = new Random()

    // This is a function which samples a token index from output logits with temperature
    def sampleWithTemperature(logits: INDArray, temp: Double): Int = {
      val scaled = logits.div(temp)
      val expScaled = Transforms.exp(scaled)
      val probs = expScaled.div(expScaled.sum(1))

      // Creating a probability array
      val probArray = Array.ofDim[Double](probs.columns())
      for (i <- 0 until probs.columns()) {
        probArray(i) = probs.getDouble(Long.box(i))
      }

      // Calculating cumulative probabilities for sampling
      val cumSum = probArray.scanLeft(0.0)(_ + _).tail
      val sample = rand.nextDouble()
      cumSum.zipWithIndex.find(_._1 >= sample).map(_._2).getOrElse(0)
    }

    // Generating the specified number of tokens
    for (_ <- 1 to length) {
      val embedding = createEmbeddingMatrix(currentSequence)
      val attentionOutput = selfAttention(embedding)
      val flattenedAttention = attentionOutput.reshape(1, config.embeddingSize * config.windowSize)

      val output = model.output(flattenedAttention)
      val nextTokenIndex = sampleWithTemperature(output, temperature)

      // Adding the sampled token index to the generated sequence
      generated += nextTokenIndex
      currentSequence = (currentSequence.tail :+ nextTokenIndex).takeRight(config.windowSize)
    }
    logger.info(s"Text generation completed")
    // Decoding the generated token indices back to text
    tokenizer.decode(generated.toSeq)

  }

  // Getting the Spark configuration
  def getSparkConf(): SparkConf = {
    new SparkConf()
      .setAppName(config.sparkAppName)
      .setMaster(config.sparkMaster)
      .set("spark.executor.memory", config.executorMemory)
      .set("spark.driver.memory", config.driverMemory)
      .set("spark.serializer", config.serializer)
      .set("spark.kryoserializer.buffer.max", "512m")
      .set("spark.kryoserializer.buffer", "256m")
      .registerKryoClasses(Array(
        classOf[MultiLayerNetwork],
        classOf[INDArray],
        classOf[Array[Byte]],
        classOf[org.nd4j.linalg.api.buffer.DataBuffer]
      ))
  }
}

// This is the main entry point of the application
object Text_Generation {
  def main(args: Array[String]): Unit = {
    val environment = if (args.isEmpty) "local" else args(0)
    val config = new ConfigurationManager(environment)
    val model = new TextGeneration(config)

    // Creating a Spark context with the specified configuration
    val sc = new SparkContext(model.getSparkConf())
    sc.setLogLevel("INFO")

    // Create FileHandler with SparkContext
    val fileHandler = new FileHandler(config, sc)
    val metricsWriter = fileHandler.createMetricsWriter()

    try {
      val textRDD = sc.textFile(config.inputPath).map(_.trim).filter(_.nonEmpty).cache()
      println(s"Number of partitions: ${textRDD.getNumPartitions}")
      println(s"Total number of lines: ${textRDD.count()}")

      // Training the model using the training data
      val trainedModel = model.train(sc, textRDD, metricsWriter)
      val tokenizer = new SimpleTokenizer()
      tokenizer.fit(textRDD.collect())

      // Reading a sample seed word from the file handler for generating text
      val seedWord = fileHandler.readSampleWord()
      val generatedText = model.generateText(trainedModel, tokenizer, seedWord, 50)
      val cleanedText = generatedText.replaceAll("\\s+", " ")
      fileHandler.writeGeneratedText(cleanedText)

      //Printing the generated text to the console
      println(s"Generated text: $cleanedText")
      println(s"Text has been saved to: ${config.outputPath}")
    } finally {
      metricsWriter.close()
      sc.stop()
    }
  }
}