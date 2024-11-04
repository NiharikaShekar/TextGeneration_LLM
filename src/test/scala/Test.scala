import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.nd4j.linalg.factory.Nd4j
import java.io.{File, PrintWriter}
import org.apache.spark.{SparkConf, SparkContext}
import scala.io.Source

class Test extends AnyFlatSpec with Matchers {

  // Unit Tests for SimpleTokenizer
  "SimpleTokenizer" should "correctly fit and encode text" in {
    val tokenizer = new SimpleTokenizer()
    val texts = Seq("hello world", "hello test")
    tokenizer.fit(texts)

    val encoded = tokenizer.encode("hello world")
    encoded.length shouldBe 2
    encoded(0) shouldBe tokenizer.encode("hello")(0)
  }

  it should "correctly decode indices back to text" in {
    val tokenizer = new SimpleTokenizer()
    val texts = Seq("hello world")
    tokenizer.fit(texts)

    val encoded = tokenizer.encode("hello world")
    val decoded = tokenizer.decode(encoded)
    decoded.trim shouldBe "hello world"
  }

  it should "handle unknown words during encoding" in {
    val tokenizer = new SimpleTokenizer()
    tokenizer.fit(Seq("hello world"))

    val encoded = tokenizer.encode("hello test")
    encoded.length shouldBe 2
    encoded(1) shouldBe 0  // Unknown word should be mapped to 0
  }

  // Unit Tests for CustomTrainingListener
  "CustomTrainingListener" should "track model score" in {
    val listener = new CustomTrainingListener()
    val model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
      .list()
      .layer(0, new DenseLayer.Builder().nIn(1).nOut(1).build())
      .build())
    model.init()

    listener.iterationDone(model, 1, 1)
    listener.getLastScore shouldBe model.score()
  }

  // Integration Tests for FileHandler
  "FileHandler" should "correctly write and read files" in {
    val config = new ConfigurationManager("test")
    val sparkConf = new SparkConf()
      .setAppName("TestApp")
      .setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val fileHandler = new FileHandler(config, sc)


    new File("src/test/resources/input").mkdirs()
    new File("src/test/resources/output").mkdirs()


    val sampleWriter = new PrintWriter("src/test/resources/input/sample.txt")
    sampleWriter.write("test")
    sampleWriter.close()

    val word = fileHandler.readSampleWord()
    word shouldBe "test"

    fileHandler.writeGeneratedText("generated text")
    val outputFile = new File("src/test/resources/output/generated_text.txt")
    outputFile.exists() shouldBe true

    sc.stop()
  }

  // Integration Test for ConfigurationManager
  "ConfigurationManager" should "load correct configuration" in {
    val config = new ConfigurationManager("test")
    config.sparkAppName shouldBe "DistributedLanguageModel-Test"
    config.sparkMaster shouldBe "local[2]"
    config.vocabularySize shouldBe 1000
  }

  // Integration Test for Metrics Writer
  "MetricsWriter" should "write metrics correctly" in {
    val config = new ConfigurationManager("test")
    val sc = new SparkContext(new SparkConf().setAppName("TestApp").setMaster("local[2]"))
    val fileHandler = new FileHandler(config, sc)
    val metricsWriter = fileHandler.createMetricsWriter()

    metricsWriter.write("1,0.005,0.5,75.0,100,1000,5000,4,1000,1GB\n")
    metricsWriter.close()

    val metricsFile = new File("src/test/resources/metrics/training_metrics.csv")
    metricsFile.exists() shouldBe true

    sc.stop()
  }

}