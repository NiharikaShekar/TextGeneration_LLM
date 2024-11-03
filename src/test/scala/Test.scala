import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.apache.spark.{SparkConf, SparkContext}
import org.nd4j.linalg.factory.Nd4j
import java.io.{File, PrintWriter}
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator
import org.nd4j.linalg.dataset.DataSet
import scala.collection.JavaConverters._


// Unit Tests
class TokenizerSpec extends AnyFlatSpec with Matchers {
  "SimpleTokenizer" should "correctly fit and encode text" in {
    val tokenizer = new SimpleTokenizer()
    val texts = Seq("hello world", "hello test")
    tokenizer.fit(texts)

    val encoded = tokenizer.encode("hello world")
    encoded.length should be(2)
  }

  it should "correctly decode indices back to text" in {
    val tokenizer = new SimpleTokenizer()
    val texts = Seq("hello world test")
    tokenizer.fit(texts)

    val encoded = tokenizer.encode("hello world")
    val decoded = tokenizer.decode(encoded)
    decoded.trim should be("hello world")
  }

  it should "handle unknown words during encoding" in {
    val tokenizer = new SimpleTokenizer()
    val texts = Seq("hello world")
    tokenizer.fit(texts)

    val encoded = tokenizer.encode("hello unknown")
    encoded(1) should be(0) // Unknown word should be encoded as 0
  }
}


// Integration Tests
class TextGenerationIntegrationSpec extends AnyFlatSpec with Matchers {
  var sc: SparkContext = _
  var config: ConfigurationManager = _
  var textGen: TextGeneration = _

  // Setup before tests
  override def withFixture(test: NoArgTest) = {
    config = new ConfigurationManager("test")
    sc = new SparkContext(new SparkConf()
      .setAppName("TestApp")
      .setMaster("local[2]"))
    textGen = new TextGeneration(config)

    try test()
    finally sc.stop()
  }


  it should "create sliding windows correctly" in {
    val tokens = Seq(1, 2, 3, 4, 5)
    val windows = textGen.createSlidingWindows(tokens)

    windows.size should be(4) // For window size 1
    windows.head._1.size should be(config.windowSize)
  }

  it should "handle file operations correctly" in {
    val fileHandler = new FileHandler(config)

    // Create test file
    val testDir = new File("src/test/resources/input")
    testDir.mkdirs()
    val writer = new PrintWriter(config.samplePath)
    writer.write("test")
    writer.close()

    val word = fileHandler.readSampleWord()
    word should be("test")
  }

  it should "generate text with sample input" in {
    // Create sample tokenizer and model
    val tokenizer = new SimpleTokenizer()
    val sampleTexts = Seq("hello world test generate")
    tokenizer.fit(sampleTexts)

    // Create a dummy validation iterator
    val inputArray = Nd4j.zeros(1, config.embeddingSize * config.windowSize)
    val labelsArray = Nd4j.zeros(1, config.vocabularySize)
    val dummyDataSet = new DataSet(inputArray, labelsArray)
    val validationData = List(dummyDataSet).asJava
    val validationIterator = new ListDataSetIterator(validationData, 1)

    // Build model with validation iterator
    val model = textGen.buildModel(validationIterator)

    // Initialize the model before generating text
    model.init()

    try {
      val generated = textGen.generateText(model, tokenizer, "hello", 5, 1.0)
      generated should not be empty
    } catch {
      case e: Exception =>
        fail(s"Text generation failed with error: ${e.getMessage}")
    }
  }


  it should "process RDD data correctly" in {
    val testData = Seq("hello world", "test data")
    val rdd = sc.parallelize(testData)

    val tokenizer = new SimpleTokenizer()
    tokenizer.fit(testData)

    // Test if RDD operations work
    val count = rdd.count()
    count should be(2)
  }

}

