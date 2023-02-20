import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object App {

  def add_sequence_colunm(df: DataFrame, 
                          col: String,
                          spark: SparkSession) : DataFrame = {
    
    val rdd = df
        .rdd
        .zipWithIndex()
        .map(x => Row.fromSeq(x._1.toSeq ++ Array(x._2)))
    
    val schema = df.schema.add(StructField(col, LongType))
  
    spark.createDataFrame(rdd, schema)
  }

  def main(args: Array[String]): Unit = {
    
    val spark = SparkSession
        .builder
        .appName("Messages Prep (Scala)")
        .master("local[6]")
        .config("spark.driver.memory", "12G")
        .config("spark.eventLog.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    val file = spark.read.parquet("inline.parquet")
    val num_rows = file.count()
    val num_groups = (num_rows / 10) + 1 

    val messages_to_send = add_sequence_colunm(file, "sequential_id", spark)
      .withColumn("predict", 
                  struct(file
                          .schema
                          .fields
                          .map(column => col(column.name)): _*))
      .withColumn("predict_group", col("sequential_id") % lit(num_groups))
      .groupBy(col("predict_group"))
      .agg(collect_list("predict").alias("predicts"))
      .select(to_json(col("predicts")).alias("predicts"))

    messages_to_send
      .write
      .mode("overwrite")
      .parquet("teste.parquet")

    spark.stop()
  }
}