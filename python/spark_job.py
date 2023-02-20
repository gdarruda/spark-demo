from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import *
from pyspark.sql import Row

spark = (SparkSession
            .builder
            .appName("Messages Prep (Python)")
            .master("local[6]")
            .config("spark.driver.memory", "12G")
            .config("spark.eventLog.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

file = spark.read.parquet('inline.parquet')
num_rows = file.count()
num_groups = (num_rows // 10) + 1

def add_sequence_column(df: DataFrame, col: str) -> DataFrame:
    
    return (df
            .rdd
            .zipWithIndex()
            .map(lambda x: Row(**(x[0].asDict() | {col: x[1]})))
            .toDF())

messages_to_send = (add_sequence_column(file, 'sequential_id')
    .withColumn('predict', struct([col(c.name) 
                                   for c 
                                   in file.schema]))
    .withColumn('predict_group', col('sequential_id') % lit(num_groups))
    .groupBy(col('predict_group'))
    .agg(collect_list('predict').alias('predicts'))
    .select(to_json(col("predicts"))))

(messages_to_send
    .write
    .mode("overwrite")
    .parquet("teste.parquet"))

spark.stop()