import os
import sys
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

HDFS_ENV = "hdfs:///user/your_name/envs/gemma_inference_env.tar.gz#environment"
HDFS_GGUF = "hdfs:///user/your_name/models/qwen3-embedding-0.6b-q8_0.gguf"

os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder \
    .appName("Qwen3_Text_Embeddings") \
    .master("yarn") \
    .config("spark.archives", HDFS_ENV) \
    .config("spark.files", HDFS_GGUF) \
    .config("spark.executorEnv.PYSPARK_PYTHON", "./environment/bin/python") \
    .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON", "./environment/bin/python") \
    .config("spark.executorEnv.PATH", "./environment/bin:$PATH") \
    .config("spark.executorEnv.LD_LIBRARY_PATH", "./environment/lib:./environment/lib64:$LD_LIBRARY_PATH") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "2") \
    .config("spark.dynamicAllocation.maxExecutors", "60") \
    .config("spark.executor.cores", "4") \
    .config("spark.task.cpus", "4") \
    .config("spark.executor.memory", "8G") \
    .config("spark.executor.memoryOverhead", "2G") \
    .config("spark.executorEnv.OMP_NUM_THREADS", "4") \
    .getOrCreate()

schema = StructType([
    StructField("path", StringType(), True),
    StructField("embedding", ArrayType(FloatType()), True),
    StructField("error", StringType(), True)
])

def create_embeddings(iterator):
    from llama_cpp import Llama
    from pyspark import SparkFiles
    import pandas as pd

    model_path = SparkFiles.get("qwen3-embedding-0.6b-q8_0.gguf")
    
    # Initialize with embedding=True
    # n_ctx must encompass your longest transcript token count. 
    llm = Llama(
        model_path=model_path,
        embedding=True,
        n_ctx=8192, 
        n_threads=4, 
        verbose=False 
    )

    for pdf in iterator:
        embeddings = []
        errors = []
        for text in pdf['combined_output']: 
            if not text or str(text).startswith("ERROR"):
                embeddings.append(None)
                errors.append("INVALID_INPUT")
                continue
            
            try:
                # llama.cpp natively handles the pooling for the embedding output
                response = llm.create_embedding(text)
                emb = response["data"][0]["embedding"]
                embeddings.append(emb)
                errors.append(None)
                
            except Exception as e:
                embeddings.append(None)
                errors.append(str(e))

        yield pd.DataFrame({
            "path": pdf['path'],
            "embedding": embeddings,
            "error": errors
        })

df = spark.read.parquet("hdfs:///data/output/transcriptions_diarized/")
df = df.coalesce(120)

results = df.mapInPandas(create_embeddings, schema=schema)
results.write.mode("overwrite").parquet("hdfs:///data/output/text_embeddings/")
