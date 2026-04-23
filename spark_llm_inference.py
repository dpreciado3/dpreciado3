import os
import sys
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# HDFS Paths
HDFS_ENV = "hdfs:///user/your_name/envs/llm_inference_env.tar.gz#environment"
HDFS_GGUF = "hdfs:///user/your_name/models/gemma-4-26b-a4b-it-Q4_K_M.gguf"

os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder \
    .appName("Gemma4_ZeroShot_Sentiment") \
    .master("yarn") \
    .config("spark.archives", HDFS_ENV) \
    .config("spark.files", HDFS_GGUF) \
    \
    .config("spark.executorEnv.PYSPARK_PYTHON", "./environment/bin/python") \
    .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON", "./environment/bin/python") \
    .config("spark.executorEnv.PATH", "./environment/bin:$PATH") \
    .config("spark.executorEnv.LD_LIBRARY_PATH", "./environment/lib:./environment/lib64:$LD_LIBRARY_PATH") \
    \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "2") \
    .config("spark.dynamicAllocation.maxExecutors", "30") \
    .config("spark.dynamicAllocation.shuffleTracking.enabled", "true") \
    \
    .config("spark.executor.cores", "4") \
    .config("spark.task.cpus", "4") \
    .config("spark.executor.memory", "24G") \
    .config("spark.executor.memoryOverhead", "4G") \
    .config("spark.executorEnv.OMP_NUM_THREADS", "4") \
    .getOrCreate()

# Output Schema
schema = StructType([
    StructField("path", StringType(), True),
    StructField("transcription", StringType(), True),
    StructField("sentiment", StringType(), True)
])

def classify_sentiment(iterator):
    from llama_cpp import Llama
    import pandas as pd
    from pyspark import SparkFiles

    # 1. Locate the file distributed by spark.files
    model_path = SparkFiles.get("gemma-4-26b-a4b-it-Q4_K_M.gguf")
    
    # 2. Initialize Llama instance once per partition
    # n_ctx must be large enough for your longest transcript. 
    # High n_ctx drastically increases RAM usage. Adjust if OOM occurs.
    llm = Llama(
        model_path=model_path,
        n_ctx=4096, 
        n_threads=4, 
        verbose=False 
    )

    for pdf in iterator:
        sentiments = []
        for text in pdf['combined_output']: 
            if not text or str(text).startswith("ERROR"):
                sentiments.append("ERROR")
                continue
            
            # Gemma 4 instruction template with native system prompt support
            prompt = (
                "<start_of_turn>system\n"
                "You are a strict sentiment classification agent. Classify the following transcript as POSITIVE, NEUTRAL, or NEGATIVE. Output exactly one word and no other text.\n"
                "<end_of_turn>\n"
                "<start_of_turn>user\n"
                f"Transcript:\n{text}\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
            
            try:
                # max_tokens=5 ensures inference stops immediately after classification
                # temperature=0.0 eliminates hallucination and creative output
                response = llm(
                    prompt,
                    max_tokens=5,
                    temperature=0.0,
                    echo=False
                )
                
                raw_output = response['choices'][0]['text'].strip().upper()
                
                if "POSITIVE" in raw_output:
                    sentiments.append("POSITIVE")
                elif "NEGATIVE" in raw_output:
                    sentiments.append("NEGATIVE")
                elif "NEUTRAL" in raw_output:
                    sentiments.append("NEUTRAL")
                else:
                    sentiments.append(f"UNKNOWN_FORMAT: {raw_output}")
                
            except Exception as e:
                sentiments.append(f"INFERENCE_ERROR: {str(e)}")

        yield pd.DataFrame({
            "path": pdf['path'],
            "transcription": pdf['combined_output'],
            "sentiment": sentiments
        })

# Load the previously saved diarized transcriptions
df = spark.read.parquet("hdfs:///data/output/transcriptions_diarized/")

# Keep partition count aligned with the newly reduced maxExecutors
df = df.coalesce(60)

results = df.mapInPandas(classify_sentiment, schema=schema)
results.write.mode("overwrite").parquet("hdfs:///data/output/sentiments/")
