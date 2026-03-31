import os
import sys

# 1. Path to your packed environment on HDFS
# Format: hdfs:///path/to/env.tar.gz#alias
HDFS_ENV_PATH = "hdfs:///user/your_name/envs/whisper_env.tar.gz#environment"

# 2. Set environment variables for the Driver and Executors
# These must be set BEFORE creating the SparkSession
os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable  # Use current Jupyter kernel for driver

# 3. Initialize Spark Session with YARN configurations
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Whisper_Jupyter_Inference") \
    .master("yarn") \
    .config("spark.archives", HDFS_ENV_PATH) \
    .config("spark.executorEnv.PYSPARK_PYTHON", "./environment/bin/python") \
    .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON", "./environment/bin/python") \
    .config("spark.executor.cores", "4") \
    .config("spark.task.cpus", "4") \
    .config("spark.executor.memory", "8G") \
    .config("spark.executor.memoryOverhead", "2G") \
    .config("spark.executorEnv.OMP_NUM_THREADS", "1") \
    .config("spark.executorEnv.MKL_NUM_THREADS", "1") \
    .getOrCreate()

# 4. Define the Processing Logic (Pandas UDF)
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("audio_path", StringType(), True),
    StructField("transcription", StringType(), True)
])

def transcribe_batch(iterator):
    from faster_whisper import WhisperModel
    import fsspec
    
    # Initialize model inside the worker
    model = WhisperModel(
        "base", 
        device="cpu", 
        compute_type="int8", 
        cpu_threads=4
    )
    
    for pdf in iterator:
        transcriptions = []
        for path in pdf['audio_path']:
            try:
                with fsspec.open(path, "rb") as f:
                    segments, _ = model.transcribe(f, beam_size=5)
                    text = " ".join([s.text for s in segments])
                    transcriptions.append(text.strip())
            except Exception as e:
                transcriptions.append(f"ERROR: {str(e)}")
                
        yield pd.DataFrame({
            "audio_path": pdf['audio_path'],
            "transcription": transcriptions
        })

# 5. Execute on a sample
# Replace with your actual HDFS directory or file list
data = [{"audio_path": "hdfs:///data/audio/sample1.wav"}]
df = spark.createDataFrame(data)

results = df.mapInPandas(transcribe_batch, schema=schema)

# Show results or write to HDFS
results.show(truncate=False)

# spark.stop()
