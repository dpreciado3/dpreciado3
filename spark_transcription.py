import os
import sys
import io
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# 1. Define the HDFS path to your packed environment
HDFS_ENV_PATH = "hdfs:///user/your_name/envs/whisper_env.tar.gz#environment"

# 2. Set environment variables BEFORE initializing the SparkSession
os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# 3. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Whisper_Jupyter_BinaryFile") \
    .master("yarn") \
    .config("spark.archives", HDFS_ENV_PATH) \
    .config("spark.executorEnv.PYSPARK_PYTHON", "./environment/bin/python") \
    .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON", "./environment/bin/python") \
    .config("spark.executor.cores", "4") \
    .config("spark.task.cpus", "4") \
    .config("spark.executor.memory", "10G") \
    .config("spark.executor.memoryOverhead", "4G") \
    .config("spark.executorEnv.OMP_NUM_THREADS", "1") \
    .config("spark.executorEnv.MKL_NUM_THREADS", "1") \
    .getOrCreate()

# 4. Define the output schema
schema = StructType([
    StructField("path", StringType(), True),
    StructField("transcription", StringType(), True)
])

# 5. Define the Pandas UDF
def transcribe_binary_batch(iterator):
    from faster_whisper import WhisperModel
    
    # Initialize the model on the worker using INT8 quantization
    model = WhisperModel(
        "base", 
        device="cpu", 
        compute_type="int8", 
        cpu_threads=4
    )

    for pdf in iterator:
        transcriptions = []
        for content in pdf['content']:
            try:
                # Wrap the raw byte array in a BytesIO object for Whisper to read
                audio_file = io.BytesIO(content)
                
                segments, _ = model.transcribe(audio_file, beam_size=5)
                text = " ".join([s.text for s in segments])
                transcriptions.append(text.strip())
            except Exception as e:
                transcriptions.append(f"ERROR: {str(e)}")

        yield pd.DataFrame({
            "path": pdf['path'],
            "transcription": transcriptions
        })

# 6. Load data using Spark's binaryFile format
# This reads the raw bytes of the files into the 'content' column
raw_df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.wav") \
    .load("hdfs:///data/audio_folder/")

# Drop unnecessary metadata columns (modificationTime, length) to reduce shuffle overhead
df = raw_df.select("path", "content")

# 7. Apply the inference UDF
results = df.mapInPandas(transcribe_binary_batch, schema=schema)

# 8. Trigger an action (Show results or write to storage)
results.show(truncate=False)

# spark.stop()
