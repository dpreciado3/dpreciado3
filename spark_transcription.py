import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# 1. Define the schema for the output DataFrame
schema = StructType([
    StructField("audio_path", StringType(), True),
    StructField("transcription", StringType(), True)
])

def transcribe_batch(iterator):
    """
    This function runs on the Spark executors.
    It initializes the model once per partition and processes files in batches.
    """
    from faster_whisper import WhisperModel
    import fsspec
    
    # Initialize the model using INT8 quantization for CPU speed.
    # cpu_threads should match your spark.task.cpus configuration (e.g., 4).
    model = WhisperModel(
        model_size_or_path="base", 
        device="cpu", 
        compute_type="int8", 
        cpu_threads=4
    )
    
    for pdf in iterator:
        transcriptions = []
        for path in pdf['audio_path']:
            try:
                # fsspec allows reading directly from HDFS or other remote storage
                # "rb" returns a binary file-like object which faster-whisper accepts natively
                with fsspec.open(path, "rb") as f:
                    segments, _ = model.transcribe(f, beam_size=5)
                    text = " ".join([segment.text for segment in segments])
                    transcriptions.append(text.strip())
            except Exception as e:
                transcriptions.append(f"ERROR: {str(e)}")
                
        # Yield the results back to Spark as a Pandas DataFrame
        yield pd.DataFrame({
            "audio_path": pdf['audio_path'],
            "transcription": transcriptions
        })

if __name__ == "__main__":
    spark = SparkSession.builder.appName("WhisperCPUInference").getOrCreate()

    # 2. Gather your file paths. 
    # This could be reading a Hive table or dynamically listing HDFS directories.
    # Example hardcoded paths:
    paths = [
        "hdfs://namenode:8020/data/audio/file1.wav", 
        "hdfs://namenode:8020/data/audio/file2.wav"
    ]
    
    # Create a Spark DataFrame of the paths
    df = spark.createDataFrame(pd.DataFrame({"audio_path": paths}))

    # 3. Apply the distributed inference UDF
    transcribed_df = df.mapInPandas(transcribe_batch, schema=schema)

    # 4. Write the results back to your storage system (HDFS, Hive, etc.)
    transcribed_df.write.mode("overwrite").parquet("hdfs://namenode:8020/data/output/transcriptions.parquet")
    
    spark.stop()
