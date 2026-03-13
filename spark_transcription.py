import os
from pyspark.sql import SparkSession
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
MODEL_PATH = "./models/whisper-medium-int8" # Local path in the distributed archive
OUTPUT_PATH = "hdfs:///user/data/transcriptions"

def transcribe_partition(iterator):
    """
    Initializes the model once per partition and processes files.
    """
    # Initialize model on CPU with int8 quantization
    # 'local' device refers to CPU in CTranslate2
    model = WhisperModel(
        MODEL_PATH, 
        device="cpu", 
        compute_type="int8", 
        cpu_threads=4 # Matches spark.task.cpus
    )
    
    results = []
    for row in iterator:
        file_path = row['path']
        # Note: In Hadoop, you may need to stream from HDFS 
        # or ensure files are accessible via a local mount (NFS/Fuse)
        segments, info = model.transcribe(file_path, beam_size=5)
        
        full_text = " ".join([segment.text for segment in segments])
        results.append((file_path, full_text, info.language))
        
    return iter(results)

def main():
    spark = SparkSession.builder \
        .appName("Whisper-CPU-Inference") \
        .config("spark.task.cpus", "4") \
        .getOrCreate()

    # Load file paths from Hive or HDFS
    # Assuming a DataFrame with a 'path' column
    df = spark.read.parquet("hdfs:///user/data/audio_metadata")

    # Repartition based on the number of executors to maximize parallel CPU usage
    num_executors = int(spark.conf.get("spark.executor.instances", "10"))
    df = df.repartition(num_executors)

    # Run inference
    transcriptions_rdd = df.rdd.mapPartitions(transcribe_partition)

    # Convert back to DF and save
    result_df = transcriptions_rdd.toDF(["file_path", "text", "detected_lang"])
    result_df.write.mode("overwrite").parquet(OUTPUT_PATH)

if __name__ == "__main__":
    main()
  
