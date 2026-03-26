import io
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("path", StringType(), True),
    StructField("transcription", StringType(), True)
])

def transcribe_binary_batch(iterator):
    from faster_whisper import WhisperModel
    
    # Initialize model once per partition
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
                # Wrap the raw bytes in a file-like object
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

if __name__ == "__main__":
    spark = SparkSession.builder.appName("WhisperBinaryFile").getOrCreate()

    # 1. Use Spark's native binary reader
    # This reads files into a DF with columns: [path, modificationTime, length, content]
    raw_df = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.wav") \
        .load("hdfs:///data/audio_folder/")

    # 2. Select only the necessary columns to reduce memory shuffle
    df = raw_df.select("path", "content")

    # 3. Process
    results = df.mapInPandas(transcribe_binary_batch, schema=schema)

    results.write.mode("overwrite").parquet("hdfs:///data/output_transcriptions")
    
    spark.stop()
    
