import os
import sys
import io
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

# 1. HDFS Archive Paths
# #environment: Python env with faster-whisper, pyannote.audio, soundfile, torch (CPU)
# #whisper_model: The faster-whisper model files
# #diarize_model: The pyannote config.yaml and model weights
HDFS_ENV = "hdfs:///user/your_name/envs/whisper_diarization_env.tar.gz#environment"
HDFS_WHISPER = "hdfs:///user/your_name/models/whisper_base.zip#model_dir"
HDFS_DIARIZE = "hdfs:///user/your_name/models/diarize_model.zip#diarize_dir"

# 2. Driver Environment
os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# 3. Tuned Spark Session
spark = SparkSession.builder \
    .appName("Whisper_Diarization_Production") \
    .master("yarn") \
    .config("spark.archives", f"{HDFS_ENV}, {HDFS_WHISPER}, {HDFS_DIARIZE}") \
    \
    # System Library & Python Path Fixes
    .config("spark.executorEnv.PYSPARK_PYTHON", "./environment/bin/python") \
    .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON", "./environment/bin/python") \
    .config("spark.executorEnv.PATH", "./environment/bin:$PATH") \
    .config("spark.yarn.appMasterEnv.PATH", "./environment/bin:$PATH") \
    .config("spark.executorEnv.LD_LIBRARY_PATH", "./environment/lib:./environment/lib64:$LD_LIBRARY_PATH") \
    .config("spark.yarn.appMasterEnv.LD_LIBRARY_PATH", "./environment/lib:./environment/lib64:$LD_LIBRARY_PATH") \
    \
    # Queue-Specific Dynamic Allocation (Resilience for Preemption)
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.minExecutors", "2") \
    .config("spark.dynamicAllocation.maxExecutors", "60") \
    .config("spark.dynamicAllocation.shuffleTracking.enabled", "true") \
    \
    # Resource Allocation (CPU and Memory for 2+ Models)
    .config("spark.executor.cores", "4") \
    .config("spark.task.cpus", "4") \
    .config("spark.executor.memory", "18G") \
    .config("spark.executor.memoryOverhead", "6G") \
    \
    # Thread Control to prevent contention
    .config("spark.executorEnv.OMP_NUM_THREADS", "1") \
    .config("spark.executorEnv.MKL_NUM_THREADS", "1") \
    .getOrCreate()

# 4. Output Schema
schema = StructType([
    StructField("path", StringType(), True),
    StructField("combined_output", StringType(), True)
])

# 5. Vectorized UDF (Diarization + Transcription + Alignment)
def process_audio_batch(iterator):
    import torch
    import io
    import soundfile as sf
    import pandas as pd
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    # Force CPU and thread limits
    torch.set_num_threads(2)
    
    # Initialize Whisper once per partition
    whisper_model = WhisperModel(
        "./model_dir", 
        device="cpu", 
        compute_type="int8", 
        cpu_threads=2
    )

    # Initialize Pyannote once per partition (offline mode)
    # Ensure config.yaml inside the zip uses relative paths
    diarization_pipeline = Pipeline.from_pretrained("./diarize_dir/config.yaml")

    for pdf in iterator:
        outputs = []
        for content in pdf['content']:
            try:
                # --- A. Load & Resample ---
                audio_stream = io.BytesIO(content)
                data, samplerate = sf.read(audio_stream)
                
                # Convert to mono if needed
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                
                # Format for Pyannote (PyTorch Tensor: [Channels, Samples])
                waveform = torch.from_numpy(data).float().unsqueeze(0)

                # --- B. Diarization ---
                diarization_result = diarization_pipeline({
                    "waveform": waveform, 
                    "sample_rate": samplerate
                })
                
                speaker_segments = []
                for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                    speaker_segments.append({
                        "start": turn.start, 
                        "end": turn.end, 
                        "speaker": speaker
                    })

                # --- C. Transcription ---
                audio_stream.seek(0) # Reset pointer
                whisper_segments, _ = whisper_model.transcribe(audio_stream, beam_size=5)
                
                text_segments = []
                for s in whisper_segments:
                    text_segments.append({
                        "start": s.start, 
                        "end": s.end, 
                        "text": s.text.strip()
                    })

                # --- D. Manual Alignment (Intersection) ---
                final_lines = []
                for t_seg in text_segments:
                    best_speaker = "UNKNOWN"
                    max_overlap = 0
                    
                    for d_seg in speaker_segments:
                        # Find overlap duration between whisper text and speaker turn
                        overlap_start = max(t_seg["start"], d_seg["start"])
                        overlap_end = min(t_seg["end"], d_seg["end"])
                        overlap = max(0, overlap_end - overlap_start)
                        
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_speaker = d_seg["speaker"]
                    
                    final_lines.append(f"[{best_speaker}] {t_seg['text']}")

                outputs.append("\n".join(final_lines))

            except Exception as e:
                outputs.append(f"ERROR_PROCESSING: {str(e)}")

        yield pd.DataFrame({
            "path": pdf['path'],
            "combined_output": outputs
        })

# 6. Execution Block
# Load binary files from HDFS
raw_df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.wav") \
    .load("hdfs:///data/audio_input/")

# Pre-calculate partition count to optimize for the 60 executor limit
# (Total files / Files per task) - aim for tasks that take ~2-5 mins
df = raw_df.select("path", "content").coalesce(120)

# Run UDF
results = df.mapInPandas(process_audio_batch, schema=schema)

# Write to HDFS as Parquet (Scalable for production)
results.write.mode("overwrite").parquet("hdfs:///data/output/transcriptions_diarized/")
