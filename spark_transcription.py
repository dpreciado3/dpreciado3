# Updated Schema
schema = StructType([
    StructField("path", StringType(), True),
    StructField("transcription", StringType(), True),
    StructField("diarization", StringType(), True) # Added column
])

def transcribe_and_diarize_batch(iterator):
    import io
    import torch
    import soundfile as sf
    import pandas as pd
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    # 1. Thread locking to prevent C++ and PyTorch from colliding
    torch.set_num_threads(2)
    
    # 2. Initialize Whisper
    whisper_model = WhisperModel(
        "./model_dir", 
        device="cpu", 
        compute_type="int8", 
        cpu_threads=2 # Reduced to leave cores for PyTorch
    )

    # 3. Initialize Pyannote (Pointing to your localized, offline model archive)
    diarization_pipeline = Pipeline.from_pretrained("./diarization_model/config.yaml")

    for pdf in iterator:
        transcriptions = []
        diarizations = []
        
        for content in pdf['content']:
            try:
                # --- A. Data Preparation ---
                audio_stream = io.BytesIO(content)
                
                # Decode bytes to numpy array for Pyannote using soundfile
                data, samplerate = sf.read(audio_stream)
                
                # Convert to mono if stereo, and format for PyTorch (channels, samples)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                waveform = torch.from_numpy(data).float().unsqueeze(0)

                # --- B. Run Diarization ---
                # Pyannote accepts a dictionary directly in lieu of a file path
                diarization_result = diarization_pipeline({
                    "waveform": waveform, 
                    "sample_rate": samplerate
                })
                
                # Format diarization output: [(start, end, speaker_id)]
                speaker_segments = []
                for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                    speaker_segments.append({
                        "start": turn.start, 
                        "end": turn.end, 
                        "speaker": speaker
                    })

                # --- C. Run Transcription ---
                # Reset stream pointer to 0 so Whisper can read it from the beginning
                audio_stream.seek(0)
                whisper_segments, _ = whisper_model.transcribe(audio_stream, beam_size=5)
                
                # Format transcription output
                text_segments = []
                for s in whisper_segments:
                    text_segments.append({
                        "start": s.start, 
                        "end": s.end, 
                        "text": s.text
                    })

                # --- D. Timestamp Alignment (Intersection) ---
                # Assign the speaker who overlaps the most with each Whisper segment
                final_transcript = []
                for t_seg in text_segments:
                    assigned_speaker = "UNKNOWN"
                    max_overlap = 0
                    
                    for d_seg in speaker_segments:
                        # Calculate overlap duration
                        overlap_start = max(t_seg["start"], d_seg["start"])
                        overlap_end = min(t_seg["end"], d_seg["end"])
                        overlap = max(0, overlap_end - overlap_start)
                        
                        if overlap > max_overlap:
                            max_overlap = overlap
                            assigned_speaker = d_seg["speaker"]
                            
                    final_transcript.append(f"[{assigned_speaker}] {t_seg['text']}")

                transcriptions.append(" ".join([t["text"] for t in text_segments]))
                diarizations.append("\n".join(final_transcript))

            except Exception as e:
                transcriptions.append(f"EXECUTION_ERROR: {str(e)}")
                diarizations.append("ERROR")

        yield pd.DataFrame({
            "path": pdf['path'],
            "transcription": transcriptions,
            "diarization": diarizations
        })
