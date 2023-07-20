import whisperx

def assign_speaker_labels(audio_file, diarize_model, min_speakers, max_speakers):
    diarize_segments = diarize_model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
    return diarize_segments
