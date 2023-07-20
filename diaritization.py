import whisperx

def assign_speaker_labels(audio_file, diarize_model, min_speakers, max_speakers):
    diarize_segments = diarize_model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
    # result = whisperx.assign_word_speakers(diarize_segments, result) 
    # return result
    return diarize_segments
