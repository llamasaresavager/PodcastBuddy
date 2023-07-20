import whisperx

def assign_word_speakers(diarize_segments, result):
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result
