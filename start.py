import torch
import whisperx
import gc
import json
from dotenv import load_dotenv
import os
from torch import cuda, device #added this for checking if cuda is available 

device = device("cuda" if cuda.is_available() else "cpu")
audio_file = "3min_craig_audio_fresh.wav" #name of audio file
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy) #should be able to be either "float16" or "int8"
dump_model = False #which when faulse we will not do "gc.collect() torch.cuda.empty_cache() del model" and when true we will.
whisper_model="large-v2" #this is a constant but other implementations will come later
diaritization_model="pyannote/speaker-diarization" #this is not currently used but implementations will come later
min_speakers=2 #min number of speakers in audio file
max_speakers=2 #max number of speakers in audio file
language = "" #this is not currently used but implementations will come later
save_output_dir = "" #this is not currently used but implementations will come later
output_file_name = "" #this is not currently used but implementations will come later

def load_environment_variables():
    load_dotenv()
    return os.environ.get("HUGGINGFACE_API_KEY")


def transcribe_with_whisper(audio_file, device, batch_size, compute_type):
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    
    # Delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model
    
    return result


def align_with_whisper(result, audio, device):
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model_a
    
    return result


def assign_speaker_labels(audio_file, diarize_model):
    diarize_segments = diarize_model(audio_file)
    diarize_model(audio_file, min_speakers=2, max_speakers=2)
    
    return diarize_segments


def assign_word_speakers(diarize_segments, result):
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result


def save_result_to_json(result, output_file):
    with open(output_file, "w") as file:
        json.dump(result, file)
    print("Result saved to:", output_file)


def main():
    # Load environment variables
    HUGGINGFACE_API_KEY = load_environment_variables()
    
    device = "cuda" 
    audio_file = "3min_craig_audio_fresh.wav"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    
    # 1. Transcribe with original whisper (batched)
    result = transcribe_with_whisper(audio_file, device, batch_size, compute_type)
    print(result["segments"]) # before alignment
    
    # 2. Align whisper output
    result = align_with_whisper(result, whisperx.load_audio(audio_file), device)
    print(result["segments"]) # after alignment
    
    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HUGGINGFACE_API_KEY, device=device)
    diarize_segments = assign_speaker_labels(audio_file, diarize_model)
    result = assign_word_speakers(diarize_segments, result)
    print(diarize_segments)
    print(result["segments"]) # segments are now assigned speaker IDs
    
    # Write result to a JSON file
    output_file = "./result.json"
    save_result_to_json(result, output_file)


if __name__ == "__main__":
    main()
