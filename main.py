import torch
import whisperx
import gc
import json
from dotenv import load_dotenv
import os
from torch import cuda, device #added this for checking if cuda is available 

from transcription import transcribe_with_whisper
from diaritization import assign_speaker_labels
from diarized_transcript import assign_word_speakers


task = "transcribe_and_diaritize" # ["transcribe" or "transcribe_and_diaritize"]
device = device("cuda" if cuda.is_available() else "cpu").type # ["cuda", "cpu"]
audio_file = "3min_craig_audio_fresh.wav" # name of audio file as string
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy) # ["float16", "int8"]
dump_model = False # sets wheter to empty model from, when true,"gc.collect() torch.cuda.empty_cache() del model" will delete the model from memory. [False, True]
whisper_model="large-v2" #["tiny", "base", "small", "medium", "large", large-v2];
min_speakers=2 #min number of speakers in audio file
max_speakers=2 #max number of speakers in audio file
language = "" #this is not currently used but implementations will come later
save_output_dir = "diarized_transcript_results"#["transcription_results", "diarized_transcript_results"] #directy for the save_result_to_json() destination 
output_file_name = "" #this will be provided by the user through a streamlit ui later

def load_environment_variables():
    load_dotenv()
    return os.environ.get("HUGGINGFACE_API_KEY")

def align_with_whisper(result, audio, device):
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model_a
    
    return result

def save_result_to_json(result, output_file):
    with open(output_file, "w") as file:
        json.dump(result, file)
    print("Result saved to:", output_file)

def main(task, device, batch_size, compute_type, dump_model, min_speakers, max_speakers):
    HUGGINGFACE_API_KEY = load_environment_variables()
    
    audio_file = "3min_craig_audio_fresh.wav"
    whisper_model="large-v2" 

    if task == "transcribe":
        result = transcribe_with_whisper(audio_file, device, batch_size, compute_type, whisper_model, dump_model)
        print(result["segments"])###how is this printing just the text of results???
        result = align_with_whisper(result, whisperx.load_audio(audio_file), device)
        output_file = f"./{save_output_dir}/{audio_file}_transcript.json"
        save_result_to_json(result, output_file)

    elif task == "transcribe_and_diaritize":
        result = transcribe_with_whisper(audio_file, device, batch_size, compute_type, whisper_model, dump_model)
        print(result["segments"])
        result = align_with_whisper(result, whisperx.load_audio(audio_file), device)

        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HUGGINGFACE_API_KEY, device=device)
        diarize_segments = assign_speaker_labels(audio_file, diarize_model, min_speakers, max_speakers)
        result = assign_word_speakers(diarize_segments, result)
        output_file = f"./{save_output_dir}/{audio_file}_diarized_transcript.json"
        save_result_to_json(result, output_file)
    else:
        print(f"Invalid task: {task}. Please select either 'transcribe' or 'transcribe_and_diaritize'.")

if __name__ == "__main__":
    main(task, device, batch_size, compute_type, dump_model, min_speakers, max_speakers)



