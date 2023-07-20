import torch
import whisperx
import gc
import json
from dotenv import load_dotenv
from io import BytesIO
import tempfile
import os
from torch import cuda, device #added this for checking if cuda is available 

from transcription import transcribe_with_whisper
from diaritization import assign_speaker_labels
from diarized_transcript import assign_word_speakers

#requierd variables to allow the user to set in Streamlit
task = "td" # ["t" or "td] for either to "transcribe" or the default "transcribe and diaritize" a "t"
audio_file = "3min_craig_audio_fresh.wav" # name of audio file as string
min_speakers=2 #min number of speakers in audio file
max_speakers=2 #max number of speakers in audio file

#slightly less important variables and their default values
min_speakers=2 #min number of speakers in audio file
max_speakers=2 #max number of speakers in audio file
whisper_model="large-v2" #["tiny", "base", "small", "medium", "large", large-v2];

#even more slightly less important variables to be displayed even less prominently if need be
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy) # ["float16", "int8"]
dump_model = False # sets wheter to empty model from cache if low on memory, when true,"gc.collect() torch.cuda.empty_cache() del model" will delete the model from memory. [False, True]
language = "" #the language spoken in the audio file. not currently implemented.

#may no longer be neccessery 
save_output_dir = "diarized_transcript_results"#["transcription_results", "diarized_transcript_results"] #directy for the save_result_to_json() destination used when the variable output_file is created.  

#may be best to move into main
device = device("cuda" if cuda.is_available() else "cpu").type # ["cuda", "cpu"]


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

def main(task, audio_file: BytesIO, device, batch_size, compute_type, dump_model, min_speakers, max_speakers, whisper_model):
    HUGGINGFACE_API_KEY = load_environment_variables()

    # Save the audio file to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        fp.write(audio_file.read())
        audio_file_path = fp.name

    if task == "t":
        result = transcribe_with_whisper(audio_file_path, device, batch_size, compute_type, whisper_model, dump_model)
        result = align_with_whisper(result, whisperx.load_audio(audio_file_path), device)
        return result

    elif task == "td":
        result = transcribe_with_whisper(audio_file_path, device, batch_size, compute_type, whisper_model, dump_model)
        result = align_with_whisper(result, whisperx.load_audio(audio_file_path), device)

        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HUGGINGFACE_API_KEY, device=device)
        diarize_segments = assign_speaker_labels(audio_file_path, diarize_model, min_speakers, max_speakers)
        result = assign_word_speakers(diarize_segments, result)
        return result
    else:
        raise ValueError(f"Invalid task: {task}. Please select either 't' or 'td'.")


    #if __name__ == "__main__":
    

