import torch
import whisperx
import gc
import json
from io import BytesIO
import tempfile
import os
from torch import cuda, device #added this for checking if cuda is available 

# language = "" #the language spoken in the audio file. not currently implemented.
#may no longer be neccessery 
# save_output_dir = "diarized_transcript_results"#["transcription_results", "diarized_transcript_results"] #directy for the save_result_to_json() destination used when the variable output_file is created.  
#may be best to move into main
user_device = device("cuda" if cuda.is_available() else "cpu").type # ["cuda", "cpu"]

# # Check and validate Hugging Face API key
# def check_hf_api_key():
    
#     if hf_api_key:
#         st.success("Hugging Face API Key is found!")
#     else:
#         st.error("Failed to update Hugging Face API Key. Please try again.")

#     tb_hf_apikey = st.text_input('Hugging Face API Key', hf_api_key)

#     return hf_api_key


def transcribe_with_whisper(audio_file, user_device, batch_size, compute_type, whisper_model, dump_model):
    model = whisperx.load_model(whisper_model, user_device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    
    if dump_model: # If dump_model is True, delete model to free up GPU resources
        gc.collect()
        torch.cuda.empty_cache()
        del model
    
    return result

def assign_speaker_labels(audio_file, diarize_model, min_speakers, max_speakers):
    diarize_segments = diarize_model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
    return diarize_segments

def assign_word_speakers(diarize_segments, result):
    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result


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

# def format_result(result, include_word_segments, include_speaker, include_text, include_time, include_words):
#     formatted_result = {}

#     if include_word_segments and 'word_segments' in result:
#         formatted_result["word_segments"] = result["word_segments"]

#     if 'segments' in result:
#         formatted_result["segments"] = []
#         for segment in result["segments"]:
#             seg_dict = {}

#             if include_text and 'text' in segment:
#                 seg_dict['text'] = segment['text']

#             if include_time and 'start' in segment and 'end' in segment:
#                 seg_dict['time'] = f"{segment['start']}-{segment['end']}"

#             if include_words and 'words' in segment:
#                 seg_dict['words'] = segment['words']

#             if include_speaker and 'speaker' in segment:
#                 seg_dict['speaker'] = segment['speaker']

#             formatted_result["segments"].append(seg_dict)

#     return formatted_result



def main(audio_file: bytes, user_device, batch_size, compute_type, dump_model, min_speakers, max_speakers, whisper_model, hf_api_key):

    # Save the audio file to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        fp.write(audio_file)
        audio_file_path = fp.name

    result = transcribe_with_whisper(audio_file_path, user_device, batch_size, compute_type, whisper_model, dump_model)
    result = align_with_whisper(result, whisperx.load_audio(audio_file_path), user_device)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_api_key, device=user_device)
    diarize_segments = assign_speaker_labels(audio_file_path, diarize_model, min_speakers, max_speakers)
    result = assign_word_speakers(diarize_segments, result)
    return result

    #if __name__ == "__main__":
    

