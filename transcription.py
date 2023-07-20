import torch
import whisperx
import gc


def transcribe_with_whisper(audio_file, user_device, batch_size, compute_type, whisper_model, dump_model):
    model = whisperx.load_model(whisper_model, user_device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    
    if dump_model: # If dump_model is True, delete model to free up GPU resources
        gc.collect()
        torch.cuda.empty_cache()
        del model
    
    return result

