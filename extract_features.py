import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import pipeline
import numpy as np
# Function to extract wav2vec features
def extract_wav2vec_features(audio_folder):
    # Load the pre-trained wav2vec processor and model
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to('cuda')
        
    chunk_size = 16000 * 10  # 10 seconds at 16kHz
    min_chunk_size = 16000  # Minimum chunk size, let's say 1 second for this example
    # Iterate over files in the audio_folder
    for filename in os.listdir(audio_folder):
        if filename.endswith("_wav.npy"):
            print("Processing:", filename)
            file_path = os.path.join(audio_folder, filename)
            waveform = np.load(file_path)

            if waveform.ndim != 1:
                raise ValueError(f"Waveform has {waveform.ndim} dimensions. Expected 1 dimension.")

            # Split waveform into smaller chunks
            chunks = [waveform[i:i + chunk_size] for i in range(0, len(waveform), chunk_size)]

            all_hidden_states = []

            for chunk in chunks:
                if len(chunk) < min_chunk_size:
                    # Pad the chunk with zeros if it's too short
                    chunk = np.pad(chunk, (0, min_chunk_size - len(chunk)), mode='constant')
                waveform_tensor = torch.tensor(chunk).float().unsqueeze(0).to('cuda')  

                # Process waveform through wav2vec2
                input_values = processor(waveform_tensor, return_tensors="pt", sampling_rate=16000).input_values
                input_values = input_values.squeeze(1)
                input_values = input_values.to("cuda")
                with torch.no_grad():
                    hidden_states = model(input_values).last_hidden_state
                    all_hidden_states.append(hidden_states.cpu().numpy())

            # Combine chunks back into a single array
            features_array = np.concatenate(all_hidden_states, axis=1).squeeze(0)  

            # Save the extracted features
            npy_filename = filename.replace("_wav.npy", "_features.npy")
            npy_path = os.path.join(audio_folder, npy_filename)
            np.save(npy_path, features_array)
    

audio_folder_path = 'eeg_challenge_data_new/derivatives/split_data_16khz'
extract_wav2vec_features(audio_folder_path)
