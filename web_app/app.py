import gradio as gr
from fastai.vision.all import *
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import uuid

# Load the trained model
learn = load_learner('final_model.pkl')

def create_mel_spectrogram(audio_path):
    # Load the audio file using librosa
    y, sr = librosa.load(audio_path)
    
    # Create the mel spectrogram directly from the audio data
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_dB = librosa.power_to_db(S=S, ref=np.max)
    
    plt.figure(figsize=(2.24, 2.24))
    librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.axis('off')
    
    # Save the spectrogram to a unique temporary file
    temp_filename = f'temp_spectrogram_{uuid.uuid4().hex}.png'
    plt.savefig(temp_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return temp_filename

def classify_audio(audio):
    try:
        # Create the mel spectrogram
        spectrogram_path = create_mel_spectrogram(audio.name)
        
        # Load the spectrogram image
        img = PILImage.create(spectrogram_path)
        
        # Get the prediction
        pred_class, pred_idx, probs = learn.predict(img)
        
        # Clean up the temporary file
        os.remove(spectrogram_path)
        
        return f"Predicted Class: {pred_class}, Probability: {probs[pred_idx].item():.4f}"
    except Exception as e:
        return str(e)

# Define the Gradio interface
interface = gr.Interface(
    fn=classify_audio, 
    inputs=gr.File(label="Upload an audio file"),
    outputs="text",
    live=True,
    
    
)

# Launch the interface
interface.launch(share=True)