import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf

from pathlib import Path

# Print the current working directory
print("Current working directory:", os.getcwd())

#y, sr = librosa.load(path="Data_Management/Laughter/_540cGuw2H0_70.0-80.0.wav")
#S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels= 128)
#    
#S_dB = librosa.power_to_db(S=S, ref=np.max)
#plt.figure(figsize=(2.24, 2.24))
#librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
#plt.axis('off')    
#plt.savefig(os.path.join("Spectrogram_Images/Laughs", "_540cGuw2H0_70.0-80.0.png"), bbox_inches='tight', pad_inches=0)


def audio_augmentation():
    def load_audio(file_path):
        y, sr = librosa.load(file_path, sr=None)  # Load the audio file with its original sampling rate
        return y, sr
    
    def pitch_shift(y, sr, n_steps):
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    
    def time_stretch(y, rate):
        return librosa.effects.time_stretch(y=y, rate=rate)

    def add_noise(y, noise_factor=0.005):
        noise = np.random.randn(len(y))
        y_noisy = y + noise_factor * noise
        y_noisy = np.clip(y_noisy, -1.0, 1.0)  # Ensure the signal remains in the valid range
        return y_noisy
    
    def augment_and_save(input_dir, output_dir, augmentations):
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.wav'):  # Assuming your audio files are .wav
                file_path = input_dir / file_name
                y, sr = load_audio(file_path)

                # Apply augmentations
                for aug in augmentations:
                    y_augmented = aug(y, sr)
                    # Create a new filename for the augmented file
                    new_file_name = f"{file_name}_{aug.__name__}.wav"
                    sf.write(output_dir / new_file_name, y_augmented, sr)
                    
    augmentations = [
    lambda y, sr: pitch_shift(y, sr, n_steps=random.uniform(-2, 2)),  # Random pitch shift between -2 and 2 semitones
    lambda y, sr: time_stretch(y, rate=random.uniform(0.8, 1.2)),  # Random time stretch between 80% and 120%
    lambda y, sr: add_noise(y, noise_factor=random.uniform(0.002, 0.01))  # Random noise between 0.002 and 0.01
    ]
    
    non_laugh_dir=Path("Data_Management/Speech")
    augmented_non_laugh_dir = Path("Data_Management/Augmented_Non_Laugh")
    
    augment_and_save(non_laugh_dir, augmented_non_laugh_dir, augmentations)
    
def create_mel_spectrogram(audio_path, output_folder, filename):
    y, sr = librosa.load(audio_path)
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels= 128)
    
    S_dB = librosa.power_to_db(S=S, ref=np.max)
    
    plt.figure(figsize=(2.24, 2.24))
    librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight', pad_inches=0)
    
    
    plt.close()
    
def laugh_spectrogram():
    input_folder = 'Data_Management/Laughter'
    output_folder = 'Spectrogram_Images/Laughs'
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            create_mel_spectrogram(os.path.join(input_folder, filename), output_folder, filename.replace('.wav', '.png'))
            
def non_laugh_spectrogram():
    input_folder = 'Data_Management/Speech'
    output_folder = 'Spectrogram_Images/non_Laughs'
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            create_mel_spectrogram(os.path.join(input_folder, filename), output_folder, filename.replace('.wav', '.png'))

def test_spectrogram(max_files=1000):
    input_folder = 'Data_Management/Test_Speech/Speech'
    output_folder = 'Spectrogram_Images/Test/Speech'
    os.makedirs(output_folder, exist_ok=True)

    file_count = 0
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            create_mel_spectrogram(os.path.join(input_folder, filename), output_folder, filename.replace('.wav', '.png'))
            file_count += 1
            if file_count >= max_files:
                break
      
def augmented_spectrogram():
    input_folder = 'Data_Management/Augmented_Non_Laugh'
    output_folder = 'Spectrogram_Images/Non_Laughs'
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            create_mel_spectrogram(os.path.join(input_folder, filename), output_folder, filename.replace('.wav', '.png'))

test_spectrogram(31)