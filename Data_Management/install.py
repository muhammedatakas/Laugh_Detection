from audioset_download import Downloader
import os 
import pandas as pd
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import logging

def download_laughter_data():
    # Download the audioset data
    downloader = Downloader(
    root_path="Data_Management",
    labels=["Laughter"],
    n_jobs=8,
    download_type='unbalanced_train',
    copy_and_replicate=False,
    )
    downloader.download(format='wav',quality=5)
    if os.path.getsize("Data_Management/Laughter") >=3000:
        return

def download_speech_data():
    # Download the audioset data
    downloader = Downloader(
    root_path="Data_Management/Speech",
    labels=["Speech"],
    n_jobs=8,
    download_type='balanced_train',
    copy_and_replicate=False,
    )
    downloader.download(format='wav',quality=5)
    if os.path.getsize("Data_Management/Speech") >=3000:
        return

def download_evulation_data():
    downloader = Downloader(
    root_path="Data_Management/Test_Laughter",
    labels=["Laughter"],
    n_jobs=8,
    download_type='eval',
    copy_and_replicate=False,
    )
    downloader.download(format='wav',quality=5)
    if os.path.getsize("Data_Management/Test_Laughter") >=3000:
        return

def download_evulation_data2():
    downloader = Downloader(
    root_path="Data_Management/Test_Speech",
    labels=["Speech"],
    n_jobs=8,
    download_type='eval',
    copy_and_replicate=False,
    )
    downloader.download(format='wav',quality=5)
    if os.path.getsize("Data_Management/Test_Speech/Speech") >=100:
        return 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_file_playable(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        return True
    except CouldntDecodeError:
        logging.error(f"Could not decode audio file: {file_path}")
        return False
    except Exception as e:
        logging.error(f"An error occurred while trying to load the audio file: {file_path}. Error: {e}")
        return False

def remove_corrupted_files_and_create_csv(root_path, labels):
    data = {
        "file": [],
        "label": []
    }
    for label in labels:
        label_path = os.path.join(root_path, label)
        for root, _, files in os.walk(label_path):
            for file in files:
                file_path = os.path.join(root, file)
                if is_file_playable(file_path):
                    data["file"].append(file)
                    data["label"].append(label)
                else:
                    logging.info(f"File {file_path} is corrupted.")
                    os.remove(file_path)
    
    df = pd.DataFrame(data)
    df.to_csv("labeled_data.csv", index=False)
    logging.info("CSV file created successfully.")

#download_laughter_data()
#download_speech_data()
#remove_corrupted_files_and_create_csv(root_path="Data_Management/laughter", labels=["Laughter"])
                    
#remove_corrupted_files_and_create_csv(root_path="Data_Management/speech", labels=["Speech"])

#download_evulation_data()
download_evulation_data2()