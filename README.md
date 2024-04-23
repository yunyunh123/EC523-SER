# EC523-SER

---

# Getting started with datasets

This shows how to use some of the functions to get audio data into a format to use with pytorch

For downloading data if you have a "kaggle.json" file you can change directory to where that file is and you won't have to enter username and password for kaggle in order to download that dataset
```
from torch.utils.data import DataLoader
from dataset import download_datasets, SpeechEmotionDataset, get_dataset_info

# Specify the directory you want the datasets to be contained in
dataset_dir = "/home/datasets"

# Download a single dataset
download_datasets(dataset_dir, dname="emodb")

# Download the rest of the datasets available
download_datasets(dataset_dir)

# Acquire info on datasets (those that have functions to get data for)
df = get_dataset_info(dataset_dir)

# Make into a Dataset object that a pytorch optimizer can use
# Can optionally specify a sampling rate for all audio files to be in
trainset = SpeechEmotionDataset(df, fs=16000)

# Check it works
trainset = SpeechEmotionDataset(df, fs=16000)
dataiter = iter(trainset)
data, label = next(dataiter)
print(data)
print(label)

# Put into a dataloader
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)
```

# Our experimental design

We chose 6 popular databases of audio, each containing collections of raw audio with corresponding emotion labels.  Each dataset different emotion labels and different numbers of .wav files for each emotion, detailed in the table below:

| Emotion | Datasets | --- | --- | --- | --- | --- |
| --- | --- | --- | --- | --- | --- | --- |
| --- | CREMAD | EMODB | RAVDESS | SAVEE | SHEMO | TESS |
| anger | 1271 | 127 | 192 | 60 | 1059 | 400 |
| anxiety | x | 69 | x | x | x | x |
| bored | x | 81 | x | x | x | x |
| calm | x | x | 192 | x | x | x |
| disgust | 1271 | 46 | 192 | 60 | x | 400 |
| fear | 1271 | x | 192 | 60 | 38 | 400 |
| happy | 1271 | 71 | 192 | 60 | 201 | 400 |
| neutral | 1087 | 79 | 96 | 120 | 1028 | 400 |
| sadness | 1271 | 62 | 192 | 60 | 449 | 400 |
| surprise | x | x | 192 | 60 | 225 | 400 |
| Language: | English | German | English | English | Persian | English |

Datasets used for this project:
* EMODB (german) - https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb
* CREMAD (english) - https://github.com/CheyneyComputerScience/CREMA-D
* RAVDESS (english) - https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
* SAVEE (english) - https://www.kaggle.com/datasets/barelydedicated/savee-database
* TESS (english) - https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
* SHEMO (persian) - https://www.kaggle.com/datasets/mansourehk/shemo-persian-speech-emotion-detection-database
