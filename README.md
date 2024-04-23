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

We chose 6 popular databases of audio, each containing collections of raw audio with corresponding emotion labels. Each dataset different emotion labels and different numbers of .wav files for each emotion, detailed in the table below:

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

Then, we trained and fine-tined three different models (Wav2Vec, CNN, and S4) on these datasets (70% training, 20% testing, 10% validation).  Specifically, we used the four emotions common to all models ```[anger, happy, neutral, sadness]```, but the functionality exists to use all emotions independently within the code.

# Our Models

## Wav2Vec
We adapted an existing Wav2Vec model and fine-tuned it on our specific datasets.
https://huggingface.co/docs/transformers/model_doc/wav2vec2

## CNN
We built a CNN similar to the architecture described in the following source:
```
W. Alsabhan. Human-Computer Interaction with a Real-Time Speech Emotion Recognition with Ensembling Techniques 1D Convolution Neural Network with Attention, Sensors, 2023. https://doi.org/10.3390/s23031386
```

## S4
We adapted the S4 implementation taken from the following repository:
https://github.com/state-spaces/s4

# Implementing our approach
Each of our models is contained within its own Jupyter notebook and draws from the same functions specified in datasets.py.  The datasets must be downloaded locally before training, which can be done simply with the code sample above.



