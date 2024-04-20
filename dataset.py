import torch
from torch.utils.data import Dataset
import scipy
import torchaudio
import pandas as pd
import os
import glob
import numpy as np

EMOTIONS = {
    "neutral": 0,
    "happy": 1,
    "sadness": 2,
    "disgust": 3,
    "anger": 4,
    "surprise": 5,
    "fear": 6,
    "anxiety": 7,
    "bored": 8,
    "calm": 9
}

def emo2onehot(emotion):
    """
    Convert emotion string to one-hot vector
    :param emotion: string
    :return: numpy.array
    """
    #v = np.zeros(len(EMOTIONS))
    v = torch.zeros(len(EMOTIONS), dtype=torch.uint8)
    emonum = EMOTIONS[emotion]
    if emonum != 0:
        v[emonum-1] = 1
    return v

#-------------------------------
#region read datasets into pandas dataframe

def _get_emodb_info(pdir):
    datadir = os.path.join(pdir, 'berlin-database-of-emotional-speech-emodb', 'wav')
    if os.path.exists(datadir):
        wav_list = glob.glob(datadir + os.sep + '*.wav')
        l = []
        emotions_dict = {
            "F": "happy",
            "W": "anger",
            "L": "bored",
            "E": "disgust",
            "A": "anxiety",
            "T": "sadness",
            "N": "neutral"
        }
        for f in wav_list:
            new_row = {}
            fname = os.path.basename(f)
            spkr = int(fname[0:2])
            emotion = fname[5]
            version = fname[6]
            new_row['filename'] = f
            new_row['speaker_n'] = spkr
            new_row['language'] = "german"
            new_row["emotion"] = emotions_dict[emotion]
            new_row['version'] = version
            new_row["database"] = "emodb"
            new_row["intensity"] = 'NA'
            l.append(new_row)

        return pd.DataFrame(l)
    else:
        raise FileNotFoundError("Download the emodb dataset")
        return None

def _get_cremad_info(pdir):
    datadir = os.path.join(pdir, 'cremad', 'AudioWAV')
    if os.path.exists(datadir):
        wav_list = glob.glob(datadir + os.sep + '*.wav')
        emo_intensities = {
            "LO": "low",
            "MD": "medium",
            "HI": "high",
            "XX": "unknown",
            "X": "unknown"
        }
        emotions_dict = {
            "HAP": "happy",
            "ANG": "anger",
            "DIS": "disgust",
            "SAD": "sadness",
            "NEU": "neutral",
            "FEA": "fear"
        }
        l = []
        for f in wav_list:
            new_row = {}
            fname = os.path.basename(f)
            spkr, _, emotion, intensity = fname.rstrip('.wav').split('_')
            new_row['filename'] = f
            new_row['speaker_n'] = int(spkr)
            new_row['language'] = "english"
            new_row["emotion"] = emotions_dict[emotion]
            new_row['version'] = 'NA'
            new_row["database"] = "cremad"
            new_row["intensity"] = emo_intensities[intensity]
            l.append(new_row)

        return pd.DataFrame(l)
    else:
        raise FileNotFoundError("Download the cremad dataset")
        return None

def _get_ravdess_info(pdir):
    datadir = os.path.join(pdir, 'ravdess-emotional-speech-audio')
    if os.path.exists(datadir):
        wav_list = glob.glob(os.path.join(datadir, '**', '*.wav'))
        emo_intensities = {
            "01": "normal",
            "02": "strong",
        }
        emotions_dict = {
            "03": "happy",
            "05": "anger",
            "07": "disgust",
            "04": "sadness",
            "01": "neutral",
            "02": "calm",
            "06": "fear",
            "08": "surprise"
        }
        version_dict = {
            "01": "speech",
            "02": "song"
        }
        l = []
        for f in wav_list:
            new_row = {}
            fname = os.path.basename(f)
            mod, vocal, emotion, intensity, statement, rep, actor = fname.rstrip(".wav").split("-")
            new_row['filename'] = f
            new_row['speaker_n'] = int(actor)
            new_row['language'] = "english"
            new_row["emotion"] = emotions_dict[emotion]
            new_row['version'] = version_dict[vocal] + "_" + rep
            new_row["database"] = "ravdess"
            new_row["intensity"] = emo_intensities[intensity]
            l.append(new_row)

        return pd.DataFrame(l)
    else:
        raise FileNotFoundError("Download the ravdess dataset")
        return None

def _get_savee_info(pdir):
    datadir = os.path.join(pdir, 'savee-database','AudioData')
    if os.path.exists(datadir):
        wav_list = glob.glob(os.path.join(datadir, '**', '*.wav'))
        emotions_dict = {
            "h": "happy",
            "a": "anger",
            "d": "disgust",
            "sa": "sadness",
            "n": "neutral",
            "f": "fear",
            "su": "surprise"
        }
        spkr_dict = {
            "DC": 1,
            "JE": 2,
            "JK": 3,
            "KL": 4
        }
        l = []
        for f in wav_list:
            new_row = {}
            dir_list = f.split(os.sep)
            fname = dir_list[-1]
            spkr = dir_list[-2]
            if len(fname) == 7:
                emotion = emotions_dict[fname[0]]
                version = fname[1:3]
            else:
                emotion = emotions_dict[fname[0:2]]
                version = fname[2:4]

            new_row['filename'] = f
            new_row['speaker_n'] = spkr_dict[spkr]
            new_row['language'] = "english"
            new_row["emotion"] = emotion
            new_row['version'] = version
            new_row["database"] = "savee"
            new_row["intensity"] = "NA"
            l.append(new_row)

        return pd.DataFrame(l)
    else:
        raise FileNotFoundError("Download the savee dataset")
        return None

def _get_tess_info(pdir):
    datadir = os.path.join(pdir, 'toronto-emotional-speech-set-tess','TESS Toronto emotional speech set data')
    if os.path.exists(datadir):
        wav_list = glob.glob(os.path.join(datadir, '**', '*.wav'))
        emotions_dict = {
            "happy": "happy",
            "angry": "anger",
            "disgust": "disgust",
            "sad": "sadness",
            "neutral": "neutral",
            "fear": "fear",
            "ps": "surprise"
        }
        spkr_dict = {
            "oaf": 1,
            "oa": 1,
            "yaf": 2,
        }
        l = []
        for f in wav_list:
            new_row = {}
            fname = os.path.basename(f).lower()
            spkr, version, emotion = fname.rstrip(".wav").split("_")
            new_row['filename'] = f
            new_row['speaker_n'] = spkr_dict[spkr]
            new_row['language'] = "english"
            new_row["emotion"] = emotions_dict[emotion]
            new_row['version'] = version
            new_row["database"] = "tess"
            new_row["intensity"] = "NA"
            l.append(new_row)

        return pd.DataFrame(l)
    else:
        raise FileNotFoundError("Download the tess dataset")
        return None


def _get_shemo_info(pdir):
    datadir = os.path.join(pdir, 'shemo-persian-speech-emotion-detection-database')
    if os.path.exists(datadir):
        wav_list = glob.glob(os.path.join(datadir, '**', '*.wav'))
        emotions_dict = {
            "h": "happy",
            "a": "anger",
            "s": "sadness",
            "w": "surprise",
            "f": "fear",
            "n": "neutral"
        }
        l = []
        for f in wav_list:
            new_row = {}
            fname = os.path.basename(f).lower()
            new_row['filename'] = f
            new_row['speaker_n'] = fname[1:3]
            new_row['language'] = "persian"
            new_row["emotion"] = emotions_dict[fname[3]]
            new_row['version'] = int(fname[4:6])
            new_row["database"] = "shemo"
            new_row["intensity"] = "NA"
            l.append(new_row)

        return pd.DataFrame(l)
    else:
        raise FileNotFoundError("Download the shemo dataset")
        return None

#endregion


# Create list of datasets
DATASETS = {
    "emodb": {
        "url": "https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb",
        "get": _get_emodb_info
    },
    "tess": {
        "url": "https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess",
        "get": _get_tess_info
    },
    "cremad": {
        "url": "https://www.kaggle.com/datasets/ejlok1/cremad",
        "get": _get_cremad_info
    },
    "ravdess": {
        "url": "https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio",
        "get": _get_ravdess_info
    },
    "savee": {
        "url": "https://www.kaggle.com/datasets/barelydedicated/savee-database",
        "get": _get_savee_info
    },
    "shemo": {
        "url": "https://www.kaggle.com/datasets/mansourehk/shemo-persian-speech-emotion-detection-database",
        "get": _get_shemo_info
    }
}

#-------------------------------
#region functions to be used by users

def download_datasets(pdir,dname=None):
    """
    Download datasets
    :param pdir: directory to save datasets
    :param dname: Which specific dataset to download. If None is passed then all datasets will be downloaded

    ex: Download only emodb dataset
    dataset_dir = "/content/drive/MyDrive/EC523/term_project/datasets"
    os.chdir("/content/drive/MyDrive/EC523/term_project")
    download_datasets(dataset_dir, dname="emodb")
    """
    import opendatasets as od

    if dname is not None:
        datasets = {dname: DATASETS[dname]}
    else:
        datasets = DATASETS

    for k, v in datasets.items():
        if not os.path.isdir(pdir + os.sep + os.path.basename(v["url"])):
            od.download(v["url"], data_dir=pdir)

    return

class SpeechEmotionDataset(Dataset):
    """Speech emotion dataset."""

    def __init__(self, dataframe, fs=None, transform=None, onehot=True):
        self.dataframe = dataframe
        self.transform_fs = fs
        self.transform = transform
        self.onehot = onehot

    def __len__(self):
        return len(self.dataframe)

    def _read_audio_torch(self,fname, resample_fs=None):
        # Load audio data
        aud, fs = torchaudio.load(fname)
        aud = aud[0,:]

        # Resample
        if resample_fs is not None:
          aud = torchaudio.functional.resample(aud,fs,resample_fs)
          fs = resample_fs

        return aud, fs

    def _read_wav_scipy(self,fname, resample_fs=None):
        # Load audio data
        fs, aud = scipy.io.wavfile.read(fname)

        if aud.ndim == 2:
          if len(aud[:,0])>len(aud[0,:]):
            aud = aud[:,0]
          else:
            aud = aud[0,:]

        # Resample
        aud_dtype = aud.dtype
        n_sample = aud.size

        if resample_fs is not None:
          aud_time = n_sample/fs
          new_n_sample = round(aud_time*resample_fs)
          aud = scipy.signal.resample(aud, new_n_sample)
          fs = resample_fs

        return aud, fs

    def __getitem__(self, idx, return_all=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get general data
        if isinstance(idx,str):
          idx_data = self.dataframe.loc[idx].to_dict()
        else:
          idx_data = self.dataframe.iloc[idx].to_dict()

        # Load audio data
        #aud, fs = self._read_wav_scipy(idx_data["filename_full"], resample_fs=self.transform_fs)
        aud, fs = self._read_audio_torch(idx_data["filename"], resample_fs=self.transform_fs)

        # Apply transformations
        if self.transform:
          aud = self.transform(aud)

        idx_data["audio"] = aud
        idx_data["fs"] = fs

        if self.onehot:
            idx_data["emotion"] = emo2onehot(idx_data["emotion"])

        if return_all:
          return idx_data
        else:
          return idx_data["audio"], idx_data["emotion"]

def get_dataset_info(pdir,dname=None):
    """
    Get dataset info for one or all datasets
    :param pdir: The directory containing the datasets
    :param dname: Datasets to get info on. Can be None for all datasets, list for subset or str for one
    :return: pandas.DataFrame containing dataset info
    """

    df_columns = ['filename', 'speaker_n', 'intensity', 'emotion', "version", "language", "database"]
    df = pd.DataFrame(columns=df_columns)

    if isinstance(dname,str):
        dname = [dname]

    if isinstance(dname,list):
        dsets = {}
        for d in dname:
            dsets[d] = DATASETS[d]
    else:
        dsets = DATASETS.copy()

    for k, v in dsets.items():
        if "get" in v.keys():
            df = pd.concat([df, v["get"](pdir)], ignore_index=True)
        else:
            print(f"The {k} dataset doesn't exist or doesn't have a `get` function")

    return df

def extract_dataset_features(data_filepath, df, dname=None):
    """
    param data_filepath : folder that contains all of the .wav files from a single database
    param df : dataframe that is the output of the get_dataset_info() function
    """
    
    import librosa
    import numpy as np
    
    if data_filepath[-1] != "/":
        data_filepath = data_filepath + "/"
    
    # needs "/" at end of filepath
    #data_filepath = "/home/arthurus-rex/Documents/EC523/Project/berlin-database-of-emotional-speech-emodb/wav/"

    database_name = "berlin-database-of-emotional-speech-emodb"
    filenames_list = os.listdir(data_filepath) # filenames without full filepath
    full_filenames_list = [data_filepath + filename for filename in filenames_list] # adding full filepath
    
    features_df = pd.DataFrame(columns=['filename', 'mfccs', 'rms', 'zcr', 'emotion'])

    # works for a single folder of audio files at once, all from the same dataset
    for i, filename in enumerate(full_filenames_list):
        X, sample_rate = librosa.load(filename) # load waveform
        
        mfccs = np.mean(librosa.feature.mfcc(y=X, n_mfcc=25,), axis = 0) # calculate feature values
        rms = librosa.feature.rms(y=X)
        zcr = librosa.feature.zero_crossing_rate(y=X)
        
        features_df.at[i, 'mfccs'] = mfccs # save X features to dataframe
        features_df.at[i, 'rms'] = np.array(rms)
        features_df.at[i, 'zcr'] = np.array(zcr)
    
        row_index = df.loc[df['filename'] == filename].index[0] # finding the correct emotion for the filename
        features_df.at[i, 'emotion'] = df.at[row_index,'emotion'] # selects correct emotion from Noah's df
    
        split_name = filename.split('/') # get the correct filename
        features_df.at[i, 'filename'] = split_name[-1]
    
    return features_df    
#endregion
