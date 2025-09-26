import json
import torch
import torch.nn as nn
import random
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader

MODEL_LABELS = {
    "None": 0,
    "Unknown": 1,
    "manually": 2,
    "tango2": 3,
    "affusion": 4,
    "ldm2": 5,
    "fakesound1": 6,
    "lassnet": 7,
    "audioeditor": 8,
    #"makeanaudio": 9,
    #"flowsep": 10,
    #"audit": 11,
    #"x2audio": 12,
    #"ldm": 13,
}

TYPE_LABELS = {
    "genuine": 0,
    "Unknown": 1,
    "inpainting": 2,
    "separation": 3,
    "edit": 4,
    "generation": 5,
    "splice": 6,
    "add": 7,
}

class MultiTaskLoss(nn.Module):
    def __init__(self, weight={"BCE":0.5, "MODEL":0.25, "TYPE":0.25}, reduction='mean'):
        """
        :param weight: mutiltask weights
        :param reduction: 'none'|'mean'|'sum'
        """
        super(MultiTaskLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.BinaryCriterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.CECriterion = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.task_to_loss = {
            "BCE": self.BinaryCriterion,
            "MODEL": self.CECriterion,
            "TYPE": self.CECriterion,
        }
        
    def forward(self, output_dict, target_dict):
        loss_dict = {}

        for task in self.task_to_loss:
            pred, target = output_dict[f"pred_{task}"], target_dict[f"target_{task}"]
            target = target.to(pred.device)
            loss = self.task_to_loss[task](pred, target)
    
            loss_dict[task] = loss * self.weight[task]

        loss_dict["sum"] = sum(loss_dict.values())
        return loss_dict

def wavlm_preprocess_fn(source):
    # [N, 160080] in batch
    # audio, sr = librosa.load(source_file, sr=self.sample_rate)
    # audio = np.concatenate((audio, np.zeros(80)), axis=0) # for WavLM alignment  
    target_length = 160080
    lens = source.shape[0]
    diff = target_length - lens
    if diff >= 0:
        source = torch.cat((source, torch.zeros(diff)), axis=0)
    else:
        source = source[0:target_length]
        
    assert source.shape == (160080,), f"source.shape: {source.shape}"
    return source

def eat_preprocess_fn(source):
    # [N, 1, 1024, 128] in batch
    target_length, norm_mean, norm_std  = 1024, -4.268,  4.569  
    
    source = source - source.mean()
    source = source.unsqueeze(dim=0)
    source = torchaudio.compliance.kaldi.fbank(source, htk_compat=True, sample_frequency=16000, use_energy=False,
        window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(dim=0)
    
    n_frames = source.shape[1]
    diff = target_length - n_frames
    if diff > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
        source = m(source)
        
    elif diff < 0:
        source = source[0:target_length, :]
                
    source = (source - norm_mean) / (norm_std * 2)
    return source

class DeepfakeDetectionDataset(Dataset):
    def __init__(self, data_file, args):
         
        if isinstance(data_file, list): 
            self.data = []
            for file_path in data_file:
                with open(file_path, "r") as reader:
                    sub_data_list = [json.loads(line) for line in reader]
                    self.data.extend(sub_data_list)
                print(f"Load dataset <<<< {file_path} >>>>, lens: {len(sub_data_list)}")
        else:
            with open(data_file, "r") as reader:
                self.data = [json.loads(line) for line in reader]
        self.time_resolution = args.time_resolution
        self.duration = args.duration
        self.sample_rate = args.sample_rate

        if args.num_examples != -1:
            random.shuffle(self.data)
            self.data = self.data[:args.num_examples]

        self.feature_extractor_name = args.feature_extractor_name
        assert self.feature_extractor_name in ["WavLM", "EAT"], \
            f"feature_extractor {self.feature_extractor_name} unsupported"
        self.source_preprocess = wavlm_preprocess_fn if self.feature_extractor_name == "WavLM" else eat_preprocess_fn
            
    def __len__(self):
        return len(self.data)

    def _load_wav(self, source_file):
        assert source_file.endswith('.wav'), "the standard format of file should be '.wav' "
        
        wav, sr = torchaudio.load(source_file)
        source = wav.mean(dim=0)

        if sr != 16e3: 
            source = torchaudio.functional.resample(source, orig_freq=sr, new_freq=self.sample_rate).float()  

        source = self.source_preprocess(source)

        return source.numpy()

    def __getitem__(self, index):
        item = self.data[index]

        audio = self._load_wav(item["filepath"])

        #binary_label = int(item["label"]) # fake->1, real->0
        label = item["fake_type"]
        binary_label = 0 if label == "genuine" else 1
        tgt = np.zeros(int(self.duration / self.time_resolution))
        if binary_label == 1:
            for onset_offset in item["onset_offset"].split("_"):
                [onset, offset] = onset_offset.split("-")
                tgt[int(float(onset) / self.time_resolution): int(float(offset) / self.time_resolution)] = 1

        return audio, binary_label, tgt, item["audio_id"], item["onset_offset"]

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        batch = []
        for i in dat:
            if i==3 or i==4:
                batch.append(dat[i].tolist())
            elif i==1:
                batch.append(np.array(dat[i]))
            elif i==0 or i==2:
                data = torch.tensor(np.array(dat[i].tolist()), dtype=torch.float32)
                batch.append(data)   
        return batch

class DeepfakeDataset(DeepfakeDetectionDataset):
    def __getitem__(self, index):
        item = self.data[index]

        audio = self._load_wav(item["filepath"])

        #binary_label = int(item["label"]) # fake->1, real->0
        label = item["fake_type"]
        binary_label = 0 if label == "genuine" else 1
        tgt = np.zeros(int(self.duration / self.time_resolution))
        if binary_label == 1:
            for onset_offset in item["onset_offset"].split("_"):
                [onset, offset] = onset_offset.split("-")
                tgt[int(float(onset) / self.time_resolution): int(float(offset) / self.time_resolution)] = 1

        model_label = item["model"]
        
        model_label_idx = MODEL_LABELS[model_label] if model_label in MODEL_LABELS else MODEL_LABELS["Unknown"]
        type_label = item["fake_type"]
        type_label_idx = TYPE_LABELS[type_label]

        return {
            "audio": audio, 
            "audio_id": item["audio_id"], 
            "onset_offset": item["onset_offset"],
            "target_BCE": tgt, 
            "target_MODEL": model_label_idx,
            "target_TYPE": type_label_idx,
            "model": model_label,
            "type": type_label,
            "binary_label": binary_label,
        }

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        batch = {}

        for i in dat:
            if i in ["audio", "target_BCE"]:
                batch[i] = torch.tensor(np.array(dat[i].tolist()), dtype=torch.float32)
            elif i in ["target_MODEL", "target_TYPE", "binary_label"]:
                batch[i] = torch.tensor(np.array(dat[i].tolist()), dtype=torch.long)
            else:
                batch[i] = dat[i].tolist()
        return batch
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False