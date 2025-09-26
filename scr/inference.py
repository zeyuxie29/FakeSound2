import os
import json
import librosa
import numpy as np
import pandas as pd
import random
import argparse
from tqdm import tqdm
from datetime import datetime
import torch

import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import SchedulerType, get_scheduler
import sed_eval
import dcase_util

from utils.util import DeepfakeDataset, set_seed
from utils import eval_util
from models import detection_model
from omegaconf import OmegaConf
from pathlib import Path
from evaluation import LABEL_INFO, cal_accuracy, convert_timestampStr_to_sedEvalList

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def parse_args():
    parser = argparse.ArgumentParser(description="Test a deepfake audio detection model.")
    parser.add_argument(
        "--test_files_yaml", '-f', type=str, default="utils/fakesound2_test_filtered.yaml", help="Test data list.",
    )
    parser.add_argument(
        "--batch_size", '-b', type=int, default=16,
        help="Batch size (per device) for the dataloader.",
    )
    parser.add_argument(
        "--threshold", '-th', type=float, default=0.5,
        help=".",
    )
    parser.add_argument(
        "--exp_path", '-e', type=str, default="ckpts/EAT_Detection_10/multiTask_inp_sep_edit_gen_add_splice",
        help="Load exp ."
    )
    parser.add_argument(
        "--original_args", type=str, default="args.json",
        help="Path for args jsonl file saved during training."
    )
    parser.add_argument(
        "--output_file_name", '-o', type=str, default=None,
        help="Path for output file."
    )
    parser.add_argument(
        "--model_pt", type=str, default="best.pt",
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--per_audio", action="store_true",
        help="Save score for each audio."
    )
    args = parser.parse_args()

    if args.output_file_name == None:
        args.output_file_name = f"{os.path.basename(args.test_files_yaml).split('.')[0]}_{args.model_pt.split('.')[0]}_result"
        
    args.original_args = os.path.join(args.exp_path, args.original_args)    
    args.model_pt = os.path.join(args.exp_path, args.model_pt)

    
    return args
    
def main():
    args = parse_args()
    train_args = dotdict(json.load(open(args.original_args)))

    # If passed along, set the training seed now.
    set_seed(train_args.seed)
    
    # Init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model
    model = getattr(detection_model, train_args.model_name)(label_info=train_args.label_info).to(device).eval()
    model.load_state_dict(torch.load(args.model_pt))   
    result_all = {}

    config = OmegaConf.load(args.test_files_yaml)
    test_files = OmegaConf.to_container(config.test_files, resolve=True)
    idx2name = {}
    for task in LABEL_INFO:
        idx2name[task] = {
            v: k for k, v in LABEL_INFO[task].items()
        }
  
    for test_set, test_file in  test_files.items():
        dataset = DeepfakeDataset(test_file, train_args)       
        dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
        print(F"Num instances in test set <<<< {test_set} >>>>: {len(dataset)}")

        audio_id_list = []
        ref_list, pred_list, ref_str_list, pred_str_list = [], [], [], []
        target_binary_list, target_model_list, target_type_list, pred_binary_list, pred_model_list, pred_type_list = [], [], [], [], [], []
        time_resolution, threshold = train_args.time_resolution, args.threshold
        for step, batch in enumerate(tqdm(dataloader)):
            audio, binary_label, tgt, audio_id, onset_offset_str = \
                batch["audio"], batch["binary_label"], batch["target_BCE"], batch["audio_id"], batch["onset_offset"]
            with torch.no_grad():

                output = model(audio.to(device))
                """
                The output should contain:
                    pred_BCE: prob for binary classification
                    pred_MODEL: prob for source classification
                    pred_TYPE: prob for manipulation type classification
                """ 
                pred = torch.sigmoid(output["pred_BCE"].unsqueeze(-1)).cpu().numpy()
                filtered_prob = eval_util.median_filter(pred, window_size=1, threshold=threshold)

            # Accuracy
            audio_id_list.extend(audio_id)
            pred_binary = np.max(filtered_prob, axis=1) > threshold
            pred_binary_list.append(pred_binary)
            target_binary_list.append(binary_label.unsqueeze(-1).cpu().numpy())
            pred_model_list.append(output["pred_MODEL"].cpu().numpy())
            target_model_list.append(batch["target_MODEL"].cpu().numpy())
            pred_type_list.append(output["pred_TYPE"].cpu().numpy())
            target_type_list.append(batch["target_TYPE"].cpu().numpy())
            
            # Segment f1
            for idx, _ in enumerate(pred):         
                # pred_info
                change_indices = eval_util.find_contiguous_regions(filtered_prob[idx])
                if len(change_indices) == 0:
                    tmp_str = "0-0"
                else:
                    tmp_str_list = []
                    for row in change_indices:
                        onset_i, offset_i = round(row[0] * time_resolution, 2), round(row[1] * time_resolution, 2)
                        tmp_str_list.append(f"{onset_i}-{offset_i}")
                    tmp_str = "_".join(tmp_str_list)
                pred_str_list.append(tmp_str)   
                pred_list.extend(convert_timestampStr_to_sedEvalList(tmp_str, audio_id[idx]))   

                # ref_info
                ref_str_list.append(onset_offset_str[idx])
                ref_list.extend(convert_timestampStr_to_sedEvalList(onset_offset_str[idx], audio_id[idx]))

        accuracy_binary, pred_binary, target_binary = cal_accuracy(pred_binary_list, target_binary_list)
        accuracy_type, pred_type, target_type = cal_accuracy(pred_type_list, target_type_list)
        accuracy_model, pred_model, target_model = cal_accuracy(pred_model_list, target_model_list)
        result_item = { 
            "Num_samples": len(dataset),
            "Accuracy_MANIPULATION": accuracy_type,
            "Accuracy_SOURCE": accuracy_model,
            "Accuracy_BINARY": accuracy_binary,
            "F1_segment": eval_util.calculate_sed_metric(ref_list, pred_list),
        }

        # Save predicted result for each audio
        if args.per_audio:
            assert len(audio_id_list) == len(dataset)
            assert len(pred_binary) == len(dataset)
            assert len(target_binary) == len(dataset)
            assert len(pred_type) == len(dataset)
            assert len(target_type) == len(dataset)
            assert len(pred_model) == len(dataset)
            assert len(target_model) == len(dataset)
            assert len(pred_str_list) == len(dataset)
            assert len(ref_str_list) == len(dataset)

            output_base = os.path.join(args.exp_path, args.output_file_name + "_per_audio")
            os.makedirs(output_base, exist_ok=True)
            output_per_audio_json = os.path.join(output_base, test_set + ".jsonl")
            with open(output_per_audio_json, "w") as f_per_audio:
                for idx, audio_id_i in enumerate(audio_id_list):
                    result_per_audio = {
                        "audio_id": audio_id_i,
                        "pred_binary": idx2name["BINARY"][int(pred_binary[idx][0] + 0)],
                        "target_binary": idx2name["BINARY"][int(target_binary[idx][0])],
                        "pred_manipulation": idx2name["MANIPULATION"][int(pred_type[idx])],
                        "target_manipulation": idx2name["MANIPULATION"][int(target_type[idx])],
                        "pred_source": idx2name["SOURCE"][int(pred_model[idx])],
                        "target_source": idx2name["SOURCE"][int(target_model[idx])],
                        "pred_f1segment": pred_str_list[idx],
                        "target_f1segment": ref_str_list[idx],
                    }
                    f_per_audio.write(json.dumps(result_per_audio, ensure_ascii=False) + "\n")
        print(result_item, "\n")
        result_all[test_set] = result_item
    json.dump(result_all, open(os.path.join(args.exp_path, args.output_file_name + ".json"), "w"), indent=4)
            
if __name__ == "__main__":
    main()
