import os
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime

from utils import eval_util
from omegaconf import OmegaConf

LABEL_INFO = {
    "MANIPULATION":{
        "genuine": 0,
        "Unknown": 1,
        "inpainting": 2,
        "separation": 3,
        "edit": 4,
        "generation": 5,
        "splice": 6,
        "add": 7,
    },
    "SOURCE":{
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
    },
    "BINARY":{
        "genuine": 0,
        "fake": 1,
    }
}

def cal_accuracy(pred, target, processed=False) -> float:
    
    if not processed:
        pred = np.concatenate(pred, axis=0) 
        target = np.concatenate(target, axis=0)

        if pred.shape[-1] > 1:
            pred = np.argmax(pred, axis=1) 

    correct = (pred == target).sum().item()
    accuracy = round(correct / target.shape[0], 4)

    return accuracy, pred, target

def convert_timestampStr_to_sedEvalList(onset_offset_str, audio_id):
    output = []
    for onset_offset in onset_offset_str.split("_"):
        [ref_onset, ref_offset] = onset_offset.split("-") 
        output.append({
            'event_label': 'fake',
            'onset': float(ref_onset),
            'offset': float(ref_offset),
            'filename': audio_id,
        })
    return output

def parse_args():
    parser = argparse.ArgumentParser(description="Test a deepfake audio detection model.")
    parser.add_argument(
        "--result_path", '-e', type=str, default="ckpts/EAT_Detection_10/multiTask_inp_sep_edit_gen_add_splice/fakesound2_test_filtered_best_result_per_audio",
        help="Load exp ."
    )
    parser.add_argument(
        "--output_file_name", '-o', type=str, default="score.json",
        help="Path for output file."
    )
    args = parser.parse_args()

    
    args.output_file_name = os.path.join(args.result_path, args.output_file_name)    
            
    return args
    
def main():
    args = parse_args()
    
    result_all = {}
    
    progress_bar = tqdm(os.listdir(args.result_path), desc="Initializing", ncols=100)
    for result_file in progress_bar:
        progress_bar.set_description(f"Processing test set {result_file}")
        if "score" in os.path.basename(result_file): continue
        test_set = os.path.basename(result_file).split('.')[0]
        
        with open(os.path.join(args.result_path, result_file), "r") as f_read:
            result_df = pd.DataFrame([json.loads(line) for line in f_read])

        result_item = { 
            "Num_samples": len(result_df),
        }
        # Convert name to index for calculation
        for task in LABEL_INFO:
            for name in ["pred", "target"]:
                column_name = f"{name}_{task.lower()}"
                result_df[column_name] = result_df[column_name].map(LABEL_INFO[task])
            accuracy, _, _ = cal_accuracy(result_df[f"pred_{task.lower()}"], result_df[f"target_{task.lower()}"], processed=True)
            result_item[f"Accuracy_{task}"] = accuracy

        target_list, pred_list = [], []
        for index, row in result_df.iterrows():
            target_list.extend(convert_timestampStr_to_sedEvalList(row['target_f1segment'] ,row['audio_id']))
            pred_list.extend(convert_timestampStr_to_sedEvalList(row['pred_f1segment'] ,row['audio_id']))
       
        result_item["F1_segment"] = eval_util.calculate_sed_metric(target_list, pred_list),
        print("\n", result_item)
        result_all[test_set] = result_item
    json.dump(result_all, open(args.output_file_name, "w"), indent=4)
            
if __name__ == "__main__":
    main()
