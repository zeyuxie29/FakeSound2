import os
import json
import numpy as np
import argparse
from tqdm.auto import tqdm
from datetime import datetime
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import SchedulerType, get_scheduler

from models import detection_model
from utils.util import DeepfakeDataset, set_seed, MultiTaskLoss, TYPE_LABELS, MODEL_LABELS

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deepfake audio detection model.")
    parser.add_argument(
        "--train_file", "-f",
        nargs='+',
        default=[
            "data/train/genuine.json",
            "data/train/inpainting/train_tango2.json",
            "data/train/inpainting/train_affusion.json",
            "data/train/inpainting/train_ldm2.json",
            "data/train/separation/train_lassnet.json",
            "data/train/editing/train_audioeditor.json",
            "data/train/splicing/train_manually.json",
            "data/train/addition/train_manually.json",
            "data/train/generatation/train_affusion.json",
        ],  
        help="Path to training file(s). Can pass multiple files by '--train_file file1.json file2.json' ."
    )
    parser.add_argument(
        "--batch_size", '-b', type=int, default=196,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--pretrained", '-p', type=str, default=None,
        help="Load pretrained model.",
    )
    parser.add_argument(
        "--learning_rate", '-lr', type=float, default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--loss_weight", '-lw', type=dict, default={"BCE":0.5, "MODEL":0.25, "TYPE":0.25},
        help="Loss weights for multitask.",
    )
    parser.add_argument(
        "--loss_weight_bce", '-lwb', type=float, default=None,
        help="Loss weights for multitask.",
    )
    parser.add_argument(
        "--loss_weight_model", '-lwm', type=float, default=None,
        help="Loss weights for multitask.",
    )
    parser.add_argument(
        "--loss_weight_type", '-lwt', type=float, default=None,
        help="Loss weights for multitask.",
    )
    parser.add_argument(
        "--num_epochs", '-e', type=int, default=10,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--output_dir", '-o', type=str, default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--model_name", '-m', type=str, default="EAT_Detection", # WavLM_Detection
        help="name of model_name"
    )
    parser.add_argument(
        "--feature_extractor_name", '-fm', type=str, default="EAT", # WavLM
        help="name of extractor"
    )
    parser.add_argument(
        "--duration", '-d', type=float, default=10,
        help="Audio duration."
    )
    parser.add_argument(
        "--time_resolution", type=float, default=0.02,
        help="."
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help="."
    )
    parser.add_argument(
        "--num_examples", '-n', type=int, default=-1,
        help="How many examples to use for training.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="A seed for reproducible training."
    )
    args = parser.parse_args()

    if args.loss_weight_bce != None:
        args.loss_weight["BCE"] = args.loss_weight_bce
    if args.loss_weight_model != None:
        args.loss_weight["MODEL"] = args.loss_weight_model
    if args.loss_weight_type != None:
        args.loss_weight["TYPE"] = args.loss_weight_type

    return args



def main():
    args = parse_args()
    args.label_info = {
        "TYPE": TYPE_LABELS, 
        "MODEL": MODEL_LABELS,
    }
    args_dict = vars(args)
    print(json.dumps(args_dict, indent=4))
    # If passed along, set the training seed now.
    set_seed(args.seed)

    # Handle output directory creation
    if args.output_dir is None or args.output_dir == "":
        args.output_dir = f"{WORKSPACE_PATH}/ckpts/{args.model_name}_{args.num_epochs}/multiTask_inp_sep_edit_gen_add_splice"        
    elif args.output_dir is not None:
        args.output_dir = f"{WORKSPACE_PATH}/ckpts/{args.model_name}_{args.num_epochs}/{args.output_dir}"
    os.makedirs(args.output_dir, exist_ok=True)
    with open("{}/args.json".format(args.output_dir), "w") as f:
        f.write(json.dumps(dict(vars(args)), indent=4))
    summary_json_path = "{}/summary.jsonl".format(args.output_dir)

    # Init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = getattr(detection_model, args.model_name)(label_info=args.label_info).to(device)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    train_dataset = DeepfakeDataset(args.train_file, args)       
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    
    # Optimizer
    criterion = MultiTaskLoss(weight=args.loss_weight)
    optimizer_parameters = model.parameters()
    if hasattr(model, "future_extractor"):
        for param in model.future_extractor.parameters():
            param.requires_grad = False
            model.future_extractor.eval()
            optimizer_parameters = model.backbone.parameters()
        print("Optimizing backbone parameters.")
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    train_info = {
        "Num instances in train": len(train_dataset),
        "Num trainable parameters": num_trainable_parameters
    }
    print(train_info)
    with open(summary_json_path, "w") as f:      
        f.write(json.dumps(train_info, indent=4) + '\n\n')

    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=2e-4,
        eps=1e-08,
    )

    num_update_steps_per_epoch = len(train_dataloader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.batch_size
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_epochs}")
    print(f"  Instantaneous batch size per device = {args.batch_size}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))

    completed_steps = 0
    starting_epoch = 0  
    # Duration of the audio clips in seconds
    best_loss, best_epoch = np.inf, 0

    for epoch in range(starting_epoch, args.num_epochs):
        model.train()
        total_loss = {loss_name: 0 for loss_name in args.loss_weight}
        total_loss["sum"] = 0

        print(f"train epoch {epoch} begin!")
        for step, batch in enumerate(train_dataloader):
            
            audio = batch["audio"]
            output_dict = model(audio.to(device))
            loss = criterion(output_dict, batch)
            
            loss['sum'].backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            for loss_name in total_loss:
                total_loss[loss_name] += loss[loss_name].detach().float()

            progress_bar.update(1)
            completed_steps += 1
 
        print(f"train epoch {epoch} finish!")
 
        result = {}
        result["epoch"] = epoch,
        result["step"] = completed_steps
        for loss_name in total_loss:
            total_loss[loss_name] = round(total_loss[loss_name].item()/len(train_dataloader), 4)
        result["train_loss"] = total_loss

        if total_loss["sum"] < best_loss:
            best_loss = total_loss["sum"]
            best_epoch = epoch
            torch.save(model.state_dict(), f"{args.output_dir}/best.pt")
        torch.save(model.state_dict(), f"{args.output_dir}/last.pt")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/epoch{epoch}.pt")
        result["best_eopch"] = best_epoch
        print(result)
        result["time"] = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

        with open(summary_json_path, "a") as f:
            f.write(json.dumps(result) + "\n\n")
  
            
if __name__ == "__main__":
    main()
