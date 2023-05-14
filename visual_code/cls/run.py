import subprocess
import argparse
import torch

parser = argparse.ArgumentParser(description="run")
parser.add_argument("--device", type=str)
parser.add_argument("--run", type=int)
args = parser.parse_args()


def get_optim(model):
    if model == "dgcnn":
        return "sgd"
    else:
        return "adam"


if args.run == 0:
        model = "dgcnn"
        data = "SONN_EASY"
        optim = get_optim(model)
        sigma=0.3
        subprocess.call(f"CUDA_VISIBLE_DEVICES='{args.device}' python main.py --model {model} --exp_name saliencytest_{model}_{data}_sigma{sigma} \
                        --data {data} --epochs 500 --kermix --sigma {sigma} --optim {optim} --wandb", shell=True)
    