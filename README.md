# Compact SiT (Self-supervised vIsion Transformer) for HistoPathology Images

This repository contains the official PyTorch self-supervised pretraining, finetuning, and evaluation codes for compact SiT (Self-supervised image Transformer) with main focus on HistoPathology images.

This repo is mainly adopted from [SiT Repository](https://github.com/Sara-Ahmed/SiT) with some modifications and improvements. Some features were added and the backbone of the model was replaced with [CCT-14/7x2](https://github.com/SHI-Labs/Compact-Transformers)(with some changes).

# Instaling Requirements
> pip install --upgrade -r requirements.txt || true

# Downloading Pre-Trained Model
> gdown --id 1ypMbvfmOAxLa8pU7A_jf3v7gayxy1zzf             # 600k_14epochs

# Self-supervised pre-training
> python main.py --batch-size 72 --epochs 10 --min-lr 5e-6 --lr 1e-4 --training-mode 'SSL' --dataset 'UH_main' --output 'output' --validate-every 1 

<!-- Self-supervised pre-trained models using SiT on our unlabeled HistoPatholgy Dataset can be downloaded from [here]() -->

# Finetuning
**Finetuning with prepared dataset**
> python main.py  --batch-size 120 --epochs 50 --min-lr 5e-6 --training-mode 'finetune' --dataset 'NCT' --finetune '<<path/to/pretrained_model>>' --output 'output' --validate-every 1 

**Finetuning with custom dataset**
> python main.py  --batch-size 120 --epochs 50 --min-lr 5e-6 --training-mode 'finetune' --dataset 'Custom' --custom_train_dataset_path '<<path/to/train_dataset>>' --custom_val_dataset_path '<<path/to/val_dataset>>' --finetune 'output/checkpoint.pth' --output 'output' --validate-every 1 

# Linear Evaluation

**Linear projection Head**
> python main.py  --batch-size 120 --epochs 50 --min-lr 5e-6 --training-mode 'finetune' --dataset 'NCT' --finetune '<<path/to/pretrained_model>>' --output 'output' --validate-every 1  --SiT_LinearEvaluation 1 

**2-layer MLP projection Head**
> python main.py  --batch-size 120 --epochs 10 --lr 1e-3 --weight-decay 5e-4 --min-lr 5e-6 --training-mode 'finetune' --dataset 'NCT' --finetune '<<path/to/pretrained_model>>' --output 'output' --validate-every 1 --SiT_LinearEvaluation 1 --representation-size 1024

**Note: assign the --dataset_location parameter to the location of the downloaded dataset**



