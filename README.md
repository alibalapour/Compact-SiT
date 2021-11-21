# Compact SiT (Self-supervised vIsion Transformer) for HistoPathology Images

This repository contains the official PyTorch self-supervised pretraining, finetuning, and evaluation codes for compact SiT (Self-supervised image Transformer) with main focus on HistoPathology images. The SiT model uses three pre-text tasks for self-supervised learning: Reconstruction, Rotation Prediction, and Contrastive Learning. These three tasks are implemented on a Vision Transformer (ViT) model to gain advantages of the attention mechanism. In this work, we replaced the backbone of SiT model and replaced it with a more efficient and less data-hungry vision transformer named CCT (compact convolution transformer). Then we pre-trained and fine-tuned the new model on histopathology images to get better results on this kind of image.  

This repo is mainly adopted from [SiT Repository](https://github.com/Sara-Ahmed/SiT) with some modifications and improvements. Some features were added and the backbone of the model was replaced with [CCT-14/7x2](https://github.com/SHI-Labs/Compact-Transformers)(with some changes).

# Procedure
First, we pre-trained the SiT-Compact model on a diverse unlabeled histopathology dataset consisting of 600k images derived from known datasets (e.g., PatchCamelyon, ICIAR2018, TUPAC2016 Mitosis, and …). this dataset is available in this [link](https://drive.google.com/file/d/1JoJxnY4zPuvjVGIALE_UCpCX97i6HA8J/view?usp=sharing).

<figure >
  <img src="https://user-images.githubusercontent.com/42287060/142759501-a98a6d7a-fe6b-4063-b7ff-c88aedc04a7a.png" class="center" alt="drawing" width="300"/>
  <figcaption>Fig.1 - sample of unlabeled dataset</figcaption>
</figure>

...

# Instaling Requirements
> pip install --upgrade -r requirements.txt || true

# Downloading Pre-Trained Model
> gdown --id 1BqoJ_IJWjOwqueCZstch-XQERcUto3dt             # 600k_30epochs

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

# Evaluate
> !python evaluate.py --batch-size 180 --dataset 'Custom' --custom_test_dataset_path '<<path/to/test_dataset>>' --model-path '<<path/to/finetuned_model>>'

