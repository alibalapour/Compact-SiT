# Compact SiT (Self-supervised vIsion Transformer) for HistoPathology Images

This repository contains the official PyTorch self-supervised pretraining, finetuning, and evaluation codes for compact SiT (Self-supervised image Transformer) with main focus on HistoPathology images. The SiT model uses three pre-text tasks for self-supervised learning: Reconstruction, Rotation Prediction, and Contrastive Learning. These three tasks are implemented on a Vision Transformer (ViT) model to gain advantages of the attention mechanism. In this work, we replaced the backbone of SiT model and replaced it with a more efficient and less data-hungry vision transformer named CCT (Compact Convolutional Transformer). Then we pre-trained and fine-tuned the new model on histopathology images to get better results on this kind of image.  

This repo is mainly adopted from [SiT Repository](https://github.com/Sara-Ahmed/SiT) with some modifications and improvements. Some features were added and the backbone of the model was replaced with [CCT-14/7x2](https://github.com/SHI-Labs/Compact-Transformers)(with some changes).

# Procedure
First, we pre-trained the SiT-Compact model on a diverse unlabeled histopathology dataset consisting of 600k images derived from known datasets (e.g., PatchCamelyon, ICIAR2018, TUPAC2016 Mitosis, and …). this dataset is available in this [link](https://drive.google.com/file/d/1JoJxnY4zPuvjVGIALE_UCpCX97i6HA8J/view?usp=sharing).

<figure >
<!--   <img src="https://user-images.githubusercontent.com/42287060/142759501-a98a6d7a-fe6b-4063-b7ff-c88aedc04a7a.png" class="center" alt="drawing" width="500"/> -->
  <img src="https://user-images.githubusercontent.com/42287060/143192349-45f49221-67b5-4737-85cb-2a311dade439.png" class="center" alt="drawing" width="750"/>
  <figcaption>Fig.1 - sample of unlabeled dataset</figcaption>
</figure>

<br>

As mentioned before, the pre-training consists of three pre-texts. 
- The original image corrupts with random drop, random replace, color distortion, blurring, and gray-scale in the reconstruction pre-text. Then the corrupted image is converted to patches and is fed to Transformer. After feedforward, the output of the last encoder in the Transformer is combined and makes a reconstructed image. Then with a loss function difference between the reconstructed image and the original image is measured.
- The second pre-text task is rotating the corrupted image randomly with 4 degrees (0, 90, 180, 270) and adding an extra token (like cls token) to predict rotation. A loss function is used to learn the rotation of images in the pre-training dataset.
- The third task is contrastive learning. In each mini-batch, two images are generated from one image and a special loss function is defined for contrastive learning. For more information, check this [paper](https://arxiv.org/abs/2002.05709).  <br>
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

