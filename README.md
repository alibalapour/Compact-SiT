# Introduction
This repository contains the PyTorch self-supervised pretraining, fine-tuning, and evaluation codes for compact self-supervised vision transformer (cSiT) with the main focus on Histopathology image classification. The SiT model uses three pre-text tasks for self-supervised learning:
<ul>
  <li>Reconstruction</li>
  <li>Rotation Prediction</li>
  <li>Contrastive Learning</li>
</ul>
These three tasks are implemented on a Vision Transformer (ViT) backbone to gain advantages of the attention mechanism. In this work, we replaced the backbone of the SiT model with a more efficient, less data-hungry vision transformer known as Compact Convolutional Transformer (CCT). Then we pre-trained and fine-tuned the new model on histopathology images to get better results on this kind of medical images.

This repo is mainly adopted from [SiT Repository](https://github.com/Sara-Ahmed/SiT) with some modifications and improvements. Some features were added and the backbone of the model was replaced with [CCT-14/7x2](https://github.com/SHI-Labs/Compact-Transformers)(with some changes).

# Contributions
<ol>
  <li>Gathering a large and diverse dataset of unlabeled histopathology images (consists of ~600k images from famous histopathology datasets)</li>
  <li>RUsing Compact Convolutional Transformer, which is a compact vision transformer, as a backbone</li>
  <li>Compared to SOTA self-supervised models, getting competitive results on two famous histopathology image classification datasets, NCT-CRC and BreakHis</li>
  <li>Testing the model in a semi-supervised setting on the NCT-CRC dataset</li>
  <li>Adding pieces of code for evaluating the results of the model</li>
</ol>

# Procedure
First, we pre-trained the SiT-Compact model on a huge diverse unlabeled histopathology dataset consisting of 600k images gathered from well-known histopathology datasets (e.g., PatchCamelyon, ICIAR2018, TUPAC2016 Mitosis, and …). After pre-training phase, we used the SSL pre-trained model in several scenarios and conducted several experiments on two famous datasets (NCT-CRC and BreakHis) to display capability of Self-Supervised Learning on histopathology images.


# Dataset
To pre-train the model in a self-supervised manner, we need a collection of unlabeled images, which should be similar to our downstream task's images. As there is not any proper unlabeled dataset of histopathology images, we created our own. To find related datasets, we used dataset tables of <a href='https://arxiv.org/abs/2011.13971'>this</a> and <a href='https://arxiv.org/abs/2005.02561'>this</a> papers.

<figure >
  <img src="https://user-images.githubusercontent.com/42287060/194409925-ba4083e4-5efd-407b-84b2-c09bdd2d8571.png" class="center" alt="drawing"/>
  <figcaption>Fig.1 - sample of unlabeled dataset</figcaption>
</figure>

List of reference datasets are available at this markdown.

# SiT
As mentioned before, the pre-training consists of three pre-texts. 
- The original image corrupts with random drop, random replace, color distortion, blurring, and gray-scale in the reconstruction pre-text. Then the corrupted image is converted to patches and is fed to Transformer. After feedforward, the output of the last encoder in the Transformer is combined and makes a reconstructed image. Then with a loss function difference between the reconstructed image and the original image is measured.
- The second pre-text task is rotating the corrupted image randomly with 4 degrees (0, 90, 180, 270) and adding an extra token (like cls token) to predict rotation. A loss function is used to learn the rotation of images in the pre-training dataset.
- The third task is contrastive learning. The main focus of this method is to learn image embeddings that are invariant to different augmented views of the same image while being discriminative among different images. In each mini-batch, two images are generated from one image, and a special loss function is defined for contrastive learning. For more information, check this [paper](https://arxiv.org/abs/2002.05709).  <br>

In the main paper ([SiT](https://arxiv.org/abs/2104.03602)), all three pre-text tasks are explained completely. <br>


<figure >
  <img src="https://user-images.githubusercontent.com/42287060/195558798-7b60b14f-51d6-4ead-bd7f-8d00d2a91f79.svg" class="center" alt="drawing"/>
  <figcaption>Fig.2 - structure of compact Self-supervised Vision Transformer</figcaption>
</figure>


# Compact Convolutional Transformer (CCT)
We used [CCT-14/7×2](https://arxiv.org/abs/2104.05704), with some modifications, as the backbone of the model. By reducing embeddings in the transfomer, we had a compact and small vision transformer with around 6 million parameters. For comparison, ViT-Base has around 86 million parameters. The code of the CCT was adapted from [this repository](https://github.com/SHI-Labs/Compact-Transformers). In CCT, the final representations of encoder are given to a sequence pool modeule and prediction is produced by all of tokens. Also, a convolutional embedder is used to generate patch embeddings.   

# Results

After pre-training, the model is fine-tuned on some famous histopathology datasets like [NCT-CRC-HE-100K](https://zenodo.org/record/1214456) and [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) and tested on them. Main focus of these tests are investigating the effect of pre-training on histopathology datasets and comparing cSiT model with other self-supervised methods used on histopathology image classification.

### Results of model on the three different mode on NCT-CRC Dataset
|               |     Accuracy    |     Macro Recall    |     Macro Precision    |     Macro F1    |     Weighted F1    |     Kappa Score    |     Macro AUC    |
|:-------------:|:---------------:|:-------------------:|:----------------------:|:---------------:|:------------------:|:------------------:|:----------------:|
|       FT      |       0.84      |         0.79        |           0.82         |       0.78      |         0.84       |        0.818       |       0.980      |
|     SSL+LE    |       0.91      |         0.87        |           0.89         |       0.86      |         0.9        |        0.890       |       0.985      |
|     SSL+FT    |       0.94      |         0.92        |           0.92         |       0.91      |         0.94       |        0.928       |       0.993      |


### Results of model on the three different mode on BreakHis Dataset (5-fold stratified cross validation)
|               |     Accuracy(Mean)    |     Recall(Mean)    |     Macro F1(Mean)    |     Weighted F1(Mean)    |     Kappa Score(Mean)    |     Precision(Mean)    |
|:-------------:|:---------------------:|:-------------------:|:---------------------:|:------------------------:|:------------------------:|:----------------------:|
|       FT      |          0.857        |         0.93        |          0.82         |           0.852          |           0.642          |          0.866         |
|     SSL+LE    |          0.848        |         0.932       |          0.812        |           0.845          |           0.625          |          0.858         |
|     SSL+FT    |          0.937        |         0.952       |          0.928        |           0.938          |           0.8546         |          0.955         |


### Results of model on the three different zoom settings of BreakHis Dataset
|     Accuracy    |      40X     |      100X    |      200X    |      400X    |      Mean    |
|:---------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|         cSiT    |     93.78    |     93.12    |     94.28    |     94.39    |     93.89    |


### Results of cSiT compared to basic SSL methods ([ref](https://arxiv.org/abs/2011.13971))
|                     Models                   |     Macro F1 on      NCT-CRC    |     Macro F1 on BreakHis    |       Backbone     |
|:--------------------------------------------:|:-------------------------------:|:---------------------------:|:------------------:|
|                  Autoencoder                 |               37.0              |             36.0            |       ResNet50     |
|                  Colorization                |               80.2              |             72.4            |       ResNet50     |
|                     CPCv2                    |               80.1              |             71.1            |       ResNet50     |
|            SSL Contrastive Learning          |               86.2              |             78.2            |       ResNet50     |
|     SSL Contrastive Learning - Best Model    |               91.4              |             80.2            |       ResNet34     |
|                     cSiT                    |               93.0              |            92.8 **          |     CCT    |





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
> python evaluate.py --batch-size 180 --dataset 'Custom' --custom_test_dataset_path '<<path/to/test_dataset>>' --model-path '<<path/to/finetuned_model>>'

