import argparse
import datetime
import numpy as np
import time
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from timm.models import create_model

from dataset.prepare_datasets import build_dataset
import vision_transformer_SiT


def get_args_parser():
    parser = argparse.ArgumentParser('SiT training and evaluation script', add_help=False)
    parser.add_argument('--model', default='SiT_compact_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--training-mode', default='finetune', choices=['finetune'],
                        type=str, help='training mode')
    parser.add_argument('--batch-size', default=120, type=int)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--custom-inference-dataset-path', default='', type=str,
                        help='path of custom test dataset')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--dataset', default='Custom', choices=['Custom', 'BreakHis'],
                        type=str, help='dataset name')
    parser.add_argument('--model-path', default='', type=str,
                        help='path of the fine-tuned model')
    parser.add_argument('--dataset-location', default='downloaded_datasets', type=str,
                        help='dataset location - dataset will be downloaded to this folder')
    parser.add_argument('--nb-class', default=2, type=int,
                        help='number of classes in fine-tuned model')

    # related to linear evaluation
    parser.add_argument('--SiT_LinearEvaluation', default=0, type=int,
                        help='If true, the backbone of the system will be freezed')
    parser.add_argument('--representation-size', default=None, type=int, help='nonLinear head')

    parser.add_argument('--feature-extractor', action='store_true', default=False,
                        help='model acts like an feature extractor')

    return parser


def main(args):
    args.custom_val_dataset_path = args.custom_inference_dataset_path
    args.dataset_return_name = True
    dataset_test, _ = build_dataset(is_train=False, args=args)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    model = create_model(
        args.model,
        pretrained=False,
        training_mode=args.training_mode,
        num_classes=args.nb_class,
        representation_size=args.representation_size,
        feature_extractor=args.feature_extractor
    )
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)

    predicted_output = []
    predicted_probs = []
    names = []
    with torch.no_grad():
        for data in tqdm(dataloader_test):
            tensor = data[0][0]
            tensor = tensor.to(args.device)
            output_1, output_2 = model(tensor)
            output = (output_1 + output_2) / 2
            predicted = torch.argmax(output, 1)
            predicted_prob = torch.softmax(output, 1)
            predicted_probs += predicted_prob.tolist()
            predicted_output += predicted.tolist()
            name = data[1]
            names += name

    names = np.array(names)
    predicted_output = np.array(predicted_output)
    predicted_probs = np.array(predicted_probs)

    dataframe = pd.DataFrame({'name': names, 'label': predicted_output, 'prob': np.max(predicted_probs, axis=-1)})
    dataframe.to_csv('results.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiT evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
