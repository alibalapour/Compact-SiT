import argparse
import datetime
import numpy as np
import time
import torch
from pathlib import Path

from dataset.prepare_datasets import build_dataset
import vision_transformer_SiT

from timm.models import create_model

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import RocCurveDisplay, roc_curve, auc


def get_args_parser():
    parser = argparse.ArgumentParser('SiT training and evaluation script', add_help=False)
    parser.add_argument('--model', default='SiT_compact_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--training-mode', default='finetune', choices=['finetune'],
                        type=str, help='training mode')
    parser.add_argument('--batch-size', default=120, type=int)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--custom_test_dataset_path', default='', type=str,
                        help='path of custom test dataset')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--dataset', default='Custom', choices=['Custom'],
                        type=str, help='dataset name')
    parser.add_argument('--model-path', default='', type=str,
                        help='path of the fine-tuned model')

    return parser


class BC_Evaluation():
    def __init__(self):
        self.evaluation_functions = dict(
            accuracy=accuracy_,
            f1=f1,
            precision=precision,
            recall=recall,
            f1_negative=f1_negative,
            precision_negative=precision_negative,
            recall_negative=recall_negative,
            roc_auc=roc_auc,
            average_precision_score=average_precision_score
        )

    def accuracy_(y, y_hat) -> float:
        return accuracy_score(y, y_hat)

    def f1(y, y_hat, alpha: float = 0.5, beta: float = 1.):
        return f1_score(y, y_hat)

    def precision(y, y_hat) -> float:
        return precision_score(y, y_hat)

    def recall(y, y_hat) -> float:
        return recall_score(y, y_hat)

    def f1_negative(y, y_hat, alpha: float = 0.5, beta: float = 1.):
        y = 1 - y
        y_hat = 1 - y_hat
        return f1_score(y, y_hat)

    def precision_negative(y, y_hat) -> float:
        y = 1 - y
        y_hat = 1 - y_hat
        return precision_score(y, y_hat)

    def recall_negative(y, y_hat) -> float:
        y = 1 - y
        y_hat = 1 - y_hat
        return recall_score(y, y_hat)

    def roc_auc(y, y_hat) -> float:
        return roc_auc_score(y, y_hat)

    def aps(y, y_hat) -> float:
        return average_precision_score(y, y_ha)

    def evaluate(y, y_hat):
        return {name: func(y, y_hat) for name, func in self.evaluation_functions.items()}


def main(args):
    args.custom_val_dataset_path = args.custom_test_dataset_path
    dataset_test, args.nb_classes = build_dataset(is_train=False, args=args)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        training_mode=args.training_mode
    )
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model = model.load_state_dict(checkpoint['model'])

    correct_test = 0
    total_test = 0
    predicted_output = []
    predicted_probs = []
    targets = []
    with torch.no_grad():
        for data in dataloader_test:
            tensor = data[0]
            tensor = tensor.to('cuda')
            label = data[1]
            targets += label.tolist()
            label = label.to('cuda')
            output_1, output_2 = model(tensor)
            output = (output_1 + output_2) / 2
            predicted = torch.argmax(output, 1)
            predicted_prob = torch.softmax(output, 1)
            predicted_probs += predicted_prob.tolist()
            predicted_output += predicted.tolist()
            total_test += tensor.size(0)
            correct_test += (predicted == label).sum().item()
    print('Accuracy on Test Data :', 100 * (correct_test / total_test), '%')

    predicted_output = np.array(predicted_output)
    predicted_probs = np.array(predicted_probs)
    targets = np.array(targets)

    if args.nb_classes == 2:  # binary classification evaluation
        print(classification_report(targets, predicted_output))
        print("kappa score :", cohen_kappa_score(targets, predicted_output))
        bc_evaluation = BC_Evaluation()
        print(bc_evaluation.evaluate(targets, predicted_output))
    else:  # multi classification evaluation
        print(classification_report(targets, predicted_output))
        print("kappa score :", cohen_kappa_score(targets, predicted_output))
        print("auc macro :", roc_auc_score(targets, predicted_probs, average="macro", multi_class="ovr"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiT evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
