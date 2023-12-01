# script to train a resnet18 model on CUB dataset all from huggingface
import os
import time
import random
import logging
import socket
import argparse  

from tqdm import tqdm

import torchvision
from sklearn.model_selection import train_test_split
from torchvision.models import (resnet18, alexnet, vgg16, wide_resnet50_2, resnet50, resnext50_32x4d,
                                mobilenet_v3_small,
                                googlenet, inception_v3, shufflenet_v2_x0_5, densenet121)
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


from transformers import AdamW, AutoImageProcessor
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import logging
import wandb
from transformers import ViTConfig, ViTForImageClassification
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


def get_model(model_name, num_classes, pretrained=False):
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'alexnet':
        model = alexnet(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'vgg16':
        model = vgg16(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'resnext50_32x4d':
        model = resnext50_32x4d(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'mobilenet_v3_small':
        model = mobilenet_v3_small(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'googlenet':
        model = googlenet(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'inception_v3':
        model = inception_v3(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'shufflenet_v2_x0_5':
        model = shufflenet_v2_x0_5(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'densenet121':
        model = densenet121(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'vit':
        configuration = ViTConfig()
        model = ViTForImageClassification(configuration)
        model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    elif model_name == 'vit-pretrained':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    else:
        raise NotImplementedError(f'Model {model_name} not implemented')
    return model


def parse_args():
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default="in9l")
    parser.add_argument('--dataset_dir', type=str, default="/fs/cbcb-scratch/aliganj/dataset")
    parser.add_argument('--dataset_ratio', type=float, default=1.0)
    parser.add_argument('--exp_name', type=str, default='in9l-vit-scratch')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--saving_epoch', type=int, default=5)
    # Model
    parser.add_argument('--model', type=str, default='vit')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--pretrained', action='store_true')
    # Hyperparams
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--lr_scheduler_step_size', type=int, default=15)
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    args = parser.parse_args()
    return args


def train(model, train_loader, criterion, optimizer, scheduler):
    model.train()
    losses = []
    for batch in tqdm(train_loader):
        images = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        pred = model(images).logits
        loss = criterion(pred, labels)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    return np.mean(losses)


def evaluate(model, test_loader, criterion):
    model.eval()
    losses = []
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch[0].to(args.device)
            labels.extend(batch[1].tolist())
            pred = model(images).logits
            preds.extend(torch.argmax(pred, dim=1).tolist())
            loss = criterion(pred, batch[1].to(args.device))
            losses.append(loss.item())
    return np.mean(losses), accuracy_score(labels, preds), precision_recall_fscore_support(labels, preds,
                                                                                           average='macro')


def load_and_evaluate(model, save_path, test_loader, original_test_loader, criterion):
    # load the best model
    model.load_state_dict(torch.load(save_path))
    test_loss, acc, (precision, recall, f1, _) = evaluate(model, test_loader, criterion)

    original_test_loss, original_acc, (original_precision, original_recall, original_f1, _) = evaluate(model,
                                                                                                       original_test_loader,
                                                                                                       criterion)
    wandb.log({"test_acc": acc, "original_test_acc": original_acc})

    logging.info(
        f'Test Loss: {test_loss:.4f} | Accuracy: {acc:.4f} | Precision: {precision:.4f} |'
        f' Recall: {recall:.4f} | F1: {f1:.4f}')
    logging.info(
        f'Original Test Loss: {original_test_loss:.4f} | Accuracy: {original_acc:.4f} |'
        f' Precision: {original_precision:.4f} | Recall: {original_recall:.4f} | F1: {original_f1:.4f}')


def main(args):
    wandb.login(key=os.environ.get("WANDB_LOGIN"))
    wandb.init(project="bg-analysis", config=args, name=args.exp_name + "_" + args.dataset_name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    original_test_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_dir, 'bg_challenge', 'original', 'val'))
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_dir, args.dataset_name, 'train'))
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_dir, args.dataset_name, 'val'))
    if args.dataset_name == "in9l":
        test_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_dir, 'bg_challenge', 'original', 'val'))
    else:
        test_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_dir, 'bg_challenge', args.dataset_name, 'val'))

    if args.dataset_ratio < 1.0:
        train_dataset = torch.utils.data.Subset(train_dataset, np.random.choice(len(train_dataset),
                                                                                          int(args.dataset_ratio * len(
                                                                                              train_dataset)),
                                                                                          replace=False))

    logging.warning(f'Train Dataset Size: {len(train_dataset)}')
    logging.warning(f'Val Dataset Size: {len(val_dataset)}')
    logging.warning(f'Original Test Dataset Size: {len(original_test_dataset)}')
    logging.warning(f'{args.dataset_name} Test Dataset Size: {len(test_dataset)}')

    if 'vit' in args.model:
        image_processor = AutoImageProcessor.from_pretrained(
            'google/vit-base-patch16-224-in21k',
        )

        # Define torchvision transforms to be applied to each image.
        if "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        else:
            size = (image_processor.size["height"], image_processor.size["width"])
        normalize = (
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
            if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
            else Lambda(lambda x: x)
        )
        tr_transform = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        test_transform = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    else:
        tr_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
        )

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
        )

    if args.dataset_ratio < 1.0:
        train_dataset.dataset.transform = tr_transform
    else:
        train_dataset.transform = tr_transform
    val_dataset.transform = test_transform
    test_dataset.transform = test_transform
    original_test_dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    original_test_loader = DataLoader(original_test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model, args.num_classes, args.pretrained)
    model.to(args.device)


    logging.warning(f'Model: {model}')
    logging.warning(f'Number of Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma)
    criterion = nn.CrossEntropyLoss()

    logging.warning("Starting Training...")
    logging.warning(f'Batch Size: {args.batch_size}')
    logging.warning(f'Learning Rate: {args.lr}')
    logging.warning(f'Weight Decay: {args.weight_decay}')
    logging.warning(f'Epochs: {args.epochs}')

    # args.save_path = os.path.join(args.save_dir, 
                                #   args.exp_name + "_" + args.dataset_name + args.model + '.pth')
    args.save_path = os.path.join(args.save_dir, args.exp_name + "_" + args.dataset_name)

    # os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    # if checkpoint exists, load it and no need to train
    # if os.path.exists(args.save_path):
    #     logging.info(f'Loading Model from {args.save_path}')
    #     load_and_evaluate(model, args.save_path, test_loader, original_test_loader, criterion)
    #     return

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        logging.warning(f'Epoch: {epoch}')
        train_loss = train(model, train_loader, criterion, optimizer, scheduler)
        wandb.log({"train_loss": train_loss})
        val_loss, val_acc, (val_precision, val_recall, val_f1, _) = evaluate(model, val_loader, criterion)
        wandb.log({"val_loss": val_loss, "val_acc": val_acc, "val_precision": val_precision, "val_recall": val_recall,
                   "val_f1": val_f1})

        logging.warning(
            f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} |'
            f' Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}')
        if (val_acc > best_acc) or (epoch % args.saving_epoch == 0):
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_path, f"epoch_{epoch}.pth"))
            logging.info(f'Saved Model at Epoch: {epoch}')
        

    load_and_evaluate(model, args.save_path, test_loader, original_test_loader, criterion)


def logging_args(args):
    args_items = vars(args)
    args_keys = list(args_items.keys())
    args_keys.sort()
    for arg in args_keys:
        logging.warning(str(arg) + ': ' + str(getattr(args, arg)))


if __name__ == '__main__':
    args = parse_args()

    args.save_dir = '/fs/cbcb-scratch/aliganj/results'

    args.save_dir = os.path.join(args.save_dir,
                                 os.path.basename(os.getcwd()),
                                 os.path.basename(__file__).split('.')[0], 
                                 time.strftime("%d-%m-%Y_") + time.strftime("%H-%M-%S_") + socket.gethostname())
    
    args.checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.verbose > 0:
        logging_args(args)
    main(args)
