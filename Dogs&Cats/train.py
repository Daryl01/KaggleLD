import os
import torch
import torch.nn.functional as F
import numpy as np
import config
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CatDogDataset
from efficientnet_pytorch import EfficientNet


def save_feature_vectors(model, loader, output_size=(1, 1), file="trainb7"):
    model.eval()
    images, labels = [], []

    for idx, (x, y), in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)

        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=output_size)
        images.append(features.reshape(x.shape[0], -1).detatch().cpu().numpy())
        labels.append(y.numpy())

    np.save(f"data_features/X_{file}.npy", np.concatenate(images, axis=0))
    np.save(f"data_features/y_{file}.npy", np.concatenate(images, axis=0))
    model.train()


def main():
    model = EfficientNet.from_pretrained("efficientnet-b7")
    train_dataset = CatDogDataset(dataset_dir='/media/daryl-loyck/part1/data/DogsCats/train/', transform=config.basic_transform)
    test_dataset = CatDogDataset(dataset_dir='/media/daryl-loyck/part1/data/DogsCats//test/', transform=config.basic_transform)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )


    model = model.torch(config.DEVICE)
    save_feature_vectors(model, train_loader, output_size=(1, 1), file="train_b7")
    save_feature_vectors(model, test_loader, output_size=(1, 1), file="test_b7")


if __name__ == "__main__":
    main()