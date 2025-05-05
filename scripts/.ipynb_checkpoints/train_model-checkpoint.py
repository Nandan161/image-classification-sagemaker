import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import smdebug.pytorch as smd
from smdebug.core.modes import ModeKeys

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def evaluate(model, dataloader, criterion, device):
    logger.info("Starting model evaluation...")
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    logger.info(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def train(model, train_loader, val_loader, criterion, optimizer, device, batch_size, hook=None):
    logger.info("Beginning training loop...")
    best_loss = float("inf")
    early_stop = 0
    patience = 3
    max_epochs = 2

    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch+1}/{max_epochs}")
        for phase in ['train', 'valid']:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()
            if hook:
                hook.set_mode(ModeKeys.TRAIN if is_train else ModeKeys.EVAL)

            loader = train_loader if is_train else val_loader
            total = 0
            correct = 0
            cumulative_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)

                with torch.set_grad_enabled(is_train):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    if is_train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                cumulative_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                if batch_idx % 10 == 0:
                    logger.info(f"{phase.capitalize()} Batch {batch_idx}: Loss = {loss.item():.4f}")

            epoch_loss = cumulative_loss / len(loader)
            epoch_acc = correct / total
            logger.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            if not is_train:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    early_stop = 0
                else:
                    early_stop += 1

        if early_stop >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    return model  


def build_model(num_classes=133):
    base_model = models.resnet50(pretrained=True)

    for param in base_model.parameters():
        param.requires_grad = False
    base_model.fc = nn.Sequential(
        nn.Linear(base_model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return base_model


from torch.utils.data import random_split

def create_dataloaders(data_dir, batch_size, val_split=0.2, test_split=0.1):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform_train)

    total_size = len(full_dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Apply different transforms to val/test
    val_dataset.dataset.transform = transform_val_test
    test_dataset.dataset.transform = transform_val_test

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main(args):
    logger.info("Launching training job...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Log number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {trainable_params}")

    # Set up SageMaker Debugger hook only if config exists
    hook = None
    debug_config_path = "/opt/ml/input/config/debughookconfig.json"
    if os.path.exists(debug_config_path):
        hook = smd.Hook.create_from_json_file()
        hook.register_hook(model)
        logger.info("Debugger hook registered.")
    else:
        logger.info("Debugger config not found. Running without Debugger.")

    train_loader, val_loader, test_loader = create_dataloaders(args.data_dir, args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.module.fc.parameters() if isinstance(model, nn.DataParallel) else model.fc.parameters(),
        lr=args.learning_rate
    )

    start = time.time()
    model = train(model, train_loader, val_loader, criterion, optimizer, device, args.batch_size, hook)
    logger.info(f"Training completed in {round(time.time() - start, 2)} seconds")

    logger.info("Evaluating on test set...")
    evaluate(model, test_loader, criterion, device)

    logger.info("Saving trained model...")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))
    logger.info("Model saved.")

    if hook:
        hook.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))

    args = parser.parse_args()
    main(args)
