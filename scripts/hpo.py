import argparse
import os
import time
import logging
import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torch.nn as nn
from smdebug import pytorch as smd
from smdebug.pytorch import get_hook
from torchvision import transforms
# Avoid truncated image issues
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def create_data_loaders(data, batch_size):
    """
    This function returns the data loaders for training, validation, and testing.
    The data path is derived from S3.
    """
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader, validation_data_loader

def net():
    """
    Initializes the model with a pre-trained ResNet50 model, with the last layer replaced.
    """
    output_size = 133
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, output_size)
    )

    return model

def train(model, train_loader, validation_loader, criterion, optimizer, device, batch_size, hook=None):
    """
    Trains the model with early stopping and validation loss tracking.
    """
    logger.info("Training started!")
    epochs = 2  # Set to 2 epochs for tuning job
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0

    for epoch in range(1, epochs + 1):
        for phase in ['train', 'valid']:
            dataset_length = len(image_dataset[phase].dataset)
            if phase == 'train':
                model.train()
                grad_enabled = True
            else:
                model.eval()
                grad_enabled = False

            running_losses = []
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(grad_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_losses.append(loss.item())
                running_corrects += torch.sum(preds == labels.data)

                processed_images_count = batch_idx * batch_size + len(inputs)
                logger.info(f"{phase} epoch: {epoch} [{processed_images_count}/{dataset_length} ({100.0 * processed_images_count / dataset_length:.0f}%)]\tLoss: {loss.item():.6f}")

            epoch_loss = sum(running_losses) / len(running_losses)
            epoch_acc = running_corrects.double().item() / len(image_dataset[phase].dataset)

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1  # Early stopping if validation loss gets worse

            status = f'{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}, best validation loss: {best_loss:.4f}'
            logger.info(status)

        if loss_counter == 1:  # Early stopping
            break

    return model

def test(model, test_loader, criterion, device, batch_size):
    """
    Tests the model on the test set and reports the accuracy and loss.
    """
    logger.info("Testing started!")
    model.eval()
    running_losses = []
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_losses.append(loss.item())
            running_corrects += torch.sum(preds == labels.data)

    total_loss = sum(running_losses) / len(running_losses)
    total_acc = running_corrects.double().item() / len(test_loader.dataset)

    logger.info(f"Testing Loss: {total_loss:.4f}")
    logger.info(f"Testing Accuracy: {total_acc:.4f}")

def main(args):
    model = net()

    # Set up device and check if multiple GPUs are available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    multiple_gpus_exist = torch.cuda.device_count() > 1
    if multiple_gpus_exist:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Set up loss function and optimizer
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(
        model.module.fc.parameters() if multiple_gpus_exist else model.fc.parameters(),
        lr=args.learning_rate
    )

    # Create data loaders from the provided S3 path
    train_loader, test_loader, validation_loader = create_data_loaders(args.data, args.batch_size)

    # Train the model
    start_time = time.time()
    model = train(model, train_loader, validation_loader, loss_criterion, optimizer, device, args.batch_size)
    logger.info(f"Training time: {round(time.time() - start_time, 2)} seconds.")

    # Test the model
    start_time = time.time()
    test(model, test_loader, loss_criterion, device, args.batch_size)
    logger.info(f"Testing time: {round(time.time() - start_time, 2)} seconds.")

    # Save the model
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])  # S3 bucket location
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()

    main(args)