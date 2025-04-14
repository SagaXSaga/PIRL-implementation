import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, Grayscale
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import os
import sys
import shutil
import datetime
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


from functions import (PIRL, AverageMeter)
from pretext_tasks import (Mirror_Detection, Grayscale_Colorization, Rotate_Jigsaw)

# Utility functions
def create_save_dir():
    save_dir = os.path.join("./checkpoints", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_checkpoint(state, is_best, save_dir):
    torch.save(state, os.path.join(save_dir, 'checkpoint.pth'))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, 'checkpoint.pth'), os.path.join(save_dir, 'best_model.pth'))

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Loaded checkpoint from '{path}' (epoch {start_epoch})")
    return start_epoch

def imshow(img, mean=(0.5,0.5,0.5), std=(1.0,1.0,1.0)):
    img = img.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = img.clip(0, 1)
    return img

def visualize_predictions(model, loader, device, classes):
    model.eval()
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    if isinstance(outputs, tuple):
      outputs = outputs[0]

    _, predicted = torch.max(outputs, 1)

    # Show 4 images
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for idx, ax in enumerate(axs.flatten()):
        ax.imshow(imshow(images[idx]))
        ax.set_title(f"Predicted: {classes[predicted[idx]]}\nActual: {classes[labels[idx]]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def train_one_epoch(model, loader, criterion, optimizer, epoch, device):
    model.train()
    losses = AverageMeter()
    to_pil = ToPILImage()  # Convert tensor to PIL Image

    for images, labels in tqdm(loader, desc=f"Epoch {epoch}"):
        images, labels = images.to(device), labels.to(device)
        images_pil = [to_pil(image) for image in images]

        # Apply Grayscale transformation
        grayscale_transform = Grayscale(num_output_channels=3)  # Convert to 3-channel grayscale
        grayscale_images = [grayscale_transform(image) for image in images_pil]
        # Convert PIL back to tensor
        grayscale_images = torch.stack([transforms.ToTensor()(img) for img in grayscale_images])
        grayscale_images = grayscale_images.to(device)

        # Forward pass for both original and grayscale images
        outputs, transformed_outputs = model(images, transformed_x=grayscale_images)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Get the first item in the tuple if it's a tuple

        if isinstance(transformed_outputs, tuple):
            transformed_outputs = transformed_outputs[0]  # Similarly handle transformed_outputs

        loss = criterion(outputs, transformed_outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch} - Training Loss: {losses.avg:.4f}")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    if 'ipykernel' in sys.modules:
        args = argparse.Namespace(resume=None)  # Set resume to None by default
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
        args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    ])


    # CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PIRL().to(device)
    criterion = model.loss
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    save_dir = create_save_dir()
    best_acc = 0
    start_epoch = 1

    # Load pretrained model if specified
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer)

    for epoch in range(1, 5):
        train_one_epoch(model, trainloader, model.loss, optimizer, epoch, device)
        acc = evaluate(model, testloader, device)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, is_best, save_dir)

        if epoch % 2 == 0:
            classes = trainset.classes
            visualize_predictions(model, testloader, device, classes)

if __name__ == '__main__':
    main()
