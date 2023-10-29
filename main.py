import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from utils.visualize import plot_train_validation_loss, display_labeled_images
from utils.preapare_data import get_datasets
from utils.train_and_validate import train_and_validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Data Preparation
train_dataset, val_dataset, train_paths, val_paths = get_datasets()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 2. Model Preparation
criterion = nn.BCEWithLogitsLoss()

# ResNet18 model
model_resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
model_resnet18.fc = torch.nn.Linear(model_resnet18.fc.in_features, 1)
optimizer_resnet18 = optim.SGD(model_resnet18.parameters(), lr=0.001, momentum=0.9)

# VGG16 model
model_vgg16 = models.vgg16(weights='VGG16_Weights.DEFAULT')
model_vgg16.classifier[6] = nn.Linear(4096, 1)
optimizer_vgg16 = optim.Adam(model_vgg16.parameters(), lr=0.0001)

# 3. Train and validate the model
num_epochs = 5

# For ResNet18
train_losses, val_losses, true_labels, predicted_labels\
    = train_and_validate('ResNet18', model_resnet18, train_loader, val_loader, criterion, optimizer_resnet18, num_epochs, device)
plot_train_validation_loss(train_losses, val_losses)
display_labeled_images('ResNet18', val_paths, true_labels, predicted_labels)

# For VGG16
train_losses, val_losses, true_labels, predicted_labels\
    = train_and_validate('VGG16', model_vgg16, train_loader, val_loader, criterion, optimizer_vgg16, num_epochs, device)
plot_train_validation_loss(train_losses, val_losses)
display_labeled_images('VGG16', val_paths, true_labels, predicted_labels)

# Save the models
torch.save(model_vgg16.state_dict(), "data/vgg16_trained.pth")
torch.save(model_resnet18.state_dict(), "data/resnet18_trained.pth")
