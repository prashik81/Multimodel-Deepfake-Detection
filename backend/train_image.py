import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transforms (IMPORTANT)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
train_data = datasets.ImageFolder("../dataset/train", transform=transform)
val_data = datasets.ImageFolder("../dataset/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

print("Classes:", train_data.classes)

# Model (MobileNet - fast)
model = models.mobilenet_v2(weights="DEFAULT")

# Replace classifier (ONLY ONCE)
model.classifier[1] = nn.Linear(model.last_channel, 2)

model = model.to(device)

# Loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Progress log
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Completed, Avg Loss: {total_loss/len(train_loader):.4f}")

    # 🔍 Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%\n")

# Save model
torch.save(model.state_dict(), "image_model.pth")

print("✅ Training Complete! Model Saved as image_model.pth")