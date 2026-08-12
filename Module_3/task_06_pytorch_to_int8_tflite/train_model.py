import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from model_definition import SimpleCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    "./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

model = SimpleCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

EPOCHS = 3

for epoch in range(EPOCHS):

    model.train()

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    print(
        f"Epoch {epoch+1}, "
        f"Loss={loss.item():.4f}"
    )

torch.save(
    model.state_dict(),
    "model.pth"
)

print("Saved model.pth")