import torch
import torch.nn as nn
import torch.optim as optim
from utils import save_model

def train(model, train_loader, device, writer):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    model.train()

    for epoch in range(10):
        total_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}], Step [{i}], Loss: {loss.item():.4f}")
                writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)

        print(f"Epoch [{epoch+1}], Avg Loss: {total_loss/len(train_loader):.4f}")
    save_model(model)
