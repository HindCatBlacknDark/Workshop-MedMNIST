import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.data import get_dataloaders
from src.models import SimpleCNN
from tqdm import tqdm

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_dataloaders(args.batch_size)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        # eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.squeeze().long().to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Test Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.checkpoint)
    print(f"Best Accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint', type=str, default='runs/best.pt')
    args = parser.parse_args()
    train(args)