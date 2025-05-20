import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from PIL import Image
import time


# 展示每个类别的样本
def save_and_show_samples(dataset, save_dir="sample"):
    os.makedirs(save_dir, exist_ok=True)
    samples = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(samples[label]) < 5:
            samples[label].append(img.squeeze().numpy())
        if all(len(v) == 5 for v in samples.values()):
            break

    for i in range(10):
        class_dir = os.path.join(save_dir, str(i))
        os.makedirs(class_dir, exist_ok=True)
        for j, sample in enumerate(samples[i]):
            sample_path = os.path.join(class_dir, f"sample_{j}.png")
            Image.fromarray((sample * 255).astype(np.uint8)).save(sample_path)

    fig, axes = plt.subplots(10, 5, figsize=(8, 12))
    for i in range(10):
        for j in range(5):
            axes[i, j].imshow(samples[i][j], cmap='gray')
            axes[i, j].axis('off')
    plt.savefig('sample/all.png')
    plt.show()


# 定义模型A（基础CNN）
class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义模型B（改进CNN）
class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


# 训练函数
def train_model(model, train_loader, epochs=5, save_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_acc, train_loss = [], []
    epoch_time = []

    for epoch in range(epochs):
        time_start = time.time()
        correct, total = 0, 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            train_loss.append(loss.item())
            train_acc.append(correct / total)
            print(f'Epoch {epoch + 1}, Loss: {train_loss[-1]:.4f}, Accuracy: {train_acc[-1]:.4f}')
        time_end = time.time()
        epoch_time.append(time_end - time_start)
    torch.save(model.state_dict(), save_path)
    for epoch in range(epochs):
        print(f"epoch {epoch+1} time:{epoch_time[epoch]}")
    return train_loss, train_acc


def evaluate_model(model, test_loader, save_dir="eval"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    error_samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            for i in range(len(labels)):
                if predicted[i] != labels[i] and len(error_samples) < 5:
                    error_samples.append((images[i].cpu(), labels[i].item(), predicted[i].item()))

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()

    report = classification_report(all_labels, all_preds, output_dict=True)
    accuracy = report['accuracy']
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']
    print(f'accuracy:{accuracy}')
    print(f'macro avg:{macro_avg}')
    print(f'weighted avg:{weighted_avg}')

    error_dir = os.path.join(save_dir, "error")
    os.makedirs(error_dir, exist_ok=True)
    for i, (img, actual, predicted) in enumerate(error_samples):
        class_dir = os.path.join(error_dir, str(predicted))
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"error_{i}.png")
        Image.fromarray((img.squeeze().numpy() * 255).astype(np.uint8)).save(img_path)


if __name__ == '__main__':
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 归一化，使数据均值为0，方差为1
    ])

    # 加载数据集
    batch_size = 64
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 展示样本
    save_and_show_samples(train_dataset)
    # 计算MNIST训练集的数据分布
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=len(train_dataset), shuffle=False)
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    mean = images.mean().item()
    std = images.std().item()
    print(f"MNIST 训练集均值: {mean:.4f}, 方差: {std:.4f}")
    # 模型A训练
    model_a = ModelA()
    loss_a, acc_a = train_model(model_a, train_loader, 5, "model_a.pth")
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(loss_a, label='Model A Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.subplot(2, 1, 2)
    plt.plot(acc_a, label='Model A Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig('model_a_train.png')
    plt.show()
    # 模型B训练
    model_b = ModelB()
    loss_b, acc_b = train_model(model_b, train_loader, 5, "model_b.pth")
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(loss_b, label='Model B Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.subplot(2, 1, 2)
    plt.plot(acc_b, label='Model B Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig('model_b_train.png')
    plt.show()
    # 模型A评估
    model_a = ModelA()
    state_dict = torch.load("model_a.pth")
    model_a.load_state_dict(state_dict)
    evaluate_model(model_a, test_loader, "eval/model_a")
    # 模型B评估
    model_b = ModelB()
    state_dict = torch.load("model_b.pth")
    model_b.load_state_dict(state_dict)
    evaluate_model(model_b, test_loader, "eval/model_b")
