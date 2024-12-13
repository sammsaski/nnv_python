import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

# Load MNIST
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


# Define the NN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, loader, criterion, optimizer, model_name):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(loader)}")

    # save the model
    torch.save(model.state_dict(), f"{model_name}.pth")
    print("Model saved successfully!")

    onnx_file_path = f"{model_name}.onnx"
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("Model saved in ONNX format")


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    cnn = CNN()
    fc = FC()

    # hyperparams
    learning_rate = 0.001
    epochs = 5

    criterion = nn.CrossEntropyLoss()
    cnn_optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    fc_optimizer = optim.Adam(fc.parameters(), lr=learning_rate)

    # train models
    train(cnn, train_loader, criterion, cnn_optimizer, "cnn")
    train(fc, train_loader, criterion, fc_optimizer, "fc")

    # evaluate models
    evaluate(cnn, test_loader)
    evaluate(fc, test_loader)

    # get one test sample
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    sample_image = images.squeeze(0).numpy()
    sample_label = labels.item()

    np.save("sample_image.npy", sample_image)
    np.save("sample_label.npy", sample_label)
    print(f"Saved test sample with label: {sample_label}")
