import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# local import
from nnv_python.set.abstract import ImageStar, imagestar_to_star


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


if __name__ == "__main__":
    # create the model
    cnn = CNN()
    fc = FC()

    # load the weights
    cnn_state_dict = torch.load("examples/MNIST/models/cnn.pth")
    fc_state_dict = torch.load("examples/MNIST/models/fc.pth")

    cnn.load_state_dict(cnn_state_dict)
    fc.load_state_dict(fc_state_dict)

    # extract weights and biases
    cnn_conv = cnn.conv.weight.detach().numpy()
    cnn_fc1 = cnn.fc1.weight.detach().numpy()
    cnn_conv_b = cnn.conv.bias.detach().numpy()
    cnn_fc1_b = cnn.fc1.bias.detach().numpy()
    cnn_weights_and_biases = [[cnn_conv, cnn_conv_b], [cnn_fc1, cnn_fc1_b]]

    fc_fc1 = fc.fc1.weight.detach().numpy()
    fc_fc2 = fc.fc2.weight.detach().numpy()
    fc_fc3 = fc.fc3.weight.detach().numpy()
    fc_fc1_b = fc.fc1.bias.detach().numpy()
    fc_fc2_b = fc.fc2.bias.detach().numpy()
    fc_fc3_b = fc.fc3.bias.detach().numpy()
    fc_weights_and_biases = [[fc_fc1, fc_fc1_b], [fc_fc2, fc_fc2_b], [fc_fc3, fc_fc3_b]]

    # load the sample
    image = np.load("examples/MNIST/data/sample_image.npy")
    label = np.load("examples/MNIST/data/sample_label.npy")

    # create upper and lower bound matrices
    epsilon = 1 / 255
    lb = np.clip(image - epsilon, 0, 1)
    ub = np.clip(image + epsilon, 0, 1)

    # create the abstract representation and apply affine transformations on it to produce output ranges
    I = ImageStar(image, lb, ub)

    # convert IS to S because fc neural network flattens input immediately
    S = imagestar_to_star(I)

    # reachability analysis (propagate through the nn)
    # for W, b in cnn_weights_and_biases:
    #     I.affine_map(W, b)

    for W, b in fc_weights_and_biases:
        S = S.affine_map(W, b)
        S = S.relu()

    lb, ub = S.get_ranges()

    # plot the ranges
    classes = np.arange(len(lb))

    # calculate midpoints for bars
    midpoints = (lb + ub) / 2
    errors = (ub - lb) / 2

    # need to flatten because shape=(10, 1)
    midpoints = midpoints.ravel()
    errors = errors.ravel()

    # plotting
    plt.figure(figsize=(10, 6))

    # bar plot with error bars
    plt.errorbar(
        classes,
        midpoints,
        yerr=errors,
        fmt="o",
        capsize=5,
        color="black",
        label="Bounds",
    )

    # add labels and title
    plt.xlabel("Classes")
    plt.ylabel("Reachable Values")
    plt.title("Reachability Analysis")
    plt.xticks(classes, [f"{i}" for i in classes])
    plt.legend()

    # save the plot
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("reachable.png", dpi=300)

    print("Reachability analysis complete!")
