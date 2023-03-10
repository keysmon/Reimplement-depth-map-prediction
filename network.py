import torch.nn as nn

class CoarseNetwork(nn.Module):
    def __init__(self):
        super(CoarseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3,96, 11, stride=4)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(3,256,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(3,384,3)
        self.conv4 = nn.Conv2d(3,384,3)
        self.conv5 = nn.Conv2d(3,384,3)
        self.fc1 = nn.Linear(256,4096)
        self.fc2 = nn.Linear(4096,4096)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
image_class = CoarseNetwork()

print(image_class.fc1)

