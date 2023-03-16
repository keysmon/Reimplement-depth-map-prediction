import torch.nn as nn

class CoarseNetwork(nn.Module):
    def __init__(self):
        super(CoarseNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(in_features=4096, out_features=1)
        )
        
    def forward(self, x):
        x = self.layers(x)
    
image_class = CoarseNetwork()

print(image_class)

