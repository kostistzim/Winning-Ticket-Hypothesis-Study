import torch.nn as nn


# 2 layer CNN architecture
class Conv2(nn.Module):
    def __init__(self, dropout_rate): # <--- ADD THIS ARGUMENT
        super(Conv2, self).__init__()
        #layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256), # Input size after one max-pool
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),  # <--- ADD DROPOUT LAYER
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),  # <--- ADD DROPOUT LAYER
            nn.Linear(256, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1) # Flatten the output
        out = self.classifier(out)
        return out

#4 layer CNN architecture
class Conv4(nn.Module):
    def __init__(self, dropout_rate): # <--- ADD THIS ARGUMENT
        super(Conv4, self).__init__()
        #layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256), 
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),  # <--- ADD DROPOUT LAYER
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),  # <--- ADD DROPOUT LAYER
            nn.Linear(256, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1) 
        out = self.classifier(out)
        return out

# 6 layer cnn architecture
class Conv6(nn.Module):
    def __init__(self, dropout_rate): # <--- ADD THIS ARGUMENT
        super(Conv6, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256), 
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),  
            nn.Linear(256, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1) 
        out = self.classifier(out)
        return out