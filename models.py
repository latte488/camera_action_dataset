import torch
from torch import nn
from convlstm import ConvLSTM

class LstmNet(nn.Module):
    def __init__(self, num_classes):
        super(LstmNet, self).__init__()
        self.features = nn.LSTM(
            input_size=3 * 128 * 128,
            hidden_size=1024,
            num_layers=3,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 2)
        x, _  = self.features(x)
        x = self.classifier(x[:, -1, :])
        return x
        

class ConvLstmNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvLstmNet, self).__init__()
        self.features = ConvLSTM(
            input_dim=3,
            hidden_dim=[16],
            kernel_size=(3, 3), 
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3)) 
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )   

    
    def forward(self, x): 
        _, x = self.features(x)
        x = self.avgpool(x[0][0])
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def Net(num_classes):
    return LstmNet(num_classes=num_classes)
