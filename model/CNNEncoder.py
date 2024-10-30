import torch.nn as nn 
import torch 


class CNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        """ 
        Small subunit of CNN
        """
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1)
        
        self.relu = nn.ReLU(inplace = True)
        
        self.pool = nn.MaxPool2d(kernel_size = 2, padding = 0, stride = 2)
        
    
    def forward(self, x):
        
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x) 
        
        return x

class CNNEncoder(nn.Module):
    """
    The CNN Encoder model of our architecture 
    
    Takes in a 3x224x224 image and produces an encoding of it. 
    
    """
    def __init__(self, embed_size, dropout = 0.1):
        super().__init__()
        
        self.convolution = nn.Sequential(
            CNNBlock(3, 16),
            CNNBlock(16, 32),
            CNNBlock(32, 64),
            CNNBlock(64, 128),
            CNNBlock(128, 256)
        )
        
        self.fc = nn.Linear(256 * 7 * 7, embed_size) 
        
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):

        x = self.convolution(x)
        x = x.view(-1, 256 * 7 * 7)
        
        return self.fc(self.dropout(x)) 
    
    