import torch.nn as nn 
import torch 

from .CNNEncoder import CNNEncoder
from .LSTMDecoder import LSTMDecoder

class ImageCaptionModel(nn.Module):
    
    def __init__(self, 
                 vocab, 
                 embed_size, 
                 lstm_hid_dim = 256,
                 num_lstm_layers = 1,
                 dropout = 0.1,
                 device = "cpu"):
        
        super().__init__()
        
        self.encoder = CNNEncoder(
            embed_size = embed_size
        ).to(device) 
        
        self.decoder = LSTMDecoder(
            vocab = vocab,
            embed_size = embed_size,
            hid_dim = lstm_hid_dim,
            num_layers = num_lstm_layers,
            device = device
        ).to(device)
        
        print(f"Number of parameters : {sum(p.numel() for p in self.parameters())}")
    
    def encode(self, image):
        return self.encoder(image)

    def decode(self, caption, cnn_features):
        return self.decoder(caption, cnn_features)

    def forward(self, image, caption):
        cnn_features = self.encoder(image) 
        return self.decoder(caption, cnn_features)

    def save(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        self.load_state_dict(torch.load(filename))
    
    def generate(self, image, max_length = 40):
        cnn_features = self.encoder(image)
        
        return self.decoder.generate_caption(cnn_features, max_length = max_length)
    