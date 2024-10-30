import torch.nn as nn 
import torch 

import torch.nn.functional as F 

    
class LSTMDecoder(nn.Module):
    """
    Decoder to generate caption 
    """
    def __init__(self, vocab, embed_size, hid_dim, num_layers = 1, dropout = 0.1, device = "cpu"):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings = vocab.get_vocab_size(), 
            embedding_dim = embed_size
        ).to(device) 
        
        self.hid_dim = hid_dim
        
        self.lstm = nn.LSTMCell(
            input_size = embed_size,
            hidden_size = hid_dim
        ).to(device) 
        
        self.linear = nn.Linear(
            hid_dim,
            vocab.get_vocab_size() 
        ).to(device) 
        
        self.linear_dropout = nn.Dropout(dropout)
        
        self.device = device
        self.vocab = vocab 
        
        

    def forward(self, caption, cnn_features):
        
        batch_size = cnn_features.size(0)
        
        hidden_state = torch.zeros((batch_size, self.hid_dim)).to(self.device)
        
        cell_state   = torch.zeros_like(hidden_state).to(self.device) 
        
        outputs = torch.empty((batch_size, caption.size(1), self.vocab.get_vocab_size())).to(self.device)
  
        embeddings = self.embedding(caption)
        
        for t in range(caption.size(1)):
            
            if t == 0:
                # CNN
                hidden_state, cell_state = self.lstm(cnn_features, (hidden_state, cell_state))
            
            else: 
                # Normal
                hidden_state, cell_state = self.lstm(embeddings[: ,t, :], (hidden_state, cell_state))
        
            output = self.linear_dropout(hidden_state)
            output = self.linear(output) 
        
            outputs[:, t, :] = output 

        return outputs

    
    def generate_caption(self, cnn_features, max_length = 40):
        
        # Initialize cell state and hidden state
        hidden_state = torch.zeros((1, self.hid_dim)).to(self.device)
        cell_state   = torch.zeros_like(hidden_state).to(self.device) 
        
        caption = []
        
        # Start with the SOS token
        input_token = torch.tensor([self.vocab.word2idx[self.vocab.sos_token]]).unsqueeze(0).to(self.device)  # Start with <sos> token

  
        for _ in range(max_length):
            embedding = self.embedding(input_token)
            
    
            if len(caption) == 0:
                hidden_state, cell_state = self.lstm(cnn_features.unsqueeze(0), (hidden_state, cell_state))
            else:
                hidden_state, cell_state = self.lstm(embedding.squeeze(0), (hidden_state, cell_state))
        
            output = self.linear(self.linear_dropout(hidden_state))
        
            predicted_token = output.argmax(dim=1)
            predicted_word = self.vocab.idx2word[predicted_token.item()]
            
            # Stop if end of sentence token is reached
            if predicted_word == self.vocab.eos_token:
                break
            
            # Append the predicted word to the caption
            caption.append(predicted_word)
        
            # Prepare the next input
            input_token = predicted_token.unsqueeze(0)
        
        return " ".join(caption)
                
        
