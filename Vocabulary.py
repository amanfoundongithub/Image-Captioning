import pickle 


class Vocabulary:
    
    def __init__(self,
                 sos_token : str = "<SOS>",
                 eos_token : str = "<EOS>",
                 unk_token : str = "<UNK>",
                 pad_token : str = "<PAD>",):
    
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        
        # Vocabulary dictionary and reverse mapping
        self.word2idx = {}
        self.idx2word = []
        self.vocab_size = 0
        
        # Add special tokens to the vocabulary
        self.add_word(pad_token)
        self.add_word(sos_token)
        self.add_word(eos_token)
        self.add_word(unk_token)
        
    def add_word(self, word: str):
        """Add a word to the vocabulary if it's not already present."""
        if word not in self.word2idx:
            self.word2idx[word] = self.vocab_size
            self.idx2word.append(word)
            self.vocab_size += 1
    
    def build(self, list_of_tokens : list[str]):
        for token in list_of_tokens:
            self.add_word(token.lower())
            
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def convert_words_to_idx(self, words : list[str]) -> list[int]:
        indices = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words]
        return [self.word2idx[self.sos_token]] + indices + [self.word2idx[self.eos_token]] 
    
    def convert_idx_to_words(self, idxs : list[int]) -> list[str]:
        return [self.idx2word[idx] if 0 <= idx < self.vocab_size else self.unk_token for idx in idxs]
    
    def get_pad_token_idx(self):
        return self.word2idx[self.pad_token]

    def save(self, filename):
        
        data = {
            "word2idx" : self.word2idx,
            "idx2word" : self.idx2word,
            "sos" : self.sos_token,
            "eos" : self.eos_token,
            "pad" : self.pad_token,
            "unk" : self.unk_token,
        }
        

        f = open(filename, "wb")
        pickle.dump(data, f) 
        f.close()
    
    def load(self, filename):
        f = open(filename, "rb")
        
        data = pickle.load(f)
        
        self.word2idx = data.get("word2idx")
        self.idx2word = data.get("idx2word")
        self.sos_token = data.get("sos")
        self.eos_token = data.get("eos")
        self.pad_token = data.get("pad")
        self.unk_token = data.get("unk")
        
        self.vocab_size = len(self.idx2word) 
        
        f.close() 
        
        
        