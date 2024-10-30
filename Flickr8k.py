from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
import pandas as pd 
import os 
from PIL import Image
from sklearn.model_selection import train_test_split

import spacy 
from tqdm import tqdm 

import torch 
from torch.nn.utils.rnn import pad_sequence

from Vocabulary import Vocabulary


spacy_tokenizer = spacy.load("en_core_web_sm")




class Flickr8kDataset(Dataset):
    
    def __init__(self, image_directory, caption_path, create_vocab = False):
        
        self.image_dir = image_directory
        self.captions = pd.read_csv(caption_path,
                                    delimiter=',', names=['image', 'caption'],
                                    skiprows = 1)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.vocab = Vocabulary()
        
        if create_vocab == True: 
            print("Building Vocabulary Started...")
            for caption in tqdm(range(len(self.captions['caption']))):
                caption = self.captions["caption"][caption] 
                tokens = self.__clean_string(caption)
                self.vocab.build(tokens)
        
            self.vocab.save("vocabulary.pkl")
            print("...Vocabulary Built and Saved Successfully!")
        
    
    
    def __len__(self):
        return len(self.captions)

    def __clean_string(self, raw_caption):
        # Process the captions as well
        doc = spacy_tokenizer(raw_caption)
        # Remove punctuations
        cleaned_tokens = [token.text.lower() for token in doc if not token.is_punct]
        return cleaned_tokens

    def __getitem__(self, index):
        img_name = self.captions.iloc[index, 0]
        caption = self.captions.iloc[index, 1]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Read the image, transform it 
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # Caption cleaned
        caption = self.vocab.convert_words_to_idx(self.__clean_string(caption))
        
        # Vocabulary convert 
        return image, caption
    
    
    def get_vocabulary(self):
        return self.vocab
    

class Flickr8k:
    def __init__(self, image_directory, caption_path, 
                 create_vocab = True, 
                 val_split = 0.1, test_split = 0.2):
        
        self.dataset = Flickr8kDataset(image_directory, caption_path, create_vocab = create_vocab)
        self.vocab = self.dataset.get_vocabulary()
        
        
        # Get indices for splitting
        self.train_indices, temp_indices = train_test_split(range(len(self.dataset)), test_size=(val_split + test_split), random_state=42)
        self.val_indices, self.test_indices = train_test_split(temp_indices, test_size=(test_split / (val_split + test_split)), random_state=42)
        
    
    def __call__(self, split = "train", dataloader : bool = False, batch_size : int = None):
        
        def collate_fn(batch):
            images, captions = zip(*batch)  # Unzip the batch

            # Convert captions to tensors
            caption_tensors = [torch.tensor(caption) for caption in captions]

            # Pad the captions
            padded_captions = pad_sequence(caption_tensors, batch_first=True, padding_value = self.vocab.get_pad_token_idx())

            return torch.stack(images), padded_captions
        
        if split == "train":
            dataset = Subset(self.dataset, self.train_indices)
            if dataloader == False: 
                return dataset
            else:
                if batch_size is None:
                    raise ValueError("Batch Size Not Provided")
                else: 
                    return DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn) 
           
        elif split == "valid":
            dataset = Subset(self.dataset, self.val_indices)
            if dataloader == False: 
                return dataset
            else:
                if batch_size is None:
                    raise ValueError("Batch Size Not Provided")
                else: 
                    return DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn) 
        elif split == "test":
            dataset = Subset(self.dataset, self.test_indices)
            if dataloader == False: 
                return dataset
            else:
                if batch_size is None:
                    raise ValueError("Batch Size Not Provided")
                else: 
                    return DataLoader(dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn) 
        else: 
            raise NameError("Split Not Found")
    
    
    
        
        
        
        