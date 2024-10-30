from model.ImageCaptionModel import ImageCaptionModel
from Vocabulary import Vocabulary
import torch

from Flickr8k import Flickr8k
import matplotlib.pyplot as plt 
import textwrap


def show_images(images, captions, max_caption_words = 10):
    plt.figure(figsize=(12, 12))
    for i in range(min(len(images), 16)):
        plt.subplot(4, 4, i + 1)
        image = images[i].permute(1, 2, 0)  
        # Unnormalize the image for display
        image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        plt.imshow(image)
        
        # Limit the caption length and wrap it for readability
        caption = captions[i]
        wrapped_caption = "\n".join(textwrap.wrap(caption, width=20))  # Wrap text to multiple lines
        
        plt.title(wrapped_caption, fontsize=8)  # Set font size to fit
        
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load vocabulary
vocab = Vocabulary()
vocab.load("vocabulary.pkl")

# Load model
model = ImageCaptionModel(
    vocab = vocab,
    embed_size = 512, 
    lstm_hid_dim = 1024, 
    num_lstm_layers = 1,
    device = device,
    dropout = 0.2
)

model.load("image_caption_model_with_attention.pt")



dataset = Flickr8k(
    image_directory = "./archive/Images",
    caption_path = "./archive/captions.txt",
    val_split = 0.1,
    test_split = 0.2,
    create_vocab = False
)

test_dataset = dataset(
    split = "test",
    dataloader = True,
    batch_size = 16
)

for x,y in test_dataset:
    images = x.to(device)
    
    images = model.encode(images)
    
    captions = []
    for image in images: 
        captions.append(model.decoder.generate_caption(image, max_length = 20))
        
    show_images(x, captions) 
     
