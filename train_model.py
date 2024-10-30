from Flickr8k import Flickr8k
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
import torch.optim as optim 

from tqdm import tqdm 


from model.ImageCaptionModel import ImageCaptionModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def show_images(images, captions):
    plt.figure(figsize=(12, 12))
    for i in range(min(len(images), 16)):
        plt.subplot(4, 4, i + 1)
        image = images[i].permute(1, 2, 0)  
        # Unnormalize the image for display
        image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        plt.imshow(image)
        plt.title(captions[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

dataset = Flickr8k(
    image_directory = "./archive/Images",
    caption_path = "./archive/captions.txt",
    val_split = 0.1,
    test_split = 0.2
)

train_dataset = dataset(
    split = "train",
    dataloader = True,
    batch_size = 64
)

valid_dataset = dataset(
    split = "valid",
    dataloader = True,
    batch_size = 64
)

vocab_size = dataset.vocab.get_vocab_size()

model = ImageCaptionModel(
    vocab = dataset.vocab,
    embed_size = 512, 
    lstm_hid_dim = 1024, 
    num_lstm_layers = 1,
    device = device,
    dropout = 0.2
)

criterion = nn.CrossEntropyLoss(ignore_index = dataset.vocab.get_pad_token_idx())

optimizer = optim.Adam(model.parameters(), lr = 0.005) 

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 8, gamma = 0.1) 

EPOCHS = 10

print("Starting training...") 
for ep in range(EPOCHS):
    
    print("------------------------------------------------------")
    print(f"Epoch #{ep + 1}/{EPOCHS}\n")
    total_train_loss = 0.0 
    
    train_pg_bar = tqdm(train_dataset, desc = f"Epoch {ep + 1} (Training)", leave = False)
    # Training part 
    for image, caption in train_pg_bar:
        model.train()
        image = image.to(device)
        caption = caption.to(device)
    
        caption_train = caption[:, :-1].to(device)
        caption_target = caption[:, 1:].to(device)
    
        cnn_features = model.encode(image)
    
        outputs = model.decode(caption_train, cnn_features)
    
        loss = criterion(outputs.view(-1, vocab_size), caption_target.contiguous().view(-1))
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    
        total_train_loss += loss.item() 
        
        train_pg_bar.set_postfix(loss = loss.item()) 
        # Clear the cache (for GPU)
        if device == "cuda":
            torch.cuda.empty_cache()
            
            
        
    avg_train_loss = total_train_loss/len(train_dataset)
    print(f"Training Summary:\n\tTraining Loss : {avg_train_loss :.4f}\n\tLearning Rate Used : {scheduler.get_last_lr()[0]}\n")
    
    # Validation part
    with torch.no_grad():  # Disable gradient computation for validation
        total_val_loss = 0.0
        for image, caption in tqdm(valid_dataset, desc = f"Epoch {ep + 1} (Validation)", leave = False):
            model.eval()  # Set model to evaluation mode
            image = image.to(device)
            caption = caption.to(device)

            caption_val = caption[:, :-1].to(device)
            caption_target = caption[:, 1:].to(device)

            cnn_features = model.encode(image)
            outputs = model.decode(caption_val, cnn_features)

            loss = criterion(outputs.view(-1, vocab_size), caption_target.contiguous().view(-1))
            total_val_loss += loss.item()
            if device == "cuda":
                torch.cuda.empty_cache()

        avg_val_loss = total_val_loss / len(valid_dataset)
        print(f"Validation Summary:\n\tValidation Loss : {avg_val_loss :.4f}\n")
    
    scheduler.step(avg_val_loss)
    # scheduler.step() 
    print("------------------------------------------------------")
print("...Training ended!")

print("\nSaving the model now...")
model.save("image_caption_model_with_attention.pt") 
print("...Model Saved!") 