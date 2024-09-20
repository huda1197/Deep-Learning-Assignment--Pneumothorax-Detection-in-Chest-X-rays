import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import defaultdict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from PIL import ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm  # To show progress bar while calculating
from model_implementation import ModifiedResNet50FT,ModifiedResNet50
from training import Trainer,TrainerWithPlateau,EnhancedTrainer,FineTune
from evaluation import evaluate_net
from enhacments import  CombinedLoss


csv_file1 = "archive/siim-acr-pneumothorax/stage_1_test_images.csv"
csv_file2 = "archive/siim-acr-pneumothorax/stage_1_train_images.csv"
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
df = pd.concat([df1, df2])
image_info = {row['new_filename']: (row['ImageId'], row['has_pneumo']) for _, row in df.iterrows()}

def get_patient_id(filename):
    return filename.split('_')[0]

def show_images(images, labels, num_images=10):
    plt.figure(figsize=(20, 10))
    num_images = min(num_images, len(images))  # Ensure we don't try to display more images than we have in batch size
    for i in range(num_images):
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.axis('off')
        plt.title(f'Pneumothorax: {labels[i]}', fontsize=12, pad=10)
        
    plt.show()

class PneumothoraxDataset(Dataset):
    def __init__(self,
                image_dir, filenames,
                image_info, transform=None):
        self.image_dir = image_dir
        self.filenames = filenames
        self.image_info = image_info
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        image = ImageOps.grayscale(image)  # Ensure image is grayscale
        image_id, label = self.image_info[img_name]
        if self.transform:
            image = self.transform(image)


        return image, label

# Step 1: Group images by patient
if __name__ == '__main__':
    image_dir = "./archive/siim-acr-pneumothorax/png_images"
    mask_dir = "./archive/siim-acr-pneumothorax/png_masks"
    image_filenames = os.listdir(image_dir)
    patient_dict = defaultdict(list)
    for filename in image_filenames:
        patient_id = get_patient_id(filename)
        patient_dict[patient_id].append(filename)

    # Step 2: Split the patients
    patient_ids = list(patient_dict.keys())
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    # Step 3: Create datasets
    train_filenames = [filename for pid in train_ids for filename in patient_dict[pid]]
    val_filenames = [filename for pid in val_ids for filename in patient_dict[pid]]
    test_filenames = [filename for pid in test_ids for filename in patient_dict[pid]]
    print(f"Training set size: {len(train_filenames)}")
    print(f"Validation set size: {len(val_filenames)}")
    print(f"Test set size: {len(test_filenames)}")


    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure the image has 1 channel (grayscale)
        transforms.Resize((512, 512)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizing for a single channel
    ])

    train_dataset = PneumothoraxDataset(image_dir, train_filenames, image_info, transform=transform)
    val_dataset = PneumothoraxDataset(image_dir, val_filenames, image_info, transform=transform)
    test_dataset = PneumothoraxDataset(image_dir, test_filenames, image_info, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=4)
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    show_images(images, labels, num_images=10)
    mean = 0.0
    std = 0.0
    nb_samples = 0

    for data in tqdm (train_loader):
        images, _ = data
        batch_samples = images.size(0)  # Number of images in the batch
        images = images.view(batch_samples, images.size(1), -1)  # Reshape to (batch_size, channels, pixels)
        mean += images.mean(2).sum(0)  # Sum up mean of each batch
        std += images.std(2).sum(0)  # Sum up std of each batch
        nb_samples += batch_samples  # Keep count of total number of samples

    mean /= nb_samples
    std /= nb_samples
    print(f'Mean: {mean}')
    print(f'Std: {std}')

    transform_norm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale (single-channel)
        transforms.Resize((512, 512)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean.item()], std=[std.item()])  # Use the calculated values
    ])

    #Preparing datasets after normalization
    train_dataset_norm = PneumothoraxDataset(image_dir,
                                            train_filenames, image_info, transform=transform_norm)
    val_dataset_norm = PneumothoraxDataset(image_dir,
                                            val_filenames, image_info, transform=transform_norm)
    test_dataset_norm = PneumothoraxDataset(image_dir,
                                            test_filenames, image_info, transform=transform_norm)
    train_loader_norm = DataLoader(train_dataset_norm,
                                    batch_size=6, shuffle=True, num_workers=4)
    val_loader_norm= DataLoader(val_dataset_norm,
                                batch_size=6, shuffle=False, num_workers=4)
    test_loader_norm = DataLoader(test_dataset,
                                batch_size=6, shuffle=False, num_workers=4)
    

    # Fine-Tuning
    model = ModifiedResNet50FT()  # Instantiate your model
    criterion = nn.CrossEntropyLoss()  #set up the loss function
    finetune = FineTune(model=model,
                        train_loader=train_loader_norm,
                        criterion=criterion,num_epochs=20)     # Create FineTune instance
    finetune.freeze_backbone()  
    finetune.train()


    #Training 
    model = ModifiedResNet50(num_classes=2) # Instantiate the model 
    criterion = nn.CrossEntropyLoss()  #Set up the loss function
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)  #set up the adam optimizer
    train_loader_norm = torch.utils.data.DataLoader(train_loader_norm.dataset,
                                                    batch_size=32, shuffle=True)  # Adjust the DataLoader to use a batch size of 32
    trainer = Trainer(model,
                        train_loader_norm, val_loader_norm,
                        optimizer, criterion, num_epochs=1)
    trained_model = trainer.train()


#Implement learning rate scheduling and early stopping.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min', factor=0.1, patience=5, verbose=True)  #Reduce learning rate when validation loss plateaus
    # Early Stopping Parameters
    early_stopping_patience = 5  # Stop training if validation loss doesn't improve for 10 epochs
    best_val_loss = float('inf')
    early_stopping_counter = 0
    trainer_plateau = TrainerWithPlateau(model,
                                        train_loader_norm, val_loader_norm,
                                        optimizer, criterion, scheduler,
                                        num_epochs=50, early_stopping_patience=5)
    trained_model = trainer_plateau.train()

    #TTA 
    model=ModifiedResNet50(num_classes=2)
    evaluate_net(model,test_loader_norm)

    #Enhacments
    model = ModifiedResNet50(num_classes=1)  # Instantiate the model
    criterion = CombinedLoss(alpha=1, gamma=2)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    # Set the number of gradient accumulation steps
    gradient_accumulation_steps = 4 
    # Assuming you have your model, train_loader, val_loader, criterion, and optimizer ready
    trainer_enhanced = EnhancedTrainer(model, train_loader, val_loader, criterion, optimizer, gradient_accumulation_steps=4, num_epochs=16)
    trained_model = trainer_enhanced.train()




