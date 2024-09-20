 Pneumothorax Detection in Chest X-rays
 
-This repository demonstrate using RESNET50 architecture to fine-tune and train models on the Chest X-Ray Images with Pneumothorax Masks dataset.
-The code is implemented using Pytorch library and python 3.11.5 64-bit and tested on windows 10.

# Dataset
The dataset used for this project is [Chest X-Ray Images with Pneumothorax Masks] you can find in this link: ('https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks')
the dataset structure is: 
├──_dataset/
    ├──png_images/
    ├──png_masks/
    ├──stage_1_test_images.csv
    ├──stage_1_train_images.csv


# Scripts descriptions
1.data_processing.py script is the main script to run the full project.
it includes the data processing and augmentation functionalities, plus calling all the functions and classes required for training and evaluation.

2.model_implementation.py includes the ModifiedResNet50 model and ModifiedResNet50FT that descibes the resnet architecture for training and finetuning.

3.training.py includes finetuning, training from scratch, training with plateau,enhacments classes for training the modeles, plus the required global  functions.

4. evaluation.py has the functions for implementing test-time augmentation (TTA).
5. 
### The used evaluation metrics are:
- `accuracy` 
- `precision` 
- ` recall` 
- `F1-score` 
- ` ROC-AUC` 
5.enhacments has two enhacments to optimize the model performance which are: 
- `Implement a custom loss function that combines binary cross-entropy with
 focal loss to address class imbalance.`
- `Implement gradient accumulation to simulate larger batch sizes and study
 its impact on model performance.`
##Execution
1. download the required packages by running the requirements file
pip install -r requirements.txt
2. Run data_processing.py file that calls all the required functions to run the entire project.


For loading the models weights use these instructions as example:
# Load the model weights
model = ModifiedResNet50(num_classes=2)
model.load_state_dict(torch.load('resnet50_pneumothorax_weights.pth'))
model.eval()  # Set the model to evaluation mode if you're using it for inference
