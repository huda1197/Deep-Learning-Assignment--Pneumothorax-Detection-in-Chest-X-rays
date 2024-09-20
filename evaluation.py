import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torchvision.transforms as transforms
import time


    # Define a function for test-time augmentation (TTA)
def tta_prediction(model, image, device):
    # List of transformations for TTA
    tta_transforms = [
        transforms.CenterCrop(size=(224, 224)),  # Center crop
        transforms.RandomCrop(size=(224, 224), pad_if_needed=True),  # Random crop (could be corners)
        transforms.RandomCrop(size=(224, 224), pad_if_needed=True),
        transforms.RandomCrop(size=(224, 224), pad_if_needed=True),
        transforms.RandomCrop(size=(224, 224), pad_if_needed=True),
    ]

    # Apply horizontal flip to each of the crops
    horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

    # Store all predictions
    all_predictions = []

    # Perform TTA by applying the transformations
    for tta_transform in tta_transforms:
        # Original crop prediction
        crop_image = tta_transform(image)
        crop_image = crop_image.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(crop_image)
            all_predictions.append(prediction)

        # Horizontal flip prediction
        flipped_image = horizontal_flip(crop_image)
        with torch.no_grad():
            prediction_flip = model(flipped_image)
            all_predictions.append(prediction_flip)

    # Average the predictions across all transformations
    avg_prediction = torch.mean(torch.stack(all_predictions), dim=0)

    return avg_prediction
    

        # Evaluation with TTA
def evaluate_with_tta(model, test_loader_norm, device):
        """
        Evaluate the model with test-time augmentation (TTA).
        :param model: Trained model
        :param test_loader: DataLoader for test set
        :param device: Device to perform computations (CPU or GPU)
        :return: Accuracy, precision, recall, F1 score, ROC-AUC
        """
        model.to(device)
        model.eval()

        all_labels_test = []
        all_preds_test = []
        
        with torch.no_grad():
            for inputs, labels in test_loader_norm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                all_labels_test.extend(labels.cpu().numpy())
                all_preds_test.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                
                # Collect TTA predictions for each image in the batch
                for i in range(inputs.size(0)):
                    tta_pred = tta_prediction(model, inputs[i], device)
                    all_preds_test.append(torch.argmax(tta_pred).cpu().numpy())
                
                all_labels_test.extend(labels.cpu().numpy())
        
        # Calculate metrics
        acc = accuracy_score(all_labels_test, all_preds_test)
        prec = precision_score(all_labels_test, all_preds_test, average='binary')
        rec = recall_score(all_labels_test, all_preds_test, average='binary')
        f1 = f1_score(all_labels_test, all_preds_test, average='binary')
        auc = roc_auc_score(all_labels_test, all_preds_test)
        
        return acc, prec, rec, f1, auc



def evaluate_net(model, test_loader_norm, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate the model using TTA
    model.to(device) 
    print("Test has started")
    start_time = time.time()

    acc_test, prec_test, rec_test, f1_test, auc_test = evaluate_with_tta(model, test_loader_norm, device)
        # Stop the timer
    end_time = time.time()
    # Calculate the total time taken
    total_time = end_time - start_time
    print(f"Testing completed in: {total_time:.2f} seconds")

    # Optional: convert time to hours, minutes, and seconds
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Testing completed in: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    # Print the evaluation results
    print(f"Test Accuracy with TTA: {acc_test}")
    print(f"Test Precision with TTA: {prec_test}")
    print(f"Test Recall with TTA: {rec_test}")
    print(f"Test F1 Score with TTA: {f1_test}")
    print(f"Test ROC-AUC with TTA: {auc_test}")
