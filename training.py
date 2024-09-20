
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import matplotlib.pyplot as plt  
import torch.optim as optim


def plot_results(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def compute_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='binary')
    rec = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    auc = roc_auc_score(labels, preds)
    return acc, prec, rec, f1, auc


def show_total_time(start_time):
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Process completed in: {total_time:.2f} seconds")

    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Process completed in: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")


class FineTune:
    def __init__(self, model, train_loader, criterion, lr=0.001, momentum=0.9, num_epochs=20, device=None):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optim.SGD(self.model.model.fc.parameters(), lr=lr, momentum=momentum)
        self.num_epochs = num_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def freeze_backbone(self):
        # Freeze all layers except the final fully connected layer
        for param in self.model.model.parameters():
            param.requires_grad = False
        for param in self.model.model.fc.parameters():
            param.requires_grad = True

    def train(self):
        start_time = time.time()
        print("Fine-Tuning has started")

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            all_labels_ft = []
            all_preds_ft = []

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                running_loss += loss.item()
                all_labels_ft.extend(labels.cpu().numpy())
                all_preds_ft.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            # Compute training metrics
            acc_ft, prec_ft, rec_ft, f1_ft, auc_ft = compute_metrics(all_labels_ft, all_preds_ft)



            # Calculate average loss for the epoch
            train_loss = running_loss / len(self.train_loader)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {train_loss:.4f}')
            # printing evaluation as well  
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}, Accuracy: """""" {acc_ft}, Precision: {prec_ft}, Recall: {rec_ft}, F1-Score: {f1_ft}, ROC-AUC: {auc_ft}')
            show_total_time(start_time)



class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, num_epochs=50, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def train(self):
        self.model.to(self.device)
        start_time = time.time()
        print("Training has started")

        for epoch in range(self.num_epochs):
            # Training loop
            self.model.train()
            running_loss = 0.0
            all_labels_train = []
            all_preds_train = []

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                running_loss += loss.item()
                all_labels_train.extend(labels.cpu().numpy())
                all_preds_train.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            # Compute training metrics
            acc_train, prec_train, rec_train, f1_train, auc_train = compute_metrics(all_labels_train, all_preds_train)


            # Calculate loss for this epoch
            train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(acc_train)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss}, Train Accuracy: {acc_train}')
            print(f'Train Accuracy: {acc_train}, Precision: {prec_train}, Recall: {rec_train}, F1: {f1_train}, AUC: {auc_train}')

            # Validation loop
            self.validate(epoch)

        # Save model weights after each epoch
        torch.save(self.model.state_dict(), f'resnet50_pneumothorax_weights.pth')
        print(f"Final model weights saved ")
        show_total_time(start_time)
        plot_results(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)

        return self.model

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        all_labels_val = []
        all_preds_val = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                all_labels_val.extend(labels.cpu().numpy())
                all_preds_val.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        acc_val = accuracy_score(all_labels_val, all_preds_val)
        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(acc_val)

        print(f'Epoch [{epoch+1}/{self.num_epochs}], Val Loss: {val_loss}, Val Accuracy: {acc_val}')


class TrainerWithPlateau:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=50, early_stopping_patience=5, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def train(self):
        self.model.to(self.device)

        # Start time for training
        start_time = time.time()
        print("Training has started")

        for epoch in range(self.num_epochs):
            # Training loop
            self.model.train()
            running_loss = 0.0
            all_labels_train_plateau= []
            all_preds_train_plateau = []

            for i,(inputs, labels) in enumerate (self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                all_labels_train_plateau.extend(labels.cpu().numpy())
                all_preds_train_plateau.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            # Compute training metrics
            acc_train, prec_train, rec_train, f1_train, auc_train = compute_metrics(all_labels_train_plateau, all_preds_train_plateau)

            # Calculate loss for this epoch
            train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(acc_train)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {train_loss}')
            print(f'Train Accuracy: {acc_train}, Precision: {prec_train}, Recall: {rec_train}, F1: {f1_train}, AUC: {auc_train}')


            # Validation loop
            self.validate(epoch)

            # Check for early stopping
            if self.early_stopping():
                break

            # Scheduler step
            self.scheduler.step(self.val_losses[-1])

        # Save model weights after each epoch
        torch.save(self.model.state_dict(), f'resnet50_pneumothorax_weights_plateau.pth')
        print(f"Model weights saved ")
        show_total_time(start_time)
        plot_results(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)

        return self.model

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        all_labels_val_plateau = []
        all_preds_val_plateau = []

        with torch.no_grad():
            for i,(inputs, labels) in enumerate (self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                all_labels_val_plateau.extend(labels.cpu().numpy())
                all_preds_val_plateau.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        acc_val = accuracy_score(all_labels_val_plateau, all_preds_val_plateau)
        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(acc_val)

        print(f'Epoch [{epoch+1}/{self.num_epochs}], Val Loss: {val_loss}, Val Accuracy: {acc_val}')

        # Save model if validation loss has improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            torch.save(self.model.state_dict(), 'best_model.pth')
            print(f"Validation loss improved. Model saved at epoch {epoch+1}.")
        else:
            self.epochs_no_improve += 1
            print(f"No improvement for {self.epochs_no_improve} epochs.")

    def early_stopping(self):
        if self.epochs_no_improve >= self.early_stopping_patience:
            print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement.")
            return True
        return False



class EnhancedTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, gradient_accumulation_steps=4, num_epochs=16, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train(self):
        self.model.to(self.device)

        # Start time for training
        start_time = time.time()
        print("Training has started")

        for epoch in range(self.num_epochs):
            # Training loop
            self.model.train()
            running_loss = 0.0
            all_labels_train = []
            all_preds_train = []

            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                labels = labels.float().unsqueeze(1)  # Match target shape to model output
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps  # Normalize the loss for gradient accumulation
                loss.backward()

                running_loss += loss.item() * self.gradient_accumulation_steps  # Accumulate loss

                # Gradient update after the specified accumulation steps
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Collect labels and predictions
                all_labels_train.extend(labels.cpu().numpy())
                all_preds_train.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            # Compute training metrics
            acc_train, prec_train, rec_train, f1_train, auc_train = compute_metrics(all_labels_train, all_preds_train)

            # Calculate loss for this epoch
            train_loss = running_loss / len(self.train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(acc_train)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {train_loss}')
            print(f'Train Accuracy: {acc_train}, Precision: {prec_train}, Recall: {rec_train}, F1: {f1_train}, AUC: {auc_train}')

            # Validation loop
            self.validate(epoch)

        # Save model weights after each epoch
        torch.save(self.model.state_dict(), f'resnet50_pneumothorax_weights_enhanced.pth')
        print(f"Model weights saved")
        # Stop the timer and show total training time
        show_total_time(start_time)

        # Plot training and validation results
        plot_results(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        all_labels_val = []
        all_preds_val = []

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                labels = labels.float().unsqueeze(1)  # Match target shape to model output
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                val_loss += loss.item()

                all_labels_val.extend(labels.cpu().numpy())
                all_preds_val.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        # Compute validation metrics
        acc_val = accuracy_score(all_labels_val, all_preds_val)
        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(acc_val)

        print(f'Epoch [{epoch+1}/{self.num_epochs}], Val Loss: {val_loss}, Val Accuracy: {acc_val}')





