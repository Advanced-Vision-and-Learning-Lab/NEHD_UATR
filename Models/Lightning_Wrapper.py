"""
Created on Thursday April 25 22:32:00 2024
Wrap models in a PyTorch Lightning Module for training and evaluation
@author: salimalkharsa
"""

import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as L
import torchmetrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
import os


class Lightning_Wrapper(L.LightningModule):
    def __init__(self, audio_feature_extractor, intermediate_model, base_model, num_classes,
                 intermediate_learning_rate=1e-3, base_model_learning_rate=1e-5, step_size=1000, gamma=.1, log_dir=None,
                 label_names=None, average='weighted'):
        super().__init__()
        self.save_hyperparameters(
            ignore=['criterion', 'intermediate_model', 'base_model'])
        self.audio_feature_extractor = audio_feature_extractor
        self.intermediate_model = intermediate_model
        self.base_model = base_model
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.log_dir = log_dir
        self.intermediate_learning_rate = intermediate_learning_rate
        self.base_model_learning_rate = base_model_learning_rate
        if not self.intermediate_model is None:
            print("Intermediate Model Learning Rate: ",
                  self.intermediate_learning_rate)
        print("Base Model Learning Rate: ", self.base_model_learning_rate)
        self.optimizer = optim.Adam(
            [
                {'params': self.intermediate_model.parameters(
                ), 'lr': intermediate_learning_rate},
                {'params': self.base_model.parameters(), 'lr': base_model_learning_rate}
            ] if intermediate_model else [{'params': self.base_model.parameters(), 'lr': base_model_learning_rate}])

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=step_size,
                                                   gamma=gamma)

        self.average = average

        # If names not provided, generate names (Class 1, ... , Class C)
        if label_names is None:
            self.label_names = []
            for class_name in range(0, self.num_classes):
                self.label_names.append('Class {}'.format(class_name))
        else:
            self.label_names = label_names

        # Change tasks based on number of classes (only consider binary and multiclass)
        if self.num_classes == 2:
            task = "binary"
        else:
            task = "multiclass"

        self.train_accuracy = torchmetrics.classification.Accuracy(
            task=task, num_classes=self.num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(
            task=task, num_classes=self.num_classes)
        # self.val_f1 = torchmetrics.F1Score(task=task, num_classes=self.num_classes, average=average)
        # self.val_precision = torchmetrics.Precision(task=task, num_classes=self.num_classes, average=average)
        # self.val_recall = torchmetrics.Recall(task=task, num_classes=self.num_classes, average=average)

        # self.test_accuracy = torchmetrics.classification.Accuracy(task=task, num_classes=self.num_classes)
        # self.test_f1 = torchmetrics.F1Score(task=task, num_classes=self.num_classes, average=average)
        # self.test_precision = torchmetrics.Precision(task=task, num_classes=self.num_classes, average=average)
        # self.test_recall = torchmetrics.Recall(task=task, num_classes=self.num_classes, average=average)

        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []

        self.train_accuracies = []
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        self.best_epoch = 0
        self.best_val_accuracy = 0.0

    def forward(self, x):
        x = self.audio_feature_extractor(x)
        if self.intermediate_model is not None:
            x = self.intermediate_model(x)
        outputs = self.base_model(x)
        return outputs

    # Work to be done
    def configure_optimizers(self):

        return [self.optimizer], [self.scheduler]

    # def on_epoch_start(self):
    #     # Freeze model1 after 50 epochs
    #     if self.intermediate_model is not None and self.current_epoch == 40:
    #         print("Intermediate Model Freeze")
    #         for param in self.intermediate_model.parameters():
    #             param.requires_grad = False 

    def training_step(self, batch, batch_idx):
        inputs, labels, index = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, labels.long())

        accuracy = getattr(self, 'train_accuracy')(outputs, labels)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, index = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, labels.long())

        accuracy = getattr(self, 'val_accuracy')(outputs, labels)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        if not self.training:
            self.val_preds.extend(outputs.argmax(dim=1).tolist())
            self.val_labels.extend(labels.tolist())
            self.log_metrics(outputs, labels, prefix='val')

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels, index = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, labels.long())

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.test_preds.extend(outputs.argmax(dim=1).tolist())
        self.test_labels.extend(labels.tolist())

        self.log_metrics(outputs, labels, prefix='test')

        return loss

    def log_metrics(self, outputs, labels, prefix):
        # accuracy = getattr(self, f'{prefix}_accuracy')(outputs, labels)
        # f1 = getattr(self, f'{prefix}_f1')(outputs, labels)
        # precision = getattr(self, f'{prefix}_precision')(outputs, labels)
        # recall = getattr(self, f'{prefix}_recall')(outputs, labels)

        # self.log(f'{prefix}_accuracy', accuracy, on_step=False, on_epoch=True)
        # self.log(f'{prefix}_{self.average}_f1', f1, on_step=False, on_epoch=True)
        # self.log(f'{prefix}_{self.average}_precision', precision, on_step=False, on_epoch=True)
        # self.log(f'{prefix}_{self.average}_recall', recall, on_step=False, on_epoch=True)

        # Log confusion matrix for validation and test sets
        if prefix in ['val', 'test']:
            preds = outputs.argmax(dim=1).tolist()
            # Change the labels here to be the actual labels
            label_names = self.label_names

            # Change the preds here to be the predicted labels with the label names in the same structure as the labels
            preds = [label_names[x] for x in preds]
            # Change the true label names to be the actual labels
            labels = [label_names[x] for x in labels.tolist()]

            # Compute confusion matrix, including all labels
            cm = confusion_matrix(labels, preds, labels=label_names)
        self.log_confusion_matrix(cm, prefix)

    def on_train_epoch_end(self):
        if not self.intermediate_model is None:
            self.intermediate_model.on_train_epoch_end()
        val_loss = self.trainer.callback_metrics.get('val_loss')
        val_accuracy = self.trainer.callback_metrics.get('val_accuracy')
        if val_loss is not None and val_accuracy is not None:
            print(
                f"\n Epoch {self.current_epoch} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        self.val_accuracies.append(val_accuracy.item())
        self.val_losses.append(val_loss.item())
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.best_epoch = self.current_epoch  # Update the best_epoch variable
            print(
                f"New best validation accuracy: {val_accuracy:.4f} at epoch {self.best_epoch}")
        train_loss = self.trainer.callback_metrics.get('train_loss')
        train_accuracy = self.trainer.callback_metrics.get('train_accuracy')
        if train_loss is not None and train_accuracy is not None:
            print(
                f"\n Epoch {self.current_epoch} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        self.train_accuracies.append(train_accuracy.item())
        self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self):
        accuracy, precision, recall, f1 = self.calculate_metrics(
            self.val_labels, self.val_preds)

        # Log metrics
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

        if self.val_preds and self.val_labels:
            cm = confusion_matrix(
                self.val_labels, self.val_preds, labels=range(self.num_classes))
            self.log_confusion_matrix(cm, 'val')
            self.val_preds.clear()
            self.val_labels.clear()

    def on_test_epoch_end(self):
        accuracy, precision, recall, f1 = self.calculate_metrics(
            self.test_labels, self.test_preds)

        # Log metrics
        self.log('test_accuracy', accuracy)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        if self.test_preds and self.test_labels:
            cm = confusion_matrix(
                self.test_labels, self.test_preds, labels=range(self.num_classes))
            self.log_confusion_matrix(cm, 'test')
            self.test_preds.clear()
            self.test_labels.clear()

    def log_confusion_matrix(self, cm, prefix):
        # Save confusion matrix to CSV file in log directory
        csv_file = os.path.join(self.log_dir, f'{prefix}_confusion_matrix.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write confusion matrix to CSV row by row
            writer.writerow([''] + self.label_names)  # Header row
            for i, row in enumerate(cm):
                writer.writerow([self.label_names[i]] + row.tolist())

    def on_save_checkpoint(self, checkpoint):
        # Save the lists to the checkpoint
        checkpoint['train_accuracies'] = self.train_accuracies
        checkpoint['val_accuracies'] = self.val_accuracies[1:]
        checkpoint['train_losses'] = self.train_losses
        checkpoint['val_losses'] = self.val_losses[1:]
        checkpoint['best_epoch'] = self.best_epoch

    def on_load_checkpoint(self, checkpoint):
        # Load the lists from the checkpoint
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_epoch = checkpoint.get('best_epoch', 0)

    def calculate_metrics(self, y_true, y_pred):
        # Calculate metrics using sklearn
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=self.average)
        recall = recall_score(y_true, y_pred, average=self.average)
        f1 = f1_score(y_true, y_pred, average=self.average)

        return accuracy, precision, recall, f1
