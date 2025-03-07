from infra.DataProcessor import ProcessFERDataset
from models.FER_Model import FER2013_Model
from Trainer import FERTrainer
import torch
from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import numpy as np

def Train(model, train_loader, test_loader, epochs, plot = True, useScheduler = False):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  trainer = FERTrainer(model, device, train_loader, test_loader, useScheduler)

  train_loss_list = np.zeros(epochs)
  train_acc_list = np.zeros(epochs)
  test_loss_list = np.zeros(epochs)
  test_acc_list = np.zeros(epochs)

  initial = True
  for epoch in tqdm(range(1, epochs+1)):
    train_loss, train_acc = trainer.train_step()
    test_loss, test_acc = trainer.eval_step()

    train_loss_list[epoch-1] = train_loss
    train_acc_list[epoch-1] = train_acc
    test_loss_list[epoch-1] = test_loss
    test_acc_list[epoch-1] = test_acc

    if initial:
      print(f'Epoch {epoch}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f} Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')
      initial = False

    if epoch % 10 == 0:
      print(f'Epoch {epoch}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f} Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')

    if epoch == 75:
      print(f'Epoch {epoch}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f} Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}')

  if plot:
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def GetPrecisionRecall(model, test_dataloader, emotion_labels):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  y_true_labels = []
  y_predictions = []

  with torch.inference_mode():
    for batch, (X, y) in enumerate(test_dataloader):
      X, y = X.to(device), y.to(device)

      y_pred = model(X)
      y_pred = y_pred.argmax(dim=1)

      for i in range(len(y)):
        y_true_labels.append(y[i].item())
        y_predictions.append(y_pred[i].item())

  y_true_labels = np.array(y_true_labels)
  y_predictions = np.array(y_predictions)

  # Compute precision, recall, F1-score, and support for each class
  precision, recall, f1, support = precision_recall_fscore_support(y_true_labels, y_predictions, zero_division=0)

  # Compute the confusion matrix
  conf_matrix = confusion_matrix(y_true_labels, y_predictions)

  # Generate classification report
  class_report = classification_report(y_true_labels, y_predictions, zero_division=0)

  # Print per-class metrics
  print("Class-wise Precision, Recall, F1-score, and Support:")
  for i in range(len(precision)):
      print(f"Class {emotion_labels[i]} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-score: {f1[i]:.4f}, Support: {support[i]}")

  # Print confusion matrix
  print("\nConfusion Matrix:")
  print(conf_matrix)

  # Print detailed classification report
  print("\nClassification Report:")
  print(class_report)


def main():
    dataProcessor = ProcessFERDataset()

    train_dataloader, test_dataloader = dataProcessor.GetTrainTestLoaders(batch_size=64)
    
    dataProcessor.TestLoadersByPlotting(train_dataloader)
    dataProcessor.TestLoadersByPlotting(test_dataloader)

    ferModel = FER2013_Model(input_shape=1, output_shape=7)

    Train(ferModel, train_dataloader, test_dataloader, 75, True)

    GetPrecisionRecall(ferModel, test_dataloader, dataProcessor.GetClassLabels())



if __name__ == '__main__':
    main()
