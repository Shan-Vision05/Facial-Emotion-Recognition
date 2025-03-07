from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

class FER2013Dataset(Dataset):
  def __init__(self, data, labels):
    self.data = data
    self.labels = labels

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image, label = self.data[idx], self.labels[idx]
    return image, label
  

class ProcessFERDataset:

    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            ])

        train_dataset = FER2013(root="./data", split="train", transform=transform)
        test_dataset = FER2013(root="./data", split="test", transform=transform)

        self.train_data = [train_dataset[i][0] for i in range(len(train_dataset))]
        self.train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]

        self.test_data = [train_dataset[i][0] for i in range(len(test_dataset))]
        self.test_labels = [train_dataset[i][1] for i in range(len(test_dataset))]

        self.emotion_labels= {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }

    def GetTrainTestLoaders(self, batch_size = 64):
        train_dataset = FER2013Dataset(self.train_data, self.train_labels)
        test_dataset = FER2013Dataset(self.test_data, self.test_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader
    
    def GetClassLabels(self):
        return self.emotion_labels
    
    def TestLoadersByPlotting(self, dataloader):
        images, labels = next(iter(dataloader))

        fig, axes = plt.subplots(4, 8, figsize=(10, 5))

        for i, ax in enumerate(axes.flat):
            if i >= len(images):
                break
            img = images[i].squeeze(0)
            ax.imshow(img, cmap="gray")
            ax.set_title(self.emotion_labels[int(labels[i])])
            ax.axis("off")

        plt.tight_layout()
        plt.show()
