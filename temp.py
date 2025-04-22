import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
import multiprocessing

# Define a simple dataset
class RandomDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define a simple LightningModule
class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Define a DataModule
class RandomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = RandomDataset(size=1000)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# Main function
def main():
    # Initialize the model and datamodule
    model = SimpleModel()
    data_module = RandomDataModule()

    # Initialize the trainer with ddp_spawn strategy
    trainer = Trainer(strategy='ddp_spawn', max_epochs=1, devices=4 if torch.cuda.is_available() else 0)

    # Fit the model
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    # Set the start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()