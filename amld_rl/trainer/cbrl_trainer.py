from typing import List, Optional

import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from amld_rl.trainer.base_trainer import BaseTrainer
from amld_rl.models.combinatorial_rl import CombinatorialRLModel


class CombinatorialRLTrainer(BaseTrainer):
    def __init__(
            self,
            n_epochs: int,
            model: CombinatorialRLModel,
            train_dataset: Dataset,
            val_dataset: Dataset,
            batch_size=128,
            threshold=None,
            device: Optional[str] = "cpu",
            plot_update_period: Optional[int] = 50,
            validation_period: Optional[int] = 100
    ) -> None:
        """

        @param n_epochs:
        @param model:
        @param train_dataset:
        @param val_dataset:
        @param batch_size:
        @param threshold:
        @param device:
        @param plot_update_period:
        @param validation_period:
        """

        super().__init__()

        self.n_epochs = n_epochs
        self.device = device
        self.model = model

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        self.train_tour: List = []
        self.val_tour: List = []
        self.epochs: int = 0

        # Logging
        self.plot_update_period = plot_update_period
        self.validation_period = validation_period

    def train(self):

        for epoch in range(self.n_epochs):
            for batch_id, sample_batch in enumerate(tqdm(self.train_loader)):

                self.model.combinatorial_rl_net.train()
                sample_batch = sample_batch.to(self.device)
                inputs = Variable(sample_batch)

                training_step_logs = self.model.step(
                    batch_id,
                    inputs
                )

                loss = training_step_logs["loss"]
                self.train_tour.append(loss.cpu().detach().numpy())

                print(f"Loss Value: {loss}\n")

                if batch_id % self.validation_period == 0:
                    self.model.combinatorial_rl_net.eval()
                    for val_batch in self.val_loader:
                        with torch.no_grad():
                            val_batch = val_batch.to(self.device)
                            inputs = Variable(val_batch)
                            R, probs, actions, actions_idxs = self.model.val_step(
                                inputs)
                            self.val_tour.append(
                                R.mean().cpu().detach().numpy())

            if self.threshold and self.train_tour[-1] < self.threshold:
                print("STOP!\n")
                break
            # self.model.combinatorial_rl_net.save_weights(epoch, "checkpoints")
            self.plot(epoch)
            print(f"Epoch {epoch} data saved in directory ckeckpoints \n")
            self.epochs += 1

    def plot(self, epoch):

        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('train tour length: epoch %s reward %s' % (
            epoch, self.train_tour[-1] if len(self.train_tour) else 'collecting'))
        plt.plot(self.train_tour)
        plt.grid()
        plt.subplot(132)
        plt.title(
            'val tour length: epoch %s reward %s' % (epoch, self.val_tour[-1] if len(self.val_tour) else 'collecting'))
        plt.plot(self.val_tour)
        plt.grid()
        plt.show()


def reward(sample_solution, device: Optional[str] = "cpu"):
    """
    Args:
    sample_solution seq_len of [batch_size]
    """
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = Variable(torch.zeros([batch_size], device=device))

    for i in range(n - 1):
        tour_len += torch.norm(sample_solution[i] -
                               sample_solution[i + 1], dim=1)

    tour_len += torch.norm(sample_solution[n - 1] - sample_solution[0], dim=1)

    return tour_len
