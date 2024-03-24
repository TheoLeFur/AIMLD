import matplotlib.pyplot as plt
import torch
import matplotlib.cm as cm
import numpy as np
from IPython.display import display, clear_output

from amld_rl.data.crl_datagen import TSPDatasetGenerator
from typing import List, Optional
from amld_rl.models.combinatorial_rl import CombinatorialRL
import random


class PlotLib:

    @staticmethod
    def plot_value(env, value: np.ndarray) -> None:
        """
        3d plot of value function
        @param env: Environment
        @param value: array containing the value function
        @return: None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def state_val(x, y):
            return value[x, y]

        X = np.arange(0, env.dealer_value_count, 1)
        Y = np.arange(0, env.player_value_count, 1)
        X, Y = np.meshgrid(X, Y)
        Z = state_val(X, Y)
        ax.plot_surface(X, Y, Z, cmap=cm.bwr, antialiased=False)
        plt.show()


class PLotInteractiveLoss:

    @staticmethod
    def init_plot():
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        train_loss, = ax.plot([], [], 'b', label='Training loss')
        val_loss, = ax.plot([], [], 'b', label='Validation loss')
        ax.grid(True)
        ax.legend()
        plt.show()

        return train_loss, val_loss, fig, ax

    @staticmethod
    def update_plot(
            fig,
            ax,
            train_loss,
            val_loss,
            train_losses,
            val_losses
    ):
        train_loss.set_data(range(1, len(train_losses) + 1), train_losses)
        val_loss.set_data(range(1, len(val_losses) + 1), val_losses)

        ax.relim()
        ax.autoscale_view()
        display(fig)

        clear_output(wait=True)
        plt.show()

        return train_loss, val_loss

    @staticmethod
    def show_plot():
        plt.ioff()
        plt.show()


class PlotTSPSolution:

    @staticmethod
    def plot_tsp_solution(
            model: CombinatorialRL,
            dataset: TSPDatasetGenerator,
            n_samples=100
    ) -> None:
        """
        Plots the solution to the TSP given by the model given a random instance

        @param dataset: dataset used for training
        @param model: Combinatorial RL model
        @return: None
        """
        random_idx = random.randint(0, dataset.size - 1)
        points = dataset[random_idx]

        best_reward, best_actions = model.greedy_sample(
            input_graph=points,
            n_samples=n_samples
        )
        # Plot the points
        plt.scatter(points[0], points[1], color='blue', label='Points')
        # Detach actions from gpu
        best_actions = list(
            map(lambda t: t.cpu().detach().numpy(), best_actions))
        # Connect the points with lines
        for i in range(len(best_actions) - 1):
            plt.plot([best_actions[i][0][0], best_actions[i + 1][0][0]],
                     [best_actions[i][0][1], best_actions[i + 1][0][1]], color='red',
                     linestyle='-')

        # Connect the last point to the first point to form a closed loop
        plt.plot([best_actions[-1][0][0], best_actions[0][0][0]],
                 [best_actions[-1][0][1], best_actions[0][0][1]],
                 color='red', linestyle='-')

        plt.title('Model solution for the TSP')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.legend()
        plt.show()
