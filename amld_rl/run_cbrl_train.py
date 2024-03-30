from dataclasses import dataclass

import torch.backends.mps
import argparse

from amld_rl.data.crl_datagen import TSPDatasetGenerator
from amld_rl.models.combinatorial_rl import CombinatorialRL, CombinatorialRLModel
from amld_rl.plots.plotlib import PlotTSPSolution
from amld_rl.trainer.cbrl_trainer import CombinatorialRLTrainer, reward


@dataclass
class FormattedDataclass:
    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = '\n'.join(f'{attr}={getattr(self, attr)}' for attr in self.__annotations__)
        return f'{class_name}(\n{attributes}\n)'


@dataclass
class ModelConfig(FormattedDataclass):
    embedding_size: int = 128
    hidden_dim: int = 128
    n_glimpses: int = 1
    tanh_exploration: int = 10
    use_tanh: bool = True
    beta: float = .9
    max_grad_norm: float = 2.
    learning_rate: float = 3e-4
    attention: str = "BHD"


@dataclass
class DatasetConfig(FormattedDataclass):
    training_samples: int = int(1e5)
    validation_samples: int = int(1e3)


@dataclass
class TSPConfig(FormattedDataclass):
    num_nodes: int = 16


@dataclass
class TrainingConfig(FormattedDataclass):
    n_epochs: int = 5


# Use mps if you have mac M1, if you have cuda replace mps by cuda, else do nothing
device = "mps" if torch.backends.mps.is_available() else "cpu"

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Script with arguments for a specific task.')
    # Define command-line arguments

    # Model Configuration parameters
    parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size (default: 128)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension (default: 128)')
    parser.add_argument('--n_glimpses', type=int, default=1, help='Number of glimpses (default: 1)')
    parser.add_argument('--tanh_exploration', type=int, default=10, help='Tanh exploration (default: 10)')
    parser.add_argument('--use_tanh', action='store_true', help='Use tanh (default: True)')
    parser.add_argument('--beta', type=float, default=0.9, help='Beta value (default: 0.9)')
    parser.add_argument('--max_grad_norm', type=float, default=2.0, help='Maximum gradient norm (default: 2.0)')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--attention', type=str, default='BHD', help='Attention mechanism (default: "BHD")')

    # Dataset Configuration parameters
    parser.add_argument('--training_samples', type=int, default=int(1e5),
                        help='Number of Training Samples (default: 100000')
    parser.add_argument('--validation_samples', type=int, default=int(1e3),
                        help='Number of Validation Samples (default: 1000')

    # Training Configuration parameters
    parser.add_argument('--n_epochs', type=int, default=5, help='Number of training epochs (default: 5')
    # Travelling Salesman Problem parameters
    parser.add_argument('--num_nodes', type=int, default=16, help='Number of nodes in the graph (default : 16')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Configure the dataclasses
    model_params: ModelConfig = ModelConfig(
        embedding_size=args.embedding_size,
        hidden_dim=args.hidden_dim,
        n_glimpses=args.n_glimpses,
        tanh_exploration=args.tanh_exploration,
        use_tanh=args.use_tanh,
        beta=args.beta,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        attention=args.attention
    )
    print(f"Start training model with parameters: \n {model_params}")

    dataset_params: DatasetConfig = DatasetConfig(
        training_samples=args.training_samples,
        validation_samples=args.validation_samples
    )

    print(f"Dataset parameters: \n {dataset_params}")
    tsp_params: TSPConfig = TSPConfig(
        num_nodes=args.num_nodes
    )
    print(f"Travelling Salesman Problem with parameters: \n{tsp_params}")

    training_params: TrainingConfig = TrainingConfig(
        n_epochs=args.n_epochs
    )
    print(f"Training parameters: \n {training_params}")

    train_20_dataset = TSPDatasetGenerator(
        num_nodes=tsp_params.num_nodes,
        num_samples=dataset_params.training_samples)

    val_20_dataset = TSPDatasetGenerator(
        num_nodes=tsp_params.num_nodes,
        num_samples=dataset_params.validation_samples
    )

    tsp_model: CombinatorialRL = CombinatorialRL(
        embedding_size=model_params.embedding_size,
        hidden_dim=model_params.hidden_dim,
        seq_len=tsp_params.num_nodes,
        n_glimpses=model_params.n_glimpses,
        reward=reward,
        tanh_exploration=model_params.tanh_exploration,
        use_tanh=model_params.use_tanh,
        attention=model_params.attention,
        device=device
    )

    model = CombinatorialRLModel(
        combinatorial_rl_net=tsp_model,
        max_grad_norm=model_params.max_grad_norm,
        learning_rate=model_params.learning_rate,
        beta=model_params.beta,
        device=device
    )

    tsp_model.to(device)

    print(f"Start training on device {device} ...")

    tsp_20_train = CombinatorialRLTrainer(
        n_epochs=training_params.n_epochs,
        model=model,
        train_dataset=train_20_dataset,
        val_dataset=val_20_dataset,
        threshold=3.99,
        device=device
    )

    # tsp_20_train.train()
    PlotTSPSolution.plot_tsp_solution(tsp_model, train_20_dataset)
