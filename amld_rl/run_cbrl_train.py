import torch.backends.mps

from amld_rl.data.crl_datagen import TSPDatasetGenerator
from torch.utils.data import Dataset, DataLoader
from amld_rl.models.combinatorial_rl import CombinatorialRL, CombinatorialRLModel
from amld_rl.trainer.cbrl_trainer import CombinatorialRLTrainer, reward
import IPython
from dataclasses import dataclass
from amld_rl.plots.plotlib import PlotTSPSolution


@dataclass
class ModelConfig:
    embedding_size: int = 128
    hidden_dim: int = 128
    n_glimpses: int = 1
    tanh_exploration: int = 10
    use_tanh: bool = True
    beta: float = .9
    max_grad_norm: float = 2.
    learning_rate: float = 1e-3
    attention: str = "BHD"


@dataclass
class DatasetConfig:
    training_samples: int = int(2.5e5)
    validation_samples: int = int(1e3)


@dataclass
class TSPConfig:
    num_nodes: int = 16


@dataclass
class TrainingConfig:
    n_epochs: int = 5


# Use mps if you have mac M1, if you have cuda replace mps by cuda, else do nothing
device = "mps" if torch.backends.mps.is_available() else "cpu"

if __name__ == '__main__':
    # Configuration dataclasses

    model_params: ModelConfig = ModelConfig()
    dataset_params: DatasetConfig = DatasetConfig()
    tsp_params: TSPConfig = TSPConfig()
    training_params: TrainingConfig = TrainingConfig()

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

    tsp_20_train = CombinatorialRLTrainer(
        n_epochs=training_params.n_epochs,
        model=model,
        train_dataset=train_20_dataset,
        val_dataset=val_20_dataset,
        threshold=3.99,
        device=device
    )
    tsp_20_train.train()
    PlotTSPSolution.plot_tsp_solution(tsp_model, train_20_dataset)
