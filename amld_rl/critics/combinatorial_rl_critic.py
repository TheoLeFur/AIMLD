import torch
import torch.nn as nn
from amld_rl.neural_nets.attention_module import AttentionModule
from typing import Optional
from amld_rl.neural_nets.mlp import MLPFactory
from torch.autograd import Variable


class CombinatorialRLCritic(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int,
            n_process_blocks: int,
            tanh_exploration: Optional[float] = 10,
            use_tanh: Optional[bool] = False,
            device: Optional[str] = "cpu",
            attention: Optional[str] = "D"
    ) -> None:
        """
        The critic is composed of:

        1. An LSTM encoder layer
        2. An LSTM process block
        3. A two layer decoder module


        The process block performs P steps of computation over the hidden state H.
        At each step, we update the hidden state H by glimpsing at the memory state. The output of
        the glimpse is given as input to the next processing step.

        Finally, we feed the result through the decoder tp give a prediction of the baseline.


        Args:
            embedding_dim: Dimension of embedding
            hidden_dim: Dimension of hidden layer
            n_process_blocks: Number of processing blocks
            tanh_exploration: Hyperparameter if the tanh exploration mode is enabled
            use_tanh: If True, we use tanh exploration
            device: Device where we run the computation
            attention: Attention type. For now we support only Dot and Bahdanau
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_process_blocks = n_process_blocks

        self.encoder = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            batch_first=True
        )

        self.process_block = AttentionModule(
            hidden_dim=hidden_dim,
            use_tanh=use_tanh,
            C=tanh_exploration,
            device=device,
            attention=attention
        )

        self.decoder = MLPFactory.build_mlp(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            output_size=1,
            n_layers=1,
            activation="relu",
            device=device
        )

        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """

        Args:
            inputs: tensor of shape [BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM]

        Returns: baseline approx.

        """

        def _load_on_device(_t: torch.Tensor) -> torch.Tensor:
            """
            Load a tensor on a specific device
            Args:
                _t: Tensor

            Returns:Tensor loaded on self.device

            """

            return _t.to(self.device)

        def _adjust_init_shape_for_inputs(_t: torch.Tensor, _batch_size: int) -> torch.Tensor:
            """

            Args:
                _t: Tensor of shape [HIDDEN_DIM]

            Returns: Tensor of shape [1, BATCH_SIZE, HIDDEN_DIM]

            """

            return _t.unsqueeze(0).repeat(_batch_size, 1).unsqueeze(0)

        batch_size: int = inputs.size(0)
        enc_init_hidden: torch.Tensor = Variable(
            torch.zeros(self.hidden_dim),
        )
        enc_init_cell: torch.Tensor = Variable(
            torch.zeros(self.hidden_dim)
        )
        # Load initial states onto device
        enc_init_hidden, enc_init_cell = map(_load_on_device, [enc_init_hidden, enc_init_cell])
        # Adjust shapes
        enc_init_hidden, enc_init_cell = map(
            lambda t: _adjust_init_shape_for_inputs(t, batch_size),
            [enc_init_hidden, enc_init_cell]
        )
        print(enc_init_hidden.shape)
        print(enc_init_cell.shape)

        encoder_outputs, (enc_hidden, encoder_cell) = self.encoder(
            inputs,
            (enc_init_hidden, enc_init_cell)
        )

        process_block_state: torch.Tensor = enc_hidden.squeeze(0)
        for _ in range(self.n_process_blocks):
            ref, logits = self.process_block(
                process_block_state,
                encoder_outputs
            )
            process_block_state = torch.bmm(
                ref,
                self.softmax(logits).unsqueeze(2)
            ).squeeze(2)

        baseline_approximation = self.decoder(process_block_state)
        return baseline_approximation
