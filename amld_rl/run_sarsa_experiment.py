import numpy as np
from amld_rl.envs.easy21 import Easy21
from amld_rl.trainer.trainer import Trainer
from amld_rl.models.sarsa_model import SarsaModel
from amld_rl.plots.plotlib import PlotLib

if __name__ == '__main__':
    params = {
        "N0": 100,
        "environment": Easy21(),
        "num_episodes": 10000,
        "episode_log_freq": 100,
        "lambda_param": 0,
        "gamma": 0.99
    }

    model = SarsaModel(params)
    trainer = Trainer(
        model=model,
        n_episodes=params["num_episodes"],
    )

    trainer.train()

    PlotLib.plot_value(
        env=params["environment"],
        value=model.V
    )
