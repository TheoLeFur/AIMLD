from amld_rl.models.mc_model import MCModel
from amld_rl.envs.easy21 import Easy21
from amld_rl.plots.plotlib import PlotLib
from amld_rl.trainer.trainer import Trainer

if __name__ == '__main__':
    params = {
        "N0": 100,
        "environment": Easy21(),
        "episode_log_freq": 100
    }
    model = MCModel(params)
    trainer = Trainer(model, n_episodes=100000)
    trainer.train()
    PlotLib.plot_value(params["environment"], model.V)
