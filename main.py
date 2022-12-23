import torch

from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import MongoObserver
from sacred.observers import FileStorageObserver

from torch.utils.tensorboard import SummaryWriter

from envs.plane_robot_env import PlaneRobotEnv
from envs.task.reach_goal import ReachGoalTask
from algorithms.ppo.ppo import PPO
from algorithms.sac.sac import SAC


SETTINGS["CONFIG"]["READ_ONLY_CONFIG"] = False

ex = Experiment("robot_arm")


ex.observers.append(MongoObserver(
    url='mongodb://mongo_user:mongo_password@127.0.0.1:27017',
    db_name='sacred'))


@ex.config
def my_config():
    algo = None

    try:
        if algo.lower() == "sac":
            # load sac config
            ex.add_config("config/base_sac.yaml")
        elif algo.lower() == "ppo":
            # load ppo config
            ex.add_config("config/base_ppo.yaml")
        else:
            raise NotImplementedError
    except AttributeError:
        raise ValueError("You have to specifiy the algorithm manually. It is either SAc or PPO possible.")

    # load base config for the environment
    ex.add_config("config/env.yaml")

@ex.main
def main(_config):
    # set torch.seed
    torch.manual_seed(_config["seed"])

    task = ReachGoalTask(epsilon=0.1)

    env = PlaneRobotEnv(
        n_joints=_config["n_joints"],
        segment_lenght=_config["segement_length"],
        task=task)

    # pth for file system logging
    logging_path = f"results/{_config['algo'].lower()}/{_config['n_joints']}_{_config['seed']}"
    logger = SummaryWriter(logging_path)
    ex.observers.append(FileStorageObserver(logging_path))

    if _config["algo"].lower() == "sac":
        algorithm = SAC(
            env,
            logging_writer=logger, 
            sacred_experiment = ex
            )
    elif _config["algo"].lower() == "ppo":
        algorithm = PPO(
            env,
            logging_writer=logger,
            sacred_experiment = ex
            )
    else:
        raise NotImplementedError

    algorithm.train(_config["n_epochs"])

    task 


if __name__ == "__main__":
    ex.run_commandline()
