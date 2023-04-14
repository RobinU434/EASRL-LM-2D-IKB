import copy
import time
import yaml
import torch
import logging

from argparse import ArgumentParser, Namespace
from gym.wrappers import RescaleAction
from torch.utils.tensorboard import SummaryWriter

from envs.plane_robot_env import PlaneRobotEnv
from envs.task.imitation_learning import ImitationTask
from envs.task.reach_goal import ReachGoalTask
from algorithms.ppo.ppo import PPO
from algorithms.sac.sac import SAC
from algorithms.sac.actor.base_actor import Actor
from algorithms.sac.actor.latent_actor import LatentActor
from logger.fs_logger import FileSystemLogger


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "algorithm",
        type=str,
        help="specify which algorithm to use")
    parser.add_argument(
        "subdir",
        type=str,
        default="test",
        help="specifies in which subdirectory to store the results")
    parser.add_argument(
        "num_runs", 
        type=int,
        default=1,
        help="number of runs you want to do with the experiment")
    return parser


def load_config(args: Namespace) -> dict:
    # First the env config file
    with open("config/env.yaml") as f:
        config = yaml.safe_load(f)

    # than add the algorithm config file
    algo = args.algorithm
    try:
        if algo.lower() == "sac":
            # load sac config
            with open("config/base_sac.yaml") as f:
                config = {**config, **yaml.safe_load(f)}
        elif algo.lower() == "ppo":
            # load ppo config
            with open("config/base_ppo.yaml") as f:
                config = {**config, **yaml.safe_load(f)}
        else:
            raise NotImplementedError(f"Use an algorithm form this list: [sac, ppo], you used{algo}")
    except AttributeError:
        raise ValueError("You have to specify the algorithm manually. It is either SAC or PPO possible.")

    # finally add all arguments into config and maybe do an overwrite
    config = {**config, **vars(args)}
    
    return config


def extract_actor_config(config: dict) -> dict:
    # there is the normal fead forward actor and the latent actor
    if config["normal_actor"]["enabled"]:
        actor_config = config["normal_actor"]
        actor_config["type"] = Actor
        return actor_config
    elif config["latent_actor"]["enabled"]:
        actor_config = config["latent_actor"]
        actor_config["type"] = LatentActor
        return actor_config
    else:
        logging.error("There must be at least one actor enabled")


def build_env(config: dict):
    print("action mode: ", config["action_mode"])
    print("action magnitude: ", config["action_magnitude"])
    
    # select task
    print(config["task"])
    if config["task"] == ReachGoalTask.__name__:
        task = ReachGoalTask(config = config)
    elif config["task"] == ImitationTask.__name__:
        task = ImitationTask(config=config)
    print(config["n_joints"], " joints")

    env = PlaneRobotEnv(
        n_joints=config["n_joints"],
        segment_length=config["segment_length"],
        task=task, 
        action_mode=config["action_mode"]
        )
    
    if config["rescale_actions"]["enabled"]:
        env = RescaleAction(env, config["rescale_actions"]["min"], config["rescale_actions"]["max"])
        logging.info("action_space: ", env.action_space)
    else:
        logging.info("action_space: ", env.action_space)
    
    return env
            

def main(config):
    actor_config = extract_actor_config(config)

    # reseed the environment
    time_stamp = int(time.time())
    torch.manual_seed(time_stamp)

    env = build_env(config)
    
    # pth for file system logging
    logging_path = f"results/{config['algorithm'].lower()}/{config['subdir']}/{actor_config['type'].__name__}/{config['n_joints']}_{time_stamp}"
    logger = SummaryWriter(logging_path)
    logging.info("initialized tensorboard summary writer")

    # store config 
    with open(logging_path + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file)
    # log config
    # logger.add_hparams(config, {})

    print("")
    print("SAC:")
    print("target entropy", config["target_entropy"])
    print("gamma: ", config["gamma"])
    print("actor: ", actor_config["type"].__name__)
    
    if config["algorithm"].lower() == "sac":
        algorithm = SAC(
            env,
            logging_writer=logger,
            fs_logger=FileSystemLogger(logging_path),
            lr_pi=config["lr_pi"],
            lr_q=config["lr_q"],
            init_alpha=config["init_alpha"],
            gamma=config["gamma"],
            batch_size=config["batch_size"],
            buffer_limit=config["buffer_limit"],
            start_buffer_size=config["start_buffer_size"],
            train_iterations=config["train_iterations"],
            tau=config["tau"],
            target_entropy=config["target_entropy"],
            lr_alpha=config["lr_alpha"],
            action_covariance_decay = config["action_covariance_decay"],
            action_covariance_mode = config["action_covariance_mode"],
            action_magnitude=config["action_magnitude"],
            actor_config=actor_config,
            )
    elif config["algorithm"].lower() == "ppo":
        algorithm = PPO(
            env,
            logging_writer=logger,
            fs_logger=FileSystemLogger(logging_path),
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            lmbda=config["lmbda"],
            eps_clip=config["eps_clip"],
            K_epoch=config["K_epoch"],
            rollout_len=config["rollout_len"],
            buffer_size=config["buffer_size"],
            minibatch_size=config["minibatch_size"],
            action_covariance_decay = config["action_covariance_decay"],
            action_covariance_mode = config["action_covariance_mode"]
            )
    else:
        raise NotImplementedError
    
    algorithm.train(config["n_epochs"])


if __name__ == "__main__":
    # check which algorithm
    parser = setup_parser(ArgumentParser())

    args = parser.parse_args()
    # load config file
    config = load_config(args)

    print(f"Start to do {args.num_runs} experiment")
    for i in range(args.num_runs):
        print(f"Started {i}th experiment")
        try:
            main(config)
        except ValueError:
            # Because some weird nan values during sampling caused by exploding gradients 
            logging.warning("run was aborted because of a ValueError")
            continue
        print(f"completed {i}th experiment")

   
