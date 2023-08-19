from argparse import ArgumentParser
import logging
import sys
from typing import Any, Dict
from envs.plane_robot_env.ikrlenv.env.plane_robot_env import PlaneRobotEnv
from rl.algorithms.ddpg.ddpg import DDPG
from rl.algorithms.ppo.ppo import PPO
from rl.algorithms.sac.sac import SAC
from rl.process.rl_process import RLProcess

from rl.utils.parser import setup_rl_parser
from utils.logging_level import set_log_level


def execute_commands(args: Dict[str, Any], process: RLProcess):
    if args["command"] == "train":
        for i in range(args["num_runs"]):
            print(f"Started {i+1}th experiment")
            process.build()
            try:
                process.train()
            except ValueError:
                # Because some weird nan values during sampling caused by exploding gradients 
                process._algorithm._dump()
                logging.error("run was aborted because of a ValueError")
                continue
            print(f"Completed {i+1}th experiment")
        process.build()
        process.train()
    elif args["command"] == "print-model":
        process.build(no_logger=True)
        process.print_model()
    elif args["command"] == "print-config":
        process.print_config()
    elif args["command"] == "inference":
        process.inference(args["checkpoint"].rstrip("/"))
    else:
        return


def main():
    parser = ArgumentParser("RL - Reinforcement Learning Pipeline")
    parser = setup_rl_parser(parser)

    args_dict = vars(parser.parse_args())
    set_log_level(args_dict.pop("log_level"))

    if args_dict["algorithm"] == "sac":
        algo_type = SAC
    elif args_dict["algorithm"] == "ppo":
        raise NotImplementedError
        # algo_type = PPO
    elif args_dict["algorithm"] == "ddpg":
        raise NotImplementedError
        # algo_type = DDPG
    else:
        parser.print_usage()
        sys.exit()

    process = RLProcess(algo_type, PlaneRobotEnv, **args_dict)
    execute_commands(args_dict, process)


if __name__ == "__main__":
    main()
