

from argparse import ArgumentParser
import sys
from typing import Any, Dict
from envs.plane_robot_env.ikrlenv.env.plane_robot_env import PlaneRobotEnv
from rl.algorithms.ddpg.ddpg import DDPG
from rl.algorithms.ppo.ppo import PPO
from rl.algorithms.sac.sac import SAC
from rl.process.rl_process import RLProcess

from rl.utils.parser import setup_rl_parser

def execute_commands(args: Dict[str, Any], process: RLProcess):
    if args["command"] == "train":
        process.build()
        process.train()
    elif args["command"] == "print-model":
        process.print_model()
    elif args["command"] == "print-config":
        process.print_config()
    else:
        return 

def main():
    parser = ArgumentParser("RL - Reinforcement Learning Pipeline")
    parser = setup_rl_parser(parser)

    args_dict = vars(parser.parse_args())


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