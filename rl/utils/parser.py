from argparse import _SubParsersAction, ArgumentParser

import torch


def setup_base_parser(parser: ArgumentParser) -> ArgumentParser:
    """add arguments for the base setting of a process

    Args:
        parser (ArgumentParser): parser to add arguments to

    Returns:
        ArgumentParser: stuffed ArgumentParser
    """
    parser.add_argument(
        "--log-level",
        choices=["FATAL", "ERROR", "WARN", "INFO", "DEBUG"],
        default="WARN",
        help="if set to true -> debug mode is activated",
    )

    return parser


def add_general_computing_arguments(parser: ArgumentParser) -> ArgumentParser:
    """add arguments every process needs

    Args:
        parser (ArgumentParser): parser to add arguments to

    Returns:
        ArgumentParser: stuffed ArgumentParser
    """
    parser.add_argument(
        "--device",
        type=str,
        help=f"GPU or CPU, current available GPU index: {torch.cuda.device_count() - 1}",
        required=True,
    )
    return parser


def add_train_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser = add_general_computing_arguments(parser)
    parser.add_argument(
        "subdir",
        type=str,
        help="specifies in which subdirectory to store the results",
        required=True,
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="number of runs you want to do with the experiment",
    )
    return parser



def add_inference_arguments(parser: ArgumentParser) -> ArgumentParser:
    """add arguments for every inference method

    Args:
        parser (ArgumentParser): _description_

    Returns:
        ArgumentParser: _description_
    """
    parser = add_general_computing_arguments(parser)
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Absolute or relative path to checkpoint file to run inference on.",
        required=True,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=1,
        help="How many samples to pass through model in inference",
    )
    
    return parser

def setup_algorithm_subparser(subparser: _SubParsersAction, algorithm) -> None:
    train_parser = subparser.add_parser("train", help=f"Train {algorithm}")
    add_train_arguments(train_parser)

    inference_parser = subparser.add_parser("inference", help="Execute ")
    


def setup_rl_parser(parser: ArgumentParser) -> ArgumentParser:
    parser = setup_base_parser(parser)
    subparsers = parser.add_subparsers(dest="algorithm", title="algorithm")

    algorithms = ["sac", "ppo", "ddpg"]
    for algorithm in algorithms:
        algorithm_parser = subparsers.add_parser(
            algorithm, help=f"Command Pallet to control {algorithm}"
        )
        algorithm_subparser = algorithm_parser.add_subparsers(
            dest="command", title="command", required=True
        )
        setup_algorithm_subparser(algorithm_subparser, algorithm)

    return parser
