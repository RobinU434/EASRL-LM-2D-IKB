

from argparse import ArgumentParser
import time

import torch


def add_base_args(parser: ArgumentParser) -> ArgumentParser:
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

    parser.add_argument(
        "--random-seed",
        type=int,
        help="Sets random seed for torch and numpy random number generator. If no seed was given. The current time stamp will be set as random seed.",
    )
    return parser

def add_train_arguments(parser: ArgumentParser) -> ArgumentParser:
    """add arguments 

    Args:
        parser (ArgumentParser): _description_

    Returns:
        ArgumentParser: _description_
    """
    parser = add_general_computing_arguments(parser)
    parser.add_argument(
        "--subdir",
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
    parser.add_argument(
        "--show",
        action="store_true",
        help="shows the finished plot immediately on display.",
    )

    return parser


