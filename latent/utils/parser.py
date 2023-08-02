import torch
from argparse import _SubParsersAction, ArgumentParser


def add_process_arguments(parser: ArgumentParser) -> ArgumentParser:
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
    """add arguments every computing job process needs

    Args:
        parser (ArgumentParser): parser to add arguments to

    Returns:
        ArgumentParser: stuffed ArgumentParser
    """
    parser = add_general_computing_arguments(parser)
    parser.add_argument(
        "--subdir",
        type=str,
        help="specifies in which subdirectory to store the results",
        required=True,
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


def add_print_config_arguments(parser: ArgumentParser) -> ArgumentParser:
    return parser


def add_print_model_arguments(parser: ArgumentParser) -> ArgumentParser:
    return parser


def setup_inference_subparser(subparser: _SubParsersAction) -> None:
    feed_forward_inference_parser = subparser.add_parser(
        "feed-forward", help="Standard feed forward inference"
    )
    add_inference_arguments(feed_forward_inference_parser)

    greedy_inference_parser = subparser.add_parser("greedy", help="greedy inference")
    add_inference_arguments(greedy_inference_parser)


def setup_model_subparser(subparser: _SubParsersAction, model: str) -> None:
    """add command and arguments for the command which are common for each model

    Args:
        subparser (_SubParsersAction): reference to ArgumentParser
        model_entity (str): to which kind of model we add
    """
    train_parser = subparser.add_parser("train", help=f"Train {model} model")
    train_parser = add_train_arguments(train_parser)

    inference_parser: ArgumentParser = subparser.add_parser(
        "inference", help=f"Execute inference on {model} model"
    )
    inference_subparser = inference_parser.add_subparsers(
        dest="inference-type", title="inference-type"
    )
    setup_inference_subparser(inference_subparser)

    print_config_parser = subparser.add_parser(
        "print-config", help=f"Print config of {model} model"
    )
    print_config_parser = add_print_config_arguments(print_config_parser)

    print_model_parser = subparser.add_parser(
        "print-model",
        help=f"Print model architecture core components of configured {model} model",
    )
    print_model_parser = add_print_model_arguments(print_model_parser)


def setup_latent_parser(parser: ArgumentParser) -> ArgumentParser:
    """add commands what kind of latent model you want to control

    Args:
        parser (ArgumentParser): blank ArgumentParser

    Returns:
        ArgumentParser: fully equipped ArgumentParser
    """
    parser = add_process_arguments(parser)
    subparsers = parser.add_subparsers(
        dest="model-type", title="model-type", required=True
    )

    models = ["supervised", "vae"]
    for model in models:
        # SUPERVISED
        supervised_parser = subparsers.add_parser(
            model, help=f"Command pallet to control a {model} model"
        )
        supervised_subparser = supervised_parser.add_subparsers(
            dest="command", title="command", required=True
        )
        setup_model_subparser(supervised_subparser, model)

    return parser
