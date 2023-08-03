import torch
from argparse import _SubParsersAction, ArgumentParser, ArgumentDefaultsHelpFormatter

from utils.parser import (
    add_base_args,
    add_inference_arguments,
    add_train_arguments,
)


def add_print_config_arguments(parser: ArgumentParser) -> ArgumentParser:
    return parser


def add_print_model_arguments(parser: ArgumentParser) -> ArgumentParser:
    return parser


def setup_inference_subparser(subparser: _SubParsersAction) -> None:
    feed_forward_inference_parser: ArgumentParser = subparser.add_parser(
        "feed-forward", help="Standard feed forward inference", formatter_class=ArgumentDefaultsHelpFormatter
    )
    add_inference_arguments(feed_forward_inference_parser)

    greedy_inference_parser: ArgumentParser = subparser.add_parser(
        "greedy", help="greedy inference",formatter_class=ArgumentDefaultsHelpFormatter
    )
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
        "inference", help=f"Execute inference on {model} model",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    inference_subparser = inference_parser.add_subparsers(
        dest="inference-type", title="inference-type",
    )
    setup_inference_subparser(inference_subparser)

    print_config_parser = subparser.add_parser(
        "print-config", help=f"Print config of {model} model",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    print_config_parser = add_print_config_arguments(print_config_parser)

    print_model_parser = subparser.add_parser(
        "print-model",
        help=f"Print model architecture core components of configured {model} model",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    print_model_parser = add_print_model_arguments(print_model_parser)


def setup_latent_parser(parser: ArgumentParser) -> ArgumentParser:
    """add commands what kind of latent model you want to control

    Args:
        parser (ArgumentParser): blank ArgumentParser

    Returns:
        ArgumentParser: fully equipped ArgumentParser
    """
    parser = add_base_args(parser)
    subparsers: _SubParsersAction = parser.add_subparsers(
        dest="model-type", title="model-type", required=True
    )
    models = ["supervised", "vae"]
    for model in models:
        # SUPERVISED
        supervised_parser = subparsers.add_parser(
            model, help=f"Command pallet to control a {model} model",
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        supervised_subparser = supervised_parser.add_subparsers(
            dest="command", title="command", required=True,
        )
        setup_model_subparser(supervised_subparser, model)

    return parser
