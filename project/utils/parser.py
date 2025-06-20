from argparse import ArgumentParser
from typing import Tuple, Dict, List


def add_render_sac_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--checkpoint",
        help="--no-documentation-exists--",
        dest="checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    parser.add_argument(
        "--stochastic",
        help="--no-documentation-exists--",
        dest="stochastic",
        action="store_true",
        required=False,
    )
    return parser


from pyargwriter.api.hydra_plugin import add_hydra_parser


def add_train_sac_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    return parser


def setup_entrypoint_parser(
    parser: ArgumentParser,
) -> Tuple[ArgumentParser, Dict[str, ArgumentParser]]:
    subparser = {}
    command_subparser = parser.add_subparsers(dest="command", title="command")
    train_sac = command_subparser.add_parser(
        "train-sac", help="--no-documentation-exists--"
    )
    train_sac = add_train_sac_args(train_sac)
    train_sac = add_hydra_parser(train_sac)
    subparser["train_sac"] = train_sac
    render_sac = command_subparser.add_parser(
        "render-sac", help="--no-documentation-exists--"
    )
    render_sac = add_render_sac_args(render_sac)
    subparser["render_sac"] = render_sac
    return parser, subparser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser, _ = setup_entrypoint_parser(parser)
    return parser
