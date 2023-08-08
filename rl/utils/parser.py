from argparse import _SubParsersAction, ArgumentDefaultsHelpFormatter, ArgumentParser

from utils.parser import add_base_args, add_general_computing_arguments, add_train_arguments



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
    train_parser: ArgumentParser = subparser.add_parser("train", help=f"Train {algorithm}", formatter_class=ArgumentDefaultsHelpFormatter)
    add_train_arguments(train_parser)

    inference_parser: ArgumentParser = subparser.add_parser("inference", help="Execute ", formatter_class=ArgumentDefaultsHelpFormatter)
    add_inference_arguments(inference_parser)

    print_config_parser: ArgumentParser = subparser.add_parser("print-config", help="Print config for the algorithm you want to train")

    print_model_parser: ArgumentParser = subparser.add_parser("print-model", help="Print architecture of networks inside SAC")



def setup_rl_parser(parser: ArgumentParser) -> ArgumentParser:
    parser = add_base_args(parser)
    subparsers = parser.add_subparsers(dest="algorithm", title="algorithm")

    algorithms = ["sac", "ppo", "ddpg"]
    for algorithm in algorithms:
        algorithm_parser = subparsers.add_parser(
            algorithm, help=f"Command Pallet to control {algorithm}", formatter_class=ArgumentDefaultsHelpFormatter
        )
        algorithm_subparser = algorithm_parser.add_subparsers(
            dest="command", title="command", required=True
        )
        setup_algorithm_subparser(algorithm_subparser, algorithm)

    return parser
