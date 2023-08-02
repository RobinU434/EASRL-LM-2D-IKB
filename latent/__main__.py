from argparse import ArgumentParser, Namespace
import logging
import sys
from typing import Any, Dict
from latent.process.latent_process import LatentProcess
from latent.process.vae_process import VAEProcess

from latent.utils.parser import setup_latent_parser
from process.base_process import Process
from latent.process.supervised_process import SupervisedProcess
from utils.logging_level import set_log_level


def execute_commands(args: Dict[str, Any], process: LatentProcess):
    print(process._save_dir)
    if args["command"] == "train":
        process.build()
        process.train()
    elif args["command"] == "inference":
        if "checkpoint" not in args.keys():
            logging.error("No checkpoint given")
            return
        process.load_checkpoint()
        if args["inference-type"] == "feed-forward":
            process.feed_forward_inference()
        elif args["inference-type"] == "greedy":
            process.greedy_inference()
    elif args["command"] == "print-config":
        process.print_config()
    elif args["command"] == "print-model":
        process.build()
        process.print_model()


def main():
    parser = ArgumentParser(description="Latent Model - train and do inference")
    parser = setup_latent_parser(parser)
    args = vars(parser.parse_args())
    set_log_level(args["log_level"])

    if args["model-type"] == "supervised":
        # init supervised process but dont build it
        process = SupervisedProcess(**args)
    elif args["model-type"] == "vae":
        process = VAEProcess(**args)
    else:
        parser.print_usage()
        sys.exit()

    execute_commands(args, process)


if __name__ == "__main__":
    main()
