from argparse import ArgumentParser
import json
import logging
import time
import yaml
import numpy as np
import torch

from progress.bar import Bar
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from vae.data.data_set import (
    ActionTargetDatasetV2,
    ConditionalActionTargetDataset,
    TargetGaussianDataset,
    YMode,
)
from vae.data.load_data_set import (
    load_action_dataset,
    load_action_target_dataset,
    load_target_dataset,
)
from vae.utils.extract_angles_and_position import (
    split_conditional_info,
    split_state_information,
)
from vae.utils.loss import (
    DistanceLoss,
    ImitationLoss,
    VAELoss,
    get_loss_func,
    DistVAELoss,
    IKLoss,
)
from vae.utils.fk import forward_kinematics
from vae.model.vae import VariationalAutoencoder, build_model
from vae.utils.post_processing import PostProcessor


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "subdir",
        type=str,
        default="test",
        help="specifies in which subdirectory to store the results",
    )
    parser.add_argument(
        "device",
        type=str,
        help=f"GPU or CPU, current avialable GPU index: {torch.cuda.device_count() - 1}",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="if set to true -> debug mode is activated",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="command just prints config and returns",
    )
    parser.add_argument(
        "--print_model", action="store_true", help="print the model architecture"
    )
    return parser


def load_config():
    with open("config/base_vae.yaml") as f:
        config = yaml.safe_load(f)

    return config


def store_config(config, path: str):
    with open(path + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file)


def log_metrics(
    entity: str, metrics: np.ndarray, logger: SummaryWriter, epoch_idx: int
) -> None:
    logger.add_scalar(
        f"vae/{entity}_r_loss", metrics["reconstruction_loss"].mean(), epoch_idx
    )
    logger.add_scalar(f"vae/{entity}_kl", metrics["kl_loss"].mean(), epoch_idx)
    logger.add_scalar(f"vae/{entity}_std", metrics["std"].mean(), epoch_idx)
    logger.add_scalar(
        f"vae/{entity}_imitation_loss", metrics["imitation_loss"].mean(), epoch_idx
    )
    logger.add_scalar(
        f"vae/{entity}_distance_loss", metrics["distance_loss"].mean(), epoch_idx
    )


def run_model(
    autoencoder: VariationalAutoencoder,
    data: DataLoader,
    criterion: IKLoss,
    device: str,
    train: bool = False,
) -> np.ndarray:
    """runs the given autoencoder on the given dataset

    Args:
        autoencoder (VariationalAutoencoder): model to run the data on
        data (DataLoader): dataloader to draw data from. It is recommended that data.dataset inherit from VAEDataset
        criterion (IKLoss): loss function
        device (str): _description_
        train (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray: structured array with keys:
        - reconstruction_loss
        - kl_loss
        - total_loss
        - std
        - imitation_loss
        - distance_loss
    """

    if not isinstance(data.dataset, VAEDataset):
        logging.warning(
            "given dataset does not inherit from VAEDataset. As long it returns (x, c_enc, c_dec, y) on __getitem__ it is fine"
        )
    # prepare logging
    log_metrics_array = []
    log_metrics_dt = [
        ("reconstruction_loss", np.float32),
        ("kl_loss", np.float32),
        ("total_loss", np.float32),
        ("std", np.float32),
        ("imitation_loss", np.float32),
        ("distance_loss", np.float32),
    ]
    for x, c_enc, c_dec, y in data:
        x = x.to(device)
        c_enc = c_enc.to(device)
        c_dec = c_dec.to(device)
        y = y.to(device)

        x_hat, mu, log_std = autoencoder(
            x, c_enc, c_dec
        )  # out shape: (batch_size, number of joints)

        # TODO: it is a bit hacky and has to be adapted for other datasets where the current angles are not inside c_dec
        current_angles = c_dec[:, -autoencoder.output_dim :]
        action = x_hat + current_angles
        if criterion.target_mode == YMode.ACTION:
            # y is expected to be the target action we want to encode
            y = y + current_angles
        loss = criterion(y=y, x_hat=action, mu=mu, log_std=log_std)

        if train:
            autoencoder.train(loss)

        # log metrics in the correct order
        log_metrics_array.append(
            np.array(
                [
                    criterion.r_loss.cpu().item(),  # reconstruction_loss
                    criterion.kl_loss.cpu().item(),  # kl loss
                    loss.cpu().item(),  # total loss
                    log_std.mean().cpu().item(),  # std
                    criterion.imitation_loss.cpu().item(),  # imitation_loss
                    criterion.distance_loss.cpu().item(),  # distance_loss
                ]
            )
        )
    # make structured array
    log_metrics_array = np.stack(log_metrics_array)
    log_metrics_array = np.rec.fromarrays(log_metrics_array.T, dtype=log_metrics_dt)

    return log_metrics_array


def train(
    autoencoder: VariationalAutoencoder,
    train_data: DataLoader,
    val_data: DataLoader,
    test_data: DataLoader,
    loss_func: VAELoss,
    logger: SummaryWriter = None,
    epochs: int = 20,
    device: str = "cpu",
    learning_rate: float = 1e-3,
    val_interval: int = 5,
    path: str = "results/vae",
):
    # post_processor = PostProcessor(- 2 * torch.pi, 2 * torch.pi)
    # post_processor = PostProcessor(- torch.pi, torch.pi)

    for epoch_idx in range(epochs):
        # reset logging history per epoch
        autoencoder.reset_history()

        # train loop
        train_metrics_array = run_model(
            autoencoder=autoencoder,
            data=train_data,
            criterion=loss_func,
            device=device,
            train=True,
        )

        # val loop
        if epoch_idx % val_interval == 0:
            val_metrics_array = run_model(
                autoencoder=autoencoder,
                data=val_data,
                criterion=loss_func,
                device=device,
                train=False,
            )

            # store model checkpoint
            autoencoder.store(
                path=path
                + f"/model_{epoch_idx}_val_r_loss_{val_metrics_array['reconstruction_loss'].mean().item():.4f}.pt",
                epoch_idx=epoch_idx,
                metrics=val_metrics_array,
            )

        # LOGGING
        if logger is not None and epoch_idx % val_interval == 0:
            print(
                f"epoch {epoch_idx}: \n\
                train: loss: {train_metrics_array['reconstruction_loss'].mean()} kl_div: {train_metrics_array['kl_loss'].mean()} \n\
                val: loss: {val_metrics_array['reconstruction_loss'].mean()} kl_div: {val_metrics_array['kl_loss'].mean()}"
            )

            log_metrics("train", train_metrics_array, logger, epoch_idx)
            # logger.add_histogram("vae/train_latent_mu", mu_array, epoch_idx)

            # autoencoder.log_parameters(epoch_idx)
            # autoencoder.log_gradients(epoch_idx)
            autoencoder.log_decoder_distr(epoch_idx)
            autoencoder.log_z_grad(epoch_idx)

            log_metrics("val", val_metrics_array, logger, epoch_idx)
    # test loop
    test_metrics_array = run_model(
        autoencoder=autoencoder,
        data=test_data,
        criterion=loss_func,
        device=device,
        train=False,
    )
    log_metrics("test", test_metrics_array, logger, epoch_idx)

    print(
        f"test: loss: {test_metrics_array['reconstruction_loss'].mean()} kl_div: {test_metrics_array['kl_loss'].mean()}"
    )

    if logger is not None:
        logger.add_hparams(
            hparam_dict={
                "learning_rate": learning_rate,
                "val_interval": val_interval,
                "kl_loss_weight": config["kl_loss_weight"],
                "reconstruction_loss_weight": config["reconstruction_loss_weight"],
                "dataset": config["dataset"],
                # "normalize": config["normalize"]
            },
            metric_dict={
                "vae/test_loss": test_metrics_array["total_loss"].mean(),
                "vae/test_r_loss": test_metrics_array["reconstruction_loss"].mean(),
                "vae/test_kl": test_metrics_array["kl_loss"].mean(),
            },
        )

    return autoencoder


if __name__ == "__main__":
    parser = setup_parser(ArgumentParser())
    args = parser.parse_args()

    # set logging level
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    config = load_config()
    config["device"] = args.device

    input_dim = config["num_joints"]
    if config["dataset"] == "action":
        train_loader, val_loader, test_loader = load_action_dataset(config)

    elif config["dataset"] == "action_target_v1":
        train_loader, val_loader, test_loader = load_action_target_dataset(config)

    elif config["dataset"] == "action_target_v2":
        train_loader, val_loader, test_loader = load_action_target_dataset(config)

    elif config["dataset"] == "conditional_action_target":
        train_loader, val_loader, test_loader = load_action_target_dataset(config)

    elif config["dataset"] == "target_gaussian":
        train_loader, val_loader, test_loader = load_target_dataset(config)
    else:
        raise ValueError("you have not selected the right dataset in your config")

    loss_config = config["loss_func"]
    loss_config["target_mode"] = train_loader.dataset.y_mode
    loss_func = get_loss_func(loss_config, args.device)

    if args.print_config:
        print(json.dumps(config, sort_keys=True, indent=4))
        exit()

    path = f"results/vae/{args.subdir}/{config['num_joints']}_{config['latent_dim']}_{int(time.time())}"
    logger = SummaryWriter(path)

    autoencoder = build_model(
        config,
        train_loader.dataset.input_dim,
        train_loader.dataset.conditional_dim,
        logger,
    )

    if args.print_model:
        print(autoencoder)
        exit()

    # store config
    store_config(config, path)

    train(
        autoencoder,
        train_loader,
        val_loader,
        test_loader,
        loss_func=loss_func,
        logger=logger,
        epochs=config["epochs"],
        device=args.device,
        learning_rate=config["learning_rate"],
        val_interval=5,
        path=path,
    )
