from argparse import ArgumentParser
import json
import time
import yaml
import numpy as np
import torch

from progress.bar import Bar
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vae.data.data_set import ActionTargetDatasetV2, ConditionalActionTargetDataset, ConditionalTargetDataset
from vae.data.load_data_set import load_action_dataset, load_action_target_dataset, load_target_dataset
from vae.utils.extract_angles_and_position import split_conditional_info, split_state_information
from vae.utils.loss import DistanceLoss, ImitationLoss, VAELoss, get_loss_func, DistVAELoss, IKLoss
from vae.utils.fk import forward_kinematics
from vae.model.vae import VariationalAutoencoder, build_model
from vae.utils.post_processing import PostProcessor


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "subdir",
        type=str,
        default="test",
        help="specifies in which subdirectory to store the results")
    parser.add_argument(
        "device",
        type=str,
        help=f"GPU or CPU, current avialable GPU index: {torch.cuda.device_count() - 1}")
    parser.add_argument(
        "--print_config",
        action='store_true',
        help='command just prints config and returns'
    )
    parser.add_argument(
        "--print_model",
        action="store_true",
        help="print the model architecture"
    )
    return parser


def load_config():
    with open("config/base_vae.yaml") as f:
        config = yaml.safe_load(f)

    return config


def store_config(config, path: str):
    with open(path + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file)


def log_metrics(entity: str, metrics: np.ndarray, logger: SummaryWriter, epoch_idx: int) -> None:
    logger.add_scalar(f"vae/{entity}_r_loss", metrics["reconstruction_loss"].mean(), epoch_idx)
    logger.add_scalar(f"vae/{entity}_kl", metrics["kl_loss"].mean(), epoch_idx)
    logger.add_scalar(f"vae/{entity}_std", metrics["std"].mean(), epoch_idx)
    logger.add_scalar(f"vae/{entity}_imitation_loss", metrics["imitation_loss"].mean(), epoch_idx)
    logger.add_scalar(f"vae/{entity}_distance_loss", metrics["distance_loss"].mean(), epoch_idx)


def run_model(
    autoencoder: VariationalAutoencoder,
    data: DataLoader,
    loss_func: IKLoss,
    device: str,
    train: bool = False,
    ):


    # prepare logging
    log_metrics_array = []
    log_metrics_dt = [
        ("reconstruction_loss", np.float32),
        ("kl_loss", np.float32),
        ("total_loss", np.float32),
        ("std", np.float32),
        ("imitation_loss", np.float32),
        ("distance_loss", np.float32)]
    for x, c_enc, c_dec, y in data:
        x = x.to(device)
        c_enc = c_enc.to(device)
        c_dec = c_dec.to(device)
        y = y.to(device)

        x_hat, mu, log_std = autoencoder(x, c_enc, c_dec)  # out shape: (batch_size, number of joints) 

        loss = loss_func(y=y, x_hat=x_hat, mu=mu, log_std=log_std)
        
        if train:
            autoencoder.train(loss)

        # log metrics in the correct order
        log_metrics_array.append(
            np.array([
                loss_func.r_loss.cpu().item(),  # reconstruction_loss
                loss_func.kl_div.cpu().item(),  # kl loss
                loss.cpu().item(),  # total loss
                log_std.mean().cpu().item(),  # std
                loss_func.imitation_loss.cpu().item(),  # imitation_loss
                loss_func.distance_loss.cpu().item(),  # distance_loss
            ])
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
    path: str = "results/vae"
    ):
    
    # post_processor = PostProcessor(- 2 * torch.pi, 2 * torch.pi)
    # post_processor = PostProcessor(- torch.pi, torch.pi)
    
    for epoch_idx in range(epochs):
        # reset logging history per epoch
        autoencoder.reset_history()

        # train loop
        train_metrics_array = run_model(
                autoencoder,
                train_data,
                loss_func,
                device,
                train=True
            )
        
        # val loop
        if epoch_idx % val_interval == 0:
            val_metrics_array = run_model(
                autoencoder,
                val_data,
                loss_func,
                device,
                train=False
            )

            # store model checkpoint
            autoencoder.store(
                path=path + f"/model_{epoch_idx}_val_r_loss_{val_metrics_array['reconstruction_loss'].mean().item():.4f}.pt",
                epoch_idx=epoch_idx,
                metrics=val_metrics_array
                )
            
        # LOGGING
        if logger is not None and epoch_idx % val_interval == 0:    
            print(f"epoch {epoch_idx}: \n\
                train: loss: {train_metrics_array['reconstruction_loss'].mean()} kl_div: {train_metrics_array['kl_loss'].mean()} \n\
                val: loss: {val_metrics_array['reconstruction_loss'].mean()} kl_div: {val_metrics_array['kl_loss'].mean()}")
            
            log_metrics("train", train_metrics_array, logger, epoch_idx)
            # logger.add_histogram("vae/train_latent_mu", mu_array, epoch_idx)
            
            # autoencoder.log_parameters(epoch_idx)
            # autoencoder.log_gradients(epoch_idx)
            autoencoder.log_decoder_distr(epoch_idx)
            autoencoder.log_z_grad(epoch_idx)
            
            log_metrics("val", val_metrics_array, logger, epoch_idx)
    # test loop
    test_metrics_array = run_model(
        autoencoder,
        test_data,
        loss_func,
        device,
        train=False
    )
    log_metrics("test", test_metrics_array, logger, epoch_idx)
            
    print(f"test: loss: {test_metrics_array['reconstruction_loss'].mean()} kl_div: {test_metrics_array['kl_loss'].mean()}")

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
                "vae/test_kl": test_metrics_array["kl_loss"].mean()}
                )

    return autoencoder


if __name__ == "__main__":
    parser = setup_parser(ArgumentParser())
    args  = parser.parse_args()
    
    config = load_config()
    config["device"] = args.device
    
    input_dim = config["num_joints"]
    if config["dataset"] == "action":
        train_dataloader, val_dataloader, test_dataloader = load_action_dataset(config)
        
    elif config["dataset"] == "action_target_v1":
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset(config)
        
    elif config["dataset"] == "action_target_v2":
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset(config)
        
    elif config["dataset"] == "conditional_action_target":
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset(config)

    elif config["dataset"] == "conditional_target":
        train_dataloader, val_dataloader, test_dataloader = load_target_dataset(config)

    loss_config = config["loss_func"]
    loss_config["target_mode"] = train_dataloader.dataset.y_mode
    loss_func = get_loss_func(loss_config, args.device)

    if args.print_config:
        print(json.dumps(config, sort_keys=True, indent=4))
        exit()
    
    path = f"results/vae/{args.subdir}/{config['num_joints']}_{config['latent_dim']}_{int(time.time())}"
    logger = SummaryWriter(path)

    autoencoder = build_model(
        config,
        train_dataloader.dataset.input_dim,
        train_dataloader.dataset.conditional_dim,
        logger)
    
    if args.print_model:
        print(autoencoder)
        exit()
    
    
    # store config
    store_config(config, path)
        
    train(
        autoencoder,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        loss_func=loss_func,
        logger=logger,
        epochs=config["epochs"],
        device=args.device,
        learning_rate=config["learning_rate"],
        val_interval=5,
        path=path)
