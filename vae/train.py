from argparse import ArgumentParser
import time
import yaml
import numpy as np
import torch

from progress.bar import Bar
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vae.data.data_set import ActionTargetDatasetV2, ConditionalActionTargetDataset
from vae.data.load_data_set import load_action_dataset, load_action_target_dataset
from vae.helper.extract_angles_and_position import split_conditional_info, split_state_information
from vae.helper.loss import DistanceLoss, ImitationLoss, VAELoss, get_loss_func, DistVAELoss
from vae.model.vae import VariationalAutoencoder


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
    return parser


def load_config():
    with open("config/base_vae.yaml") as f:
        config = yaml.safe_load(f)

    return config


def store_config(config, path: str):
    with open(path + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file)


def run_model(
    autoencoder: VariationalAutoencoder,
    data: DataLoader,
    loss_func: VAELoss,
    device: str,
    train: bool = False,
    ):

    imitation_loss_func = ImitationLoss()
    distance_loss_func = DistanceLoss()

    # prepare logging
    log_metrics_array = []
    log_metrics_dt = [
        ("reconstruction_loss", np.float32),
        ("kl_loss", np.float32),
        ("total_loss", np.float32),
        ("std", np.float32),
        ("imitation_loss", np.float32),
        ("distance_loss", np.float32)]
    for target_action, conditional_info in data:
        # in case of dataset == ActionTargetDataset: x is the action and y is the corresponding target position 
        # in case of dataset == ActionDataset: x is the action and y is an empty tensor
        target_action = target_action.to(device)
        conditional_info = conditional_info.to(device)
        
        x_hat, mu, log_std = autoencoder(target_action, conditional_info)  # out shape: (batch_size, number of joints) 
    
        # setup loss functions
        _, _, state_angles = split_state_information(conditional_info)
        target_action = target_action + state_angles
        predicted_target_action = x_hat + state_angles
        loss = loss_func(target_action, predicted_target_action, mu, log_std)
        
        if train:
            autoencoder.train(loss)

        # log metrics in the correct order
        log_metrics_array.append(
            np.array([
                loss_func.r_loss.cpu().item(),  # reconstruction_loss
                loss_func.kl_div.cpu().item(),  # kl loss
                loss.cpu().item(),  # total loss
                log_std.mean().cpu().item(),  # std
                imitation_loss_func(target_action, predicted_target_action).item(),  # imitation_loss
                distance_loss_func(target_action, predicted_target_action).item(),  # distance_loss
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
                epoch_idx=epoch_idx
                )
            
        # LOGGING
        if logger is not None and epoch_idx % val_interval == 0:    
            print(f"epoch {epoch_idx}: \n\
                train: loss: {train_metrics_array['reconstruction_loss'].mean()} kl_div: {train_metrics_array['kl_loss'].mean()} \n\
                val: loss: {val_metrics_array['reconstruction_loss'].mean()} kl_div: {val_metrics_array['kl_loss'].mean()}")
            
            logger.add_scalar("vae/train_r_loss", train_metrics_array["reconstruction_loss"].mean(), epoch_idx)
            logger.add_scalar("vae/train_kl", train_metrics_array["kl_loss"].mean(), epoch_idx)
            logger.add_scalar("vae/train_std", train_metrics_array["std"].mean(), epoch_idx)
            logger.add_scalar("vae/train_imitation_loss", train_metrics_array["imitation_loss"].mean(), epoch_idx)
            logger.add_scalar("vae/train_distance_loss", train_metrics_array["distance_loss"].mean(), epoch_idx)
            # logger.add_histogram("vae/train_latent_mu", mu_array, epoch_idx)
            
            # autoencoder.log_parameters(epoch_idx)
            # autoencoder.log_gradients(epoch_idx)
            autoencoder.log_decoder_distr(epoch_idx)
            autoencoder.log_z_grad(epoch_idx)
            
            logger.add_scalar("vae/val_r_loss", val_metrics_array["reconstruction_loss"].mean(), epoch_idx)
            logger.add_scalar("vae/val_kl", val_metrics_array["kl_loss"].mean(), epoch_idx)
            logger.add_scalar("vae/val_loss", val_metrics_array["total_loss"].mean(), epoch_idx)
            logger.add_scalar("vae/val_imitation_loss", val_metrics_array["imitation_loss"].mean(), epoch_idx)
            logger.add_scalar("vae/val_distance_loss", val_metrics_array["distance_loss"].mean(), epoch_idx)
    # test loop
    test_metrics_array = run_model(
        autoencoder,
        test_data,
        loss_func,
        device,
        train=False
    )
    logger.add_scalar("vae/test_r_loss", test_metrics_array["reconstruction_loss"].mean(), epoch_idx)
    logger.add_scalar("vae/test_kl", test_metrics_array["kl_loss"].mean(), epoch_idx)
    logger.add_scalar("vae/test_loss", test_metrics_array["total_loss"].mean(), epoch_idx)
    logger.add_scalar("vae/test_imitation_loss", test_metrics_array["imitation_loss"].mean(), epoch_idx)
    logger.add_scalar("vae/test_distance_loss", test_metrics_array["distance_loss"].mean(), epoch_idx)  

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
        conditional_info_dim = 0
        input_dim = config["num_joints"]
        
        train_dataloader, val_dataloader, test_dataloader = load_action_dataset(config)
        
    elif config["dataset"] == "action_target_v1":
        conditional_info_dim = 4 + config["num_joints"]
        input_dim = config["num_joints"]
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset(config)
        
    elif config["dataset"] == "action_target_v2":
        conditional_info_dim = 0
        input_dim = config["num_joints"] + (config["num_joints"] + 4)   # in brakets is conditional information
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset(config)
        
    elif config["dataset"] == "conditional_action_target":
        conditional_info_dim =  config["num_joints"] + 4  # for the additional state information
        input_dim = config["num_joints"] + (config["num_joints"] + 4)  # in brakets is conditional information
        
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset(config)
        
    path = f"results/vae/{args.subdir}/{config['num_joints']}_{config['latent_dim']}_{int(time.time())}"
    logger = SummaryWriter(path)

    # store config
    store_config(config, path)

    print("kl loss weight: ", config["kl_loss_weight"])
    print("reconstruction loss weight: ", config["reconstruction_loss_weight"])
    print("")
    
    print("architecture")
    print(f"{input_dim} -> [Encoder] -> {config['latent_dim']} + {conditional_info_dim} -> [Decoder] -> {config['num_joints']}")
    
    autoencoder = VariationalAutoencoder(
        input_dim=input_dim,
        latent_dim=config["latent_dim"],
        output_dim=config["num_joints"],
        learning_rate=config["learning_rate"],
        logger=logger,
        conditional_info_dim=conditional_info_dim,
        store_history=True, 
        device=args.device).to(args.device)
    
    loss_func = get_loss_func(config)
    # print("use: ", loss_func.__name__)
        
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
