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
from vae.helper.loss import VAELoss, get_loss_func, DistVAELoss
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
    # prepare logging
    reconstruction_loss_array = np.array([])
    kl_loss_array = np.array([])
    total_loss_array = np.array([])
    std_array = np.array([])
    for target_action, conditional_info in data:
        # in case of dataset == ActionTargetDataset: x is the action and y is the corresponding target position 
        # in case of dataset == ActionDataset: x is the action and y is an empty tensor
            
        target_action = target_action.to(device)
        conditional_info = conditional_info.to(device)
        
        x_hat, mu, log_std = autoencoder(target_action, conditional_info)  # out shape: (batch_size, number of joints) 
        std_array = np.concatenate([std_array, log_std.cpu().detach().numpy().flatten()])
            
        # extract angles and position
        # if type(data.dataset) ==  ConditionalActionTargetDataset or type(data.dataset) == ActionTargetDatasetV2:
        #     x_angles, target_pos, _, state_angles = split_conditional_info(target_action)
        # else:
        #     x_angles = target_action

        # setup loss functions
        if type(loss_func) == DistVAELoss:
            target_pos, _, state_angles = split_state_information(conditional_info)
            x_hat_angles = state_angles + (x_hat + loss_func.normalization) * loss_func.normalization * torch.pi
            loss = loss_func(target_pos, x_hat_angles, mu, log_std)
        else:
            # x_angles = torch.ones_like(x_angles)
            # x_angles = torch.ones_like(x_angles) * (2 - x_angles.sum(1) > 0)
            loss = loss_func(target_action, x_hat, mu, log_std)
        
        if train:
            autoencoder.train(loss)

        reconstruction_loss_array = np.concatenate([reconstruction_loss_array, np.array([loss_func.r_loss.cpu().item()])])
        kl_loss_array = np.concatenate([kl_loss_array, np.array([loss_func.kl_div.cpu().item()])])
        total_loss_array = np.concatenate([total_loss_array, np.array([loss.cpu().item()])])

    return reconstruction_loss_array, kl_loss_array, total_loss_array, std_array


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
        train_reconstruction_loss_array = []
        train_kl_loss_array = []
        std_array = np.array([])
        # mu_array = np.array([])
        # reset logging history per epoch
        autoencoder.reset_history()

        # train loop
        train_reconstruction_loss_array, train_kl_loss_array, train_total_loss_array, std_array = run_model(
                autoencoder,
                train_data,
                loss_func,
                device,
                train=True
            )
        
        # val loop
        if epoch_idx % val_interval == 0:
            val_reconstruction_loss_array, val_kl_loss_array, val_total_loss_array, _ = run_model(
                autoencoder,
                val_data,
                loss_func,
                device,
                train=False
            )

            # store model checkpoint
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': autoencoder.optimizer.state_dict(),
                'loss': val_total_loss_array.mean(),
            }, path + f"/model_{epoch_idx}_val_r_loss_{float(val_reconstruction_loss_array.mean()):.4f}.pt")     

        # LOGGING
        if logger is not None and epoch_idx % val_interval == 0:    
            train_reconstruction_loss_array = np.array(train_reconstruction_loss_array)
            train_kl_loss_array = np.array(train_kl_loss_array)
            
            print(f"epoch {epoch_idx}: \n\
                train: loss: {train_reconstruction_loss_array.mean()} kl_div: {train_kl_loss_array.mean()} \n\
                val: loss: {val_reconstruction_loss_array.mean()} kl_div: {val_kl_loss_array.mean()}")

            logger.add_scalar("vae/train_r_loss", train_reconstruction_loss_array.mean(), epoch_idx)
            logger.add_scalar("vae/train_kl", train_kl_loss_array.mean(), epoch_idx)
            logger.add_scalar("vae/train_std", std_array.mean(), epoch_idx)
            # logger.add_histogram("vae/train_latent_mu", mu_array, epoch_idx)
            
            # autoencoder.log_parameters(epoch_idx)
            # autoencoder.log_gradients(epoch_idx)
            autoencoder.log_decoder_distr(epoch_idx)
            autoencoder.log_z_grad(epoch_idx)
            
            logger.add_scalar("vae/val_r_loss", val_reconstruction_loss_array.mean(), epoch_idx)
            logger.add_scalar("vae/val_kl", val_kl_loss_array.mean(), epoch_idx)
            logger.add_scalar("vae/val_loss", val_total_loss_array.mean(), epoch_idx)

    # test loop
    test_reconstruction_loss_array, test_kl_loss_array, test_total_loss_array, _ = run_model(
        autoencoder,
        test_data,
        loss_func,
        device,
        train=False
    )

    logger.add_scalar("vae/test_r_loss", test_reconstruction_loss_array.mean(), epoch_idx)
    logger.add_scalar("vae/test_kl", test_kl_loss_array.mean(), epoch_idx)
    logger.add_scalar("vae/test_loss", test_total_loss_array.mean(), epoch_idx)

    print(f"test: loss: {test_reconstruction_loss_array.mean()} kl_div: {test_kl_loss_array.mean()}")

    if logger is not None:
        logger.add_hparams(
            hparam_dict={
                "learning_rate": learning_rate,
                "val_interval": val_interval,
                "kl_loss_weight": config["kl_loss_weight"],
                "reconstruction_loss_weight": config["reconstruction_loss_weight"],
                "dataset": config["dataset"],
                "normalize": config["normalize"]},
            metric_dict={
                "vae/test_loss": test_total_loss_array.mean(), 
                "vae/test_r_loss": test_reconstruction_loss_array.mean(),
                "vae/test_kl": test_kl_loss_array.mean()}
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
