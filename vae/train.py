import time
import yaml
import numpy as np
import torch

from progress.bar import Bar
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vae.data.data_set import ActionTargetDatasetV2, ConditionalActionTargetDataset
from vae.data.load_data_set import load_action_dataset, load_action_target_dataset
from vae.helper.extract_angles_and_position import extract_angles_and_position
from vae.helper.loss import CyclicVAELoss, VAELoss
from vae.model.vae import VariationalAutoencoder


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
    device: str
    ):
    
    # prepare logging
    reconstruction_loss_array = []
    kl_loss_array = []
    total_loss_array = []

    for x, y in data:
        x = x.to(device)
        
        x_hat, mu, log_std = autoencoder(x, y)  # out shape: (batch_size, number of joints) 
        
        if type(data.dataset) ==  ConditionalActionTargetDataset or type(data.dataset) == ActionTargetDatasetV2:
            # extract angles and position
            x_angles, _ = extract_angles_and_position(x)
        else:
            x_angles = x


        loss = loss_func(x_angles, x_hat, mu, log_std)
        
        reconstruction_loss_array.append(loss_func.r_loss.item())
        kl_loss_array.append(loss_func.kl_div.item())
        total_loss_array.append(loss.item())

    # convert to numpy array
    reconstruction_loss_array = np.array(reconstruction_loss_array)
    kl_loss_array = np.array(kl_loss_array)
    total_loss_array = np.array(total_loss_array)

    return reconstruction_loss_array, kl_loss_array, total_loss_array


def train(
    autoencoder: VariationalAutoencoder, 
    train_data: DataLoader,
    val_data: DataLoader,
    test_data: DataLoader,
    loss_func: CyclicVAELoss,
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

        for x, y in train_data:  
            # in case of dataset == ActionTargetDataset: x is the action and y is the corresponding target position 
            # in case of dataset == ActionDataset: x is the action and y is an empty tensor
            x = x.to(device) # GPU or CPU
           
            x_hat, mu, log_std = autoencoder(x, y)  # out shape: (batch_size, number of joints) 
            std_array = np.concatenate([std_array, log_std.detach().numpy().flatten()])
            # mu_array = np.concatenate([mu_array, mu.detach().numpy().flatten()])
            
            if type(train_data.dataset) ==  ConditionalActionTargetDataset or type(train_data.dataset) == ActionTargetDatasetV2:
                # extract angles and position
                x_angles, _ = extract_angles_and_position(x)
            else:
                x_angles = x

            # print(x_angles.size(), x_hat.size())
            loss = loss_func(x_angles, x_hat, mu, log_std)
        
            train_reconstruction_loss_array.append(loss_func.r_loss.item())
            train_kl_loss_array.append(loss_func.kl_div.item())
        
            autoencoder.train(loss)

        # val loop
        if epoch_idx % val_interval == 0:
            val_reconstruction_loss_array, val_kl_loss_array, val_total_loss_array = run_model(
                autoencoder,
                val_data,
                loss_func,
                device
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
            
            autoencoder.log_parameters(epoch_idx)
            autoencoder.log_gradients(epoch_idx)
            autoencoder.log_decoder_distr(epoch_idx)
            autoencoder.log_z_grad(epoch_idx)
            
            logger.add_scalar("vae/val_r_loss", val_reconstruction_loss_array.mean(), epoch_idx)
            logger.add_scalar("vae/val_kl", val_kl_loss_array.mean(), epoch_idx)
            logger.add_scalar("vae/val_loss", val_total_loss_array.mean(), epoch_idx)

    # test loop
    test_reconstruction_loss_array, test_kl_loss_array, test_total_loss_array = run_model(
        autoencoder,
        test_data,
        loss_func,
        device
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
    config = load_config()
    
    input_dim = config["num_joints"]
    if config["dataset"] == "action":
        enhance_dim = 0
        input_dim = config["num_joints"]
        
        train_dataloader, val_dataloader, test_dataloader = load_action_dataset(config)
        
    elif config["dataset"] == "action_target_v1":
        enhance_dim = 2
        input_dim = config["num_joints"]
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset(config)
        
    elif config["dataset"] == "action_target_v2":
        enhance_dim = 0
        input_dim = config["num_joints"] + 2
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset(config)
        
    elif config["dataset"] == "conditional_action_target":
        enhance_dim = 2  # for the additional target position
        input_dim = config["num_joints"] + 2
        
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset(config)
        
    path = f"results/vae/{config['num_joints']}_{config['latent_dim']}_{int(time.time())}"
    logger = SummaryWriter(path)

    # store config
    store_config(config, path)
    
    print("architecture")
    print(f"{input_dim} -> [Encoder] -> {config['latent_dim']} + {enhance_dim} -> [Decoder] -> {config['num_joints']}")
    
    autoencoder = VariationalAutoencoder(
        input_dim=input_dim,
        latent_dim=config["latent_dim"],
        output_dim=config["num_joints"],
        learning_rate=config["learning_rate"],
        logger=logger,
        conditional_info_dim=enhance_dim,
        store_history=True)
    
    if config["dataset_mode"] in ["relative_uniform", "relative_tanh"]:
        loss_func = VAELoss(
            kl_loss_weight=config["kl_loss_weight"],
            reconstruction_loss_weight=config["kl_loss_weight"]
        )
    else:
        loss_func = CyclicVAELoss(
            kl_loss_weight=config["kl_loss_weight"],
            reconstruction_loss_weight=config["reconstruction_loss_weight"],
            normalization=config["normalize"]
        )
        
    train(
        autoencoder,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        loss_func=loss_func,
        logger=logger,
        epochs=config["epochs"],
        device=config["device"],
        learning_rate=config["learning_rate"],
        val_interval=5,
        path=path)
