from typing import Tuple
from torch.utils.data import DataLoader

from vae.data.data_set import ActionDataset, ActionTargetDatasetV1, ActionTargetDatasetV2, ConditionalActionTargetDataset


def load_action_dataset(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    print(f"use: {ActionDataset.__name__}")
    train_data = ActionDataset(
        f"./datasets/{config['num_joints']}/train/actions_{config['dataset_mode']}.csv",
        config["normalize"]
        )
    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]) 
    val_data = ActionDataset(
        f"./datasets/{config['num_joints']}/val/actions_{config['dataset_mode']}.csv",
        config["normalize"]
        )
    val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"])
    test_data = ActionDataset(
        f"./datasets/{config['num_joints']}/test/actions_{config['dataset_mode']}.csv",
        config["normalize"]
        )
    test_dataloader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"])

    return train_dataloader, val_dataloader, test_dataloader


def load_action_target_dataset(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if config["dataset"] == "action_target_v1":
        print(f"use: {ActionTargetDatasetV1.__name__}")
        train_dataloader, val_dataloader, test_dataloader = load_specified_action_target_dataset(config, ActionTargetDatasetV1)
    elif config["dataset"] == "action_target_v2":
        print(f"use: {ActionTargetDatasetV2.__name__}")
        train_dataloader, val_dataloader, test_dataloader = load_specified_action_target_dataset(config, ActionTargetDatasetV2)
    elif config["dataset"] == "conditional_action_target":
        print(f"use: {ConditionalActionTargetDataset.__name__}")
        train_dataloader, val_dataloader, test_dataloader = load_specified_action_target_dataset(config, ConditionalActionTargetDataset)

    return train_dataloader, val_dataloader, test_dataloader


def load_specified_action_target_dataset(config, DatasetEntity) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_data = DatasetEntity(
        f"./datasets/{config['num_joints']}/train/actions_{config['dataset_mode']}.csv",
        f"./datasets/{config['num_joints']}/train/targets_{config['dataset_mode']}.csv",
        config["normalize"] 
        )
    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]) 
    val_data = DatasetEntity(
        f"./datasets/{config['num_joints']}/val/actions_{config['dataset_mode']}.csv",
        f"./datasets/{config['num_joints']}/val/targets_{config['dataset_mode']}.csv",
        config["normalize"] 
        )
    val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"])
    test_data = DatasetEntity(
        f"./datasets/{config['num_joints']}/test/actions_{config['dataset_mode']}.csv",
        f"./datasets/{config['num_joints']}/test/targets_{config['dataset_mode']}.csv",
        config["normalize"]
        )
    test_dataloader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"])
    
    return train_dataloader, val_dataloader, test_dataloader
