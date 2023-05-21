import logging
from typing import Tuple, Union
from torch.utils.data import DataLoader

from vae.data.data_set import ActionDataset, ActionTargetDatasetV1, ActionTargetDatasetV2, ConditionalActionTargetDataset, ConditionalTargetDataset


def load_action_dataset(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    print(f"use: {ActionDataset.__name__}")
    train_data = ActionDataset(
        f"./datasets/{config['num_joints']}/train/actions_{config['dataset_mode']}.csv",
        )
    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]) 
    val_data = ActionDataset(
        f"./datasets/{config['num_joints']}/val/actions_{config['dataset_mode']}.csv",
        )
    val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"])
    test_data = ActionDataset(
        f"./datasets/{config['num_joints']}/test/actions_{config['dataset_mode']}.csv",
        )
    test_dataloader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"])

    return train_dataloader, val_dataloader, test_dataloader


def load_action_target_dataset(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if config["dataset"] == "action_target_v1":
        print(f"use: {ActionTargetDatasetV1.__name__}")
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset_by_entity(config, ActionTargetDatasetV1)
    elif config["dataset"] == "action_target_v2":
        print(f"use: {ActionTargetDatasetV2.__name__}")
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset_by_entity(config, ActionTargetDatasetV2)
    elif config["dataset"] == "conditional_action_target":
        print(f"use: {ConditionalActionTargetDataset.__name__}")
        train_dataloader, val_dataloader, test_dataloader = load_action_target_dataset_by_entity(config, ConditionalActionTargetDataset)

    return train_dataloader, val_dataloader, test_dataloader


def load_action_target_dataset_by_entity(
        config: dict, 
        DatasetEntity: Union[ActionTargetDatasetV1, ActionTargetDatasetV2, ConditionalActionTargetDataset]
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    action_constrain_radius = config["action_constrain_radius"] if config["action_constrain_radius"] != 0 else None
    logging.info("loading training dataset")
    train_data = DatasetEntity(
        f"./datasets/{config['num_joints']}/train/actions_{config['dataset_mode']}.csv",
        f"./datasets/{config['num_joints']}/train/{config['conditional_info']}_{config['dataset_mode']}.csv",
        action_constrain_radius
        )
    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]) 
    logging.info("done")
    logging.info("loading validation dataset")
    val_data = DatasetEntity(
        f"./datasets/{config['num_joints']}/val/actions_{config['dataset_mode']}.csv",
        f"./datasets/{config['num_joints']}/val/{config['conditional_info']}_{config['dataset_mode']}.csv",
        action_constrain_radius
        )
    val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"])
    logging.info("done")
    logging.info("loading test dataset")
    test_data = DatasetEntity(
        f"./datasets/{config['num_joints']}/test/actions_{config['dataset_mode']}.csv",
        f"./datasets/{config['num_joints']}/test/{config['conditional_info']}_{config['dataset_mode']}.csv",
        action_constrain_radius
        )
    test_dataloader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"])
    logging.info("done")
    return train_dataloader, val_dataloader, test_dataloader


def load_target_dataset(config: dict):
    print(f"use: {ConditionalTargetDataset.__name__}")
    if config["conditional_info"] != "state":
        raise ValueError(f"You can only chose the {ConditionalTargetDataset.__name__} if you select 'state' as the conditional info, but you chose {config['conditional_info']}")
    train_data = ConditionalTargetDataset(
        f"./datasets/{config['num_joints']}/train/{config['conditional_info']}_{config['dataset_mode']}.csv",
        std=config["action_constrain_radius"]
        )
    train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]) 
    val_data = ConditionalTargetDataset(
        f"./datasets/{config['num_joints']}/val/{config['conditional_info']}_{config['dataset_mode']}.csv",
        std=config["action_constrain_radius"]
        )
    val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"])
    test_data = ConditionalTargetDataset(
        f"./datasets/{config['num_joints']}/test/{config['conditional_info']}_{config['dataset_mode']}.csv",
        std=config["action_constrain_radius"]
        )
    test_dataloader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"])

    return train_dataloader, val_dataloader, test_dataloader

