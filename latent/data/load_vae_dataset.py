import logging
from typing import Any, Dict, Tuple, Type, Union

from torch.utils.data import DataLoader

from latent.data.vae_dataset import VAEDataset
from latent.data.vae_dataset import (
    ActionDataset,
    ActionTargetDatasetV1,
    ActionTargetDatasetV2,
    ConditionalActionTargetDataset,
    TargetGaussianDataset,
)


def load_action_dataset(
    config: Dict[str, Any]
) -> Tuple[DataLoader[VAEDataset], DataLoader[VAEDataset], DataLoader[VAEDataset]]:
    """loads data from file system and fills it into VAEDataset classes to return train val and test data

    Args:
        config (Dict[str, Any]): config dictionary should contain at least:
            - num_joints
            - dataset_mode
            - shuffle_data
            - batch_size

    Returns:
        Tuple[DataLoader[VAEDataset], DataLoader[VAEDataset], DataLoader[VAEDataset]]: dataloader of train dataset, dataloader of val dataset, dataloader of test dataset
    """
    print(f"use: {ActionDataset.__name__}")
    train_data = ActionDataset(
        f"./datasets/{config['num_joints']}/train/actions_{config['dataset_mode']}.csv",
    )
    train_dataloader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )
    val_data = ActionDataset(
        f"./datasets/{config['num_joints']}/val/actions_{config['dataset_mode']}.csv",
    )
    val_dataloader = DataLoader(
        val_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )
    test_data = ActionDataset(
        f"./datasets/{config['num_joints']}/test/actions_{config['dataset_mode']}.csv",
    )
    test_dataloader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )

    return train_dataloader, val_dataloader, test_dataloader


def load_action_target_dataset(
    config: Dict[str, Any]
) -> Tuple[DataLoader[VAEDataset], DataLoader[VAEDataset], DataLoader[VAEDataset]]:
    """calls load_action_target_dataset_by_entity with entity in ActionTargetDatasetV1, ActionTargetDatasetV2, ConditionalActionTargetDataset

    Args:
        config (Dict[str, Any]): config dictionary should contain at least:
            - num_joints
            - dataset_mode
            - shuffle_data
            - batch_size
            - action_constrain_radius
    Raises:
        ValueError: if the configured dataset in config is not in ['action_target_v1', 'action_target_v2', 'conditional_action_target']

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: dataloader of train dataset, dataloader of val dataset, dataloader of test dataset
    """
    if config["dataset"] == "action_target_v1":
        print(f"use: {ActionTargetDatasetV1.__name__}")
        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ) = load_action_target_dataset_by_entity(config, ActionTargetDatasetV1)
    elif config["dataset"] == "action_target_v2":
        print(f"use: {ActionTargetDatasetV2.__name__}")
        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ) = load_action_target_dataset_by_entity(config, ActionTargetDatasetV2)
    elif config["dataset"] == "conditional_action_target":
        print(f"use: {ConditionalActionTargetDataset.__name__}")
        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ) = load_action_target_dataset_by_entity(config, ConditionalActionTargetDataset)
    else:
        raise ValueError(
            "config['dataset] has to be in ['action_target_v1', 'action_target_v2', 'conditional_action_target'] but value was: ",
            config["dataset"],
        )

    return train_dataloader, val_dataloader, test_dataloader


def load_action_target_dataset_by_entity(
    config: Dict[str, Any],
    dataset_entity: Union[
        Type[ActionTargetDatasetV1],
        Type[ActionTargetDatasetV2],
        Type[ConditionalActionTargetDataset],
    ],
) -> Tuple[DataLoader[VAEDataset], DataLoader[VAEDataset], DataLoader[VAEDataset]]:
    """loads either a ActionTargetDatasetV1, ActionTargetDatasetV2, ConditionalActionTargetDataset decided by dataset_entity.

    Args:
        config (Dict[str, Any]): config dictionary should contain at least:
            - num_joints
            - dataset_mode
            - shuffle_data
            - batch_size
            - action_constrain_radius
        dataset_entity (Union[Type[ActionTargetDatasetV1], Type[ActionTargetDatasetV2], Type[ConditionalActionTargetDataset]]): what kind of dataset to load

    Returns:
        Tuple[DataLoader[VAEDataset], DataLoader[VAEDataset], DataLoader[VAEDataset]]: dataloader of train dataset, dataloader of val dataset, dataloader of test dataset
    """
    action_constrain_radius = (
        config["action_constrain_radius"]
        if config["action_constrain_radius"] != 0
        else None
    )
    logging.info("loading training dataset")
    train_data = dataset_entity(
        f"./datasets/{config['num_joints']}/train/actions_{config['dataset_mode']}.csv",
        f"./datasets/{config['num_joints']}/train/{config['conditional_info']}_{config['dataset_mode']}.csv",
        action_constrain_radius,
    )
    train_dataloader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )
    logging.info("done")
    logging.info("loading validation dataset")
    val_data = dataset_entity(
        f"./datasets/{config['num_joints']}/val/actions_{config['dataset_mode']}.csv",
        f"./datasets/{config['num_joints']}/val/{config['conditional_info']}_{config['dataset_mode']}.csv",
        action_constrain_radius,
    )
    val_dataloader = DataLoader(
        val_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )
    logging.info("done")
    logging.info("loading test dataset")
    test_data = dataset_entity(
        f"./datasets/{config['num_joints']}/test/actions_{config['dataset_mode']}.csv",
        f"./datasets/{config['num_joints']}/test/{config['conditional_info']}_{config['dataset_mode']}.csv",
        action_constrain_radius,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )
    logging.info("done")
    return train_dataloader, val_dataloader, test_dataloader


def load_gaussian_target_dataset(
    config: Dict[str, Any]
) -> Tuple[DataLoader[VAEDataset], DataLoader[VAEDataset], DataLoader[VAEDataset],]:
    print(f"use: {TargetGaussianDataset.__name__}")

    if config["conditional_info"] != "state":
        raise ValueError(
            f"You can only chose the {TargetGaussianDataset.__name__} if you select 'state' as the conditional info, but you chose {config['conditional_info']}"
        )

    train_data = TargetGaussianDataset(
        f"./datasets/{config['num_joints']}/train/{config['conditional_info']}_{config['dataset_mode']}.csv",
        std=config["action_constrain_radius"],
    )
    train_dataloader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )
    val_data = TargetGaussianDataset(
        f"./datasets/{config['num_joints']}/val/{config['conditional_info']}_{config['dataset_mode']}.csv",
        std=config["action_constrain_radius"],
    )
    val_dataloader = DataLoader(
        val_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )
    test_data = TargetGaussianDataset(
        f"./datasets/{config['num_joints']}/test/{config['conditional_info']}_{config['dataset_mode']}.csv",
        std=config["action_constrain_radius"],
    )
    test_dataloader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )

    return train_dataloader, val_dataloader, test_dataloader


def load_target_dataset(
    config: dict,
) -> Tuple[DataLoader[VAEDataset], DataLoader[VAEDataset], DataLoader[VAEDataset]]:
    print(f"use: {TargetGaussianDataset.__name__}")

    if config["conditional_info"] != "state":
        raise ValueError(
            f"You can only chose the {TargetGaussianDataset.__name__} if you select 'state' as the conditional info, but you chose {config['conditional_info']}"
        )

    train_data = TargetGaussianDataset(
        f"./datasets/{config['num_joints']}/train/{config['conditional_info']}_{config['dataset_mode']}.csv",
        std=config["action_constrain_radius"],
    )
    train_dataloader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )
    val_data = TargetGaussianDataset(
        f"./datasets/{config['num_joints']}/val/{config['conditional_info']}_{config['dataset_mode']}.csv",
        std=config["action_constrain_radius"],
    )
    val_dataloader = DataLoader(
        val_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )
    test_data = TargetGaussianDataset(
        f"./datasets/{config['num_joints']}/test/{config['conditional_info']}_{config['dataset_mode']}.csv",
        std=config["action_constrain_radius"],
    )
    test_dataloader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=config["shuffle_data"]
    )

    return train_dataloader, val_dataloader, test_dataloader
