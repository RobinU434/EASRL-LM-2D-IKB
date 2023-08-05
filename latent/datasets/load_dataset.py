from typing import Type, Union

from latent.datasets.vae_dataset import (
    VAEActionDataset,
    VAEStateActionDataset,
    VAETargetGaussianDataset,
)
from latent.datasets.supervised_dataset import (
    SupervisedStateActionDataset,
    SupervisedTargetGaussianDataset,
)
from latent.datasets.utils import TargetMode

from torch.utils.data import DataLoader


def load_data(
    type: Union[
        Type[VAEActionDataset],
        Type[VAEStateActionDataset],
        Type[VAETargetGaussianDataset],
        Type[SupervisedStateActionDataset],
        Type[SupervisedTargetGaussianDataset],
    ],
    n_joints: int,
    data_entity: str, # [train, val, test]
    **data_args
):
    """loads dataset and packs it into dataloader

    Args:
        type (Union[ Type[VAEActionDataset], Type[VAEStateActionDataset], Type[VAETargetGaussianDataset], Type[SupervisedStateActionDataset], Type[SupervisedTargetGaussianDataset], ]): _description_
        n_joints (int): _description_
        data_entity (str): _description_

    Returns:
        _type_: _description_
    """
    data_args["target_mode"] = getattr(TargetMode, data_args["target_mode"])
    action_file = f"data/{n_joints}/{data_entity}/actions_{data_args['mode']}.csv"
    state_file = f"data/{n_joints}/{data_entity}/states_{data_args['mode']}.csv"
    dataset = type.from_files(action_file=action_file, state_file=state_file, **data_args)

    batch_size = data_args["batch_size"]
    shuffle = data_args["shuffle"]

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader