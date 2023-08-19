import json
import logging
from typing import Tuple, Union
import yaml
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard.writer import SummaryWriter

from supervised.data import TargetGaussianDataset, get_datasets
from supervised.loss import (
    EuclidianDistance,
    ImitationLoss,
    PointDistanceLoss,
    get_loss_func,
    IKLoss,
)
from supervised.model import Regressor, build_model
from supervised.utils import (
    forward_kinematics,
    split_state_information,
    profile_runtime,
    profile_memory
)
from vae.utils.post_processing import PostProcessor
from vae.data.data_set import YMode
from envs.common.sample_target import sample_target

from progress.bar import Bar


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
        default="cpu",
        help=f"GPU or CPU, current available GPU index: {torch.cuda.device_count() - 1}",
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
    return parser


def load_config() -> dict:
    with open("config/base_supervised.yaml") as f:
        config = yaml.safe_load(f)

    return config


def log(logger: SummaryWriter, metrics: np.ndarray, epoch_idx: int, entity: str):
    """logs loss, imitation_loss, distance_loss

    Args:
        logger (SummaryWriter): tensorboard logging object
        metrics (np.ndarray): structured array with must have keys ['loss', 'distance_loss', 'imitation_loss']
        epoch_idx (int): to which epoch you want to add the logging information
        entity (str): is either train, val or test
    """
    logger.add_scalar(f"supervised/{entity}_loss", metrics["loss"].mean(), epoch_idx)
    logger.add_scalar(
        f"supervised/{entity}_imiation_loss",
        metrics["imitation_loss"].mean(),
        epoch_idx,
    )
    logger.add_scalar(
        f"supervised/{entity}_distance_loss",
        metrics["distance_loss"].mean(),
        epoch_idx,
    )

@profile_runtime
def run_model(
    model: Regressor,
    data: DataLoader,
    criterion: IKLoss,
    train: bool = False,
    device: str = "cpu",
):
    metrics = []
    metrics_dt = [
        ("loss", np.float32),
        ("imitation_loss", np.float32),
        ("distance_loss", np.float32),
    ]

    for x, y in data:
        x = x.to(device)
        y = y.to(device)

        x_hat = model.forward(x)

        _, _, state_angles = split_state_information(x)
        loss = criterion(y, state_angles + x_hat)

        if train:
            model.train(loss)

        metrics.append(
            np.array(
                [
                    loss.item(),
                    criterion.imitation_loss.item(),
                    criterion.distance_loss.item(),
                ]
            )
        )

    metrics = np.stack(metrics)
    metrics = np.rec.fromarrays(metrics.T, dtype=metrics_dt)

    return metrics


def train(
    model: Regressor,
    train_data: DataLoader,
    val_data: DataLoader,
    loss_func: IKLoss,
    n_epochs: int,
    logger: SummaryWriter,
    val_interval: int,
    device: str,
    path: str,
) -> None:
    for epoch_idx in range(n_epochs):
        train_metrics = run_model(
            model=model,
            data=train_data,
            criterion=loss_func,
            train=True,
            device=device,
        )

        if epoch_idx % val_interval == 0:
            val_metrics = run_model(
                model=model,
                data=val_data,
                criterion=loss_func,
                train=False,
                device=device,
            )

            print(
                f"epoch: {epoch_idx}  train_loss: {train_metrics['loss'].mean()} val_loss: {val_metrics['loss'].mean()}"
            )

            log(
                logger=logger,
                metrics=train_metrics,
                epoch_idx=epoch_idx,
                entity="train",
            )
            log(logger=logger, metrics=val_metrics, epoch_idx=epoch_idx, entity="val")

            # save model
            model.save(
                path=path
                + f"/model_{epoch_idx}_val_loss_{float(val_metrics['loss'].mean()):.4f}.pt",
                epoch_idx=epoch_idx,
                metrics=val_metrics,
            )


def get_direction(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """returns a direction vector from x to y normalized on length = 1

    Args:
        x (Union[torch.Tensor, np.ndarray]): first position: shape (..., 2)
        y (Union[torch.Tensor, np.ndarray]): second position: shape (..., 2)

    Returns:
        Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        first value is the normalized direction, second is the original length of the direction
    """
    direction = y - x
    # normalize
    length = torch.linalg.norm(direction, axis=-1)
    denominator = torch.where(length != 0, length, torch.ones_like(length))
    return (direction.T / denominator).T, length


def gen_random_direction(
    current_pos: torch.Tensor, max_distance: float, max_step_length: float = 1, num: int = 1
) -> torch.Tensor:
    """calculate a delta vector wrt. the local frame of the current pos which points in a random direction

    Args:
        current_pos (torch.Tensor): current position of the arm end effector
        max_distance (float): maximum distance from the origin <-> maximum distance a given arm can reach
        max_step_length (float): maximum length of the direction vector. Defaults set to 1
        num (int): Amount of different directions. TODO: feature to implement. Defaults set to one
    Returns:
        torch.Tensor: a vector with random direction and random length [0, max_step_length]
    """
    theta = torch.rand(1)
    radius = torch.rand(1) * max_step_length

    delta = torch.tensor([torch.cos(theta), torch.sin(theta)]) * radius
    final = current_pos + delta
    # if you are outside the arms reach: add the direction pointing towards the middle
    if torch.linalg.norm(final) > max_distance:
        # substract vector towards origin with length = 1
        delta = delta - (current_pos / torch.linalg.norm(current_pos))
        # shorten or lengthen it back to radius length
        delta *= radius / torch.linalg.norm(delta)
    else:
        delta = delta.unsqueeze(dim=0)

    return delta


def build_state(
    target_pos: torch.Tensor,
    current_pos: torch.Tensor,
    state_angles: torch.Tensor,
    step_length: float = 1,
) -> torch.Tensor:
    # linear trajectory towards the target_pos
    direction, length = get_direction(current_pos, target_pos)
    # if condition is true -> target is closer than step_length
    scalar = torch.where(
        length < step_length, length, torch.ones_like(length) * step_length
    )
    direction = (direction.T * scalar).T
    state = torch.cat([direction, current_pos, state_angles], dim=1).float()
    return state


def run_greedy(
    model: Regressor,
    target_pos: torch.Tensor,
    init_state: torch.Tensor,
    step_length: float,
    max_steps: int,
    eps: float = 0.1,
    device: str = "cpu",
):
    _, current_pos, state_angles = split_state_information(init_state)

    metrics = []
    metrics_keys = ["distance", "trajectory", "arms"]
    step_idx = 1
    while step_idx <= max_steps:
        state = build_state(target_pos, current_pos, state_angles, step_length).to(
            device
        )
    
        with torch.no_grad():
            x_hat = model.forward(state)

        # update state
        state_angles = (state_angles + x_hat) % (2 * torch.pi)
        arm = forward_kinematics(state_angles)
        current_pos = arm[:, -1].to(device)
    
        # update metrics
        distance = torch.linalg.norm(current_pos - state[0, :2]).detach().cpu().numpy()
        distance = np.array([distance])
        metrics.append(
            (
                distance,
                current_pos.detach().cpu().squeeze().numpy(),
                arm.detach().cpu().squeeze().numpy(),
            )
        )

        if torch.linalg.norm(target_pos - current_pos) < eps:
            break
        step_idx += 1

    # pack metrics into a tuple of np.ndarray
    metrics = list(map(list, zip(*metrics)))

    # add first element of trajectory
    metrics[1].insert(0, init_state[:, 2:4].detach().cpu().squeeze().numpy())
    metrics[2].insert(
        0, forward_kinematics(init_state[:, 4:]).detach().squeeze().numpy()
    )

    metrics = list(map(np.stack, metrics))

    metrics = dict(zip(metrics_keys, metrics))

    return metrics

@profile_memory
# @profile_runtime
def greedy_validation(
    model: Regressor,
    data: DataLoader[TargetGaussianDataset],
    max_steps: int = 50,
    num_starts: int = 100,
    device: str = "cpu",
):
    logging.info(f"entered greedy validation")

    # set to final position to create trajectory to final target
    data.dataset.target_mode = YMode.FINAL_POSITION
    # get a subset of the dataset
    subset = Subset(data.dataset, range(num_starts))
    data = DataLoader(subset, batch_size=1)

    step_length = model.action_radius
    metrics = []
    metrics_dt = [("loss", np.float32), ("length", np.int32)]
    for x, y in data:
        # x is the start condition for a greedy run
        # y is the target position in the possible observation_space
        state = x.to(device)
        target = y.to(device)

        indi_metrics = run_greedy(
            model, target, state, step_length, max_steps, eps=0.1, device=device
        )

        dist_to_target = np.linalg.norm(
            indi_metrics["trajectory"] - target.cpu().numpy(), axis=1
        )

        metrics.append(np.array([dist_to_target.mean(), len(indi_metrics["distance"])]))

    metrics = np.stack(metrics)
    metrics = np.rec.fromarrays(metrics.T, dtype=metrics_dt)

    return metrics

@profile_memory
# @profile_runtime
def generate_dataset(
    model: Regressor,
    data: DataLoader[TargetGaussianDataset],
    mixing_coef: float = 0,  # 1 get everything from the  given dataset - 0 create a whole new dataset
    num_steps: int = 20,
    device: str = "cpu",
):
    min_mix = 0
    max_mix = 1
    if mixing_coef > max_mix or mixing_coef < min_mix:
        logging.warning(
            f"Mixin coeficient is out of boundaries: {mixing_coef} will be clamped in [{min_mix}, {max_mix}]"
        )
        mixing_coef = np.clip(mixing_coef, min_mix, max_mix)

    # extract previous states and exclude the index + final_target position
    states = data.dataset.states[:, 2:]

    # sample a subset according to the mixing coefficient -> keep the dataset size consistent
    old_states, new_starts = np.split(states, [round(mixing_coef * len(states))])

    new_current_pos = []
    new_state_angles = []
    bar = Bar("generate new dataset", max=len(new_starts) * num_steps)
    for x in new_starts:
        state = x.unsqueeze(dim=0)

        _, current_pos, state_angles = split_state_information(state)
        for _ in range(num_steps):
            # assign random direction
            delta = gen_random_direction(current_pos, model.action_radius)
            state = torch.cat([delta, current_pos, state_angles.cpu()], dim=1).to(device)

            with torch.no_grad():
                x_hat = model.forward(state.float())

            _, _, state_angles = split_state_information(state)

            # update state information
            state_angles = (state_angles + x_hat) % (2 * torch.pi)
            current_pos = forward_kinematics(state_angles)[:, -1]

        new_current_pos.append(current_pos.cpu())
        new_state_angles.append(state_angles.cpu())

        bar.next(num_steps)
    bar.finish()

    new_current_pos = torch.stack(new_current_pos).squeeze()
    new_state_angles = torch.stack(new_state_angles).squeeze()

    # sample all new targets
    new_targets = torch.from_numpy(sample_target(model.output_dim, len(new_starts)))

    new_states = torch.cat([new_targets, new_current_pos, new_state_angles], dim=1)

    # merge those old and new states
    states = torch.cat([old_states, new_states], dim=0)
   
    # shuffle states
    state = states[torch.randperm(len(states))]

    dataset = TargetGaussianDataset(
        states, model.action_radius, target_mode=YMode.INTERMEDIATE_POSITION
    )
    loader = DataLoader(dataset, data.batch_size)

    del data

    return loader


def active_train(
    model: Regressor,
    train_data: DataLoader[Dataset],
    val_data: DataLoader[Dataset],
    loss_func: IKLoss,
    n_epochs: int,
    logger: SummaryWriter,
    val_interval: int,
    mixing_coef: float = 1,
    num_deviation_steps: int = 1,
    path: str = "",
    device: str = "cpu",
):
    # parameters set because of observation of previous experiments
    max_steps = 50
    num_samples = 200
    print(f"max val rollout length: {max_steps}")

    for epoch_idx in range(n_epochs):
        logging.info(f"entered epoch {epoch_idx}")

        train_metrics = run_model(
            model=model, data=train_data, criterion=loss_func, train=True, device=device
        )

        if epoch_idx % val_interval == 0:
            val_metrics = run_model(
                model=model,
                data=train_data,
                criterion=loss_func,
                train=False,
                device=device,
            )

            val_greedy_metrics = greedy_validation(
                model=model,
                data=val_data,
                max_steps=max_steps,
                num_starts=num_samples,
                device=device,
            )

            print(
                f"epoch: {epoch_idx} train_loss: {train_metrics['loss'].mean():.4f} val_loss: {val_metrics['loss'].mean():.4f}, greedy: {val_greedy_metrics['loss'].mean():.4f}|{val_greedy_metrics['length'].mean():.2f}"
            )

            log(logger, train_metrics, epoch_idx, "train")
            log(logger, val_metrics, epoch_idx, "val")
            logger.add_scalar(
                "supervised/val_greedy_loss",
                val_greedy_metrics["loss"].mean(),
                epoch_idx,
            )
            logger.add_scalar(
                "supervised/val_greedy_length",
                val_greedy_metrics["length"].mean(),
                epoch_idx,
            )

            # save model
            model.save(
                path=path
                + f"/model_{epoch_idx}_val_loss_{float(val_metrics['loss'].mean()):.4f}.pt",
                epoch_idx=epoch_idx,
                metrics=val_metrics,
            )

            # change train dataset
            train_data = generate_dataset(
                model=model,
                data=train_data,
                mixing_coef=0.95,
                num_steps=1,
                device=device
            )


if __name__ == "__main__":
    config = load_config()

    parser = setup_parser(ArgumentParser())
    args = parser.parse_args()

    if args.print_config:
        print(json.dumps(config, sort_keys=True, indent=4))
        exit()

    # set logging level
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    model = build_model(
        **config,
    ).to(args.device)

    print(config["num_joints"], " joints")

    train_dataloader, val_dataloader = get_datasets(
        feature_source=config["feature_source"],
        num_joints=config["num_joints"],
        batch_size=config["batch_size"],
        action_radius=config["action_radius"],
    )

    loss_func = get_loss_func(
        config["loss_func"], args.device, train_dataloader.dataset.target_mode
    )

    path = f"results/supervised/{args.subdir}/{config['loss_func']}/{config['num_joints']}_{int(time.time())}"
    logger = SummaryWriter(path)

    # store config
    with open(path + "/config.yaml", "w") as config_file:
        yaml.dump(config, config_file)


    if config["normal_learn"]["enabled"]:
        train(
            model=model,
            train_data=train_dataloader,
            val_data=val_dataloader,
            loss_func=loss_func,
            n_epochs=config["n_epochs"],
            logger=logger,
            val_interval=config["val_interval"],
            device=args.device,
            path=path,
        )
    elif config["active_learn"]["enabled"]:
        active_train(
            model=model,
            train_data=train_dataloader,
            val_data=val_dataloader,
            loss_func=loss_func,
            n_epochs=config["n_epochs"],
            logger=logger,
            val_interval=config["val_interval"],
            mixing_coef=config["active_learn"]["mixing_coef"],
            num_deviation_steps=config["active_learn"]["num_deviation_steps"],
            path=path,
            device=args.device,
        )
    else:
        ValueError("no learning mode enabled")