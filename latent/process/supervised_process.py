import logging
from copy import copy
from typing import Tuple
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

import numpy as np
from progress.bar import Bar
import torch
from torch.utils.data import DataLoader

from envs.robots.ccd import IK
from latent.criterion.base_criterion import Criterion
from latent.criterion.ik_criterion import IKLoss
from latent.data.supervised_dataset import (ActionStateDataset,
                                            ActionTargetDataset,
                                            TargetGaussianDataset)
from latent.data.utils import TargetMode
from latent.model.supervised_model import Regressor
from latent.model.utils.post_processor import PostProcessor
from latent.process.latent_process import LatentProcess
from latent.trainer.base_trainer import Trainer
from latent.trainer.supervised_trainer import SupervisedTrainer
from utils.kinematics import forward_kinematics
from utils.sampling import sample_target


class SupervisedProcess(LatentProcess):
    def __init__(self, **kwargs) -> None:
        self._model_entity = "supervised"
        super().__init__(**kwargs)

        self._model: Regressor

    def inference(self, *args, **kwargs) -> None:
        pass

    def feed_forward_inference(self) -> None:
        targets = np.repeat(
            np.expand_dims(sample_target(self._model.output_dim, 1), axis=0),
            self._sample_size,
            axis=0,
        )
        print("target: ", targets[0])

        # sample start position
        start_positions = np.zeros((self._sample_size, 3))
        start_position = sample_target(self._model.output_dim, 1)
        start_position = np.expand_dims(start_position, axis=0)
        noise = np.random.normal(
            np.zeros(2), np.ones(2) * 0.05, (self._sample_size, 2)
        )  # 2D noise
        start_positions[:, 0:2] = start_position + noise

        # solve IK for start positions
        state_angles = np.zeros((self._sample_size, self._model.output_dim))
        link = np.ones(self._model.output_dim)
        bar = Bar("solve IK for state config", max=self._sample_size)
        for idx in range(self._sample_size):
            state_action, _, _, _ = IK(
                start_positions[idx], state_angles[idx].copy(), link, err_min=0.001
            )
            state_angles[idx] = np.deg2rad(state_action)  # convert to rad
            bar.next()
        bar.finish()

        # build state vector
        print("build state")
        state = np.concatenate([targets, start_positions[:, 0:2], state_angles], axis=1)

        # make batches
        print("run self._model")
        split_idx = np.arange(0, self._sample_size, self._config["batch_size"])[1:]
        batches = np.split(state, split_idx)
        actions = []
        # bar = Bar("network forward pass", max=len(split_idx))
        for batch in batches:
            batch = torch.tensor(batch).to(self._device).type(torch.float32)
            # self._model forward pass
            action = self._model(batch)
            actions.append(action)
            # bar.next()
        # bar.finish()

        actions = torch.stack(actions).detach().squeeze().numpy()
        target_actions = state_angles + actions
        arm_positions = forward_kinematics(torch.tensor(target_actions))

        # plot
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.add_patch(Circle((0, 0), self._model.output_dim, fill=False))
        if self._config["action_radius"] != 0:
            ax.add_patch(
                Circle(
                    start_position[0],
                    get_action_radius(self._config["action_radius"], self._model.output_dim),
                    fill=False,
                )
            )
        ax.scatter(start_positions[:, 0], start_positions[:, 1], c="g", s=1)
        ax.scatter(arm_positions[:, -1, 0], arm_positions[:, -1, 1], c="r", s=1)
        ax.scatter(targets[0, 0], targets[0, 1], c="b", s=1)
        fig.savefig("results/supervised_inference.png")

        fig = plt.figure()
        ax = fig.add_subplot()
        if self._model.post_processor.enabled:
            # ax.set_xlim([self._model.post_processor.min_action, self._model.post_processor.max_action])
            bins = np.linspace(-1, 1, 200)
        else:
            bins = 50
        for idx, joint_actions in enumerate(actions.T):
            ax.hist(
                joint_actions,
                bins=bins,
                alpha=1 / np.sqrt(self._model.output_dim),
                label=str(idx),
            )
        fig.legend()
        fig.savefig("results/supervised_action_distribution.png")

    def greedy_inference(self) -> None:
        pass

    def _load_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        num_joints: int = self._config["num_joints"]
        feature_source: str = self._config["feature_source"]
        batch_size: int = self._config["batch_size"]
        action_radius: float = self._config["action_radius"]

        if feature_source == "state":
            train_data = ActionStateDataset(
                action_file=f"./datasets/{num_joints}/train/actions_IK_random_start.csv",
                state_file=f"./datasets/{num_joints}/train/state_IK_random_start.csv",
                action_constrain_radius=action_radius,
            )
            train_dataloader = DataLoader(train_data, batch_size=batch_size)
            val_data = ActionStateDataset(
                action_file=f"./datasets/{num_joints}/val/actions_IK_random_start.csv",
                state_file=f"./datasets/{num_joints}/val/state_IK_random_start.csv",
                action_constrain_radius=action_radius,
            )
            val_dataloader = DataLoader(val_data, batch_size=batch_size)
            test_data = ActionStateDataset(
                action_file=f"./datasets/{num_joints}/test/actions_IK_random_start.csv",
                state_file=f"./datasets/{num_joints}/test/state_IK_random_start.csv",
                action_constrain_radius=action_radius,
            )
            test_dataloader = DataLoader(test_data, batch_size=batch_size)

        elif feature_source == "targets":
            train_data = ActionTargetDataset(
                action_file=f"./datasets/{num_joints}/train/actions_IK_random_start.csv",
                target_file=f"./datasets/{num_joints}/train/targets_IK_random_start.csv",
            )
            train_dataloader = DataLoader(train_data, batch_size=batch_size)
            val_data = ActionTargetDataset(
                action_file=f"./datasets/{num_joints}/val/actions_IK_random_start.csv",
                target_file=f"./datasets/{num_joints}/val/targets_IK_random_start.csv",
            )
            val_dataloader = DataLoader(val_data, batch_size=batch_size)
            test_data = ActionTargetDataset(
                action_file=f"./datasets/{num_joints}/test/actions_IK_random_start.csv",
                target_file=f"./datasets/{num_joints}/test/targets_IK_random_start.csv",
            )
            test_dataloader = DataLoader(test_data, batch_size=batch_size)

        elif feature_source == "gaussian_target":
            train_data = TargetGaussianDataset(
                state_file=f"./datasets/{num_joints}/train/state_IK_random_start.csv",
                std=action_radius,
                target_mode=TargetMode.INTERMEDIATE_POSITION,
            )
            train_dataloader = DataLoader(train_data, batch_size=batch_size)
            val_data = TargetGaussianDataset(
                state_file=f"./datasets/{num_joints}/val/state_IK_random_start.csv",
                std=action_radius,
                target_mode=TargetMode.INTERMEDIATE_POSITION,
            )
            val_dataloader = DataLoader(val_data, batch_size=batch_size)
            test_data = TargetGaussianDataset(
                state_file=f"./datasets/{num_joints}/test/state_IK_random_start.csv",
                std=action_radius,
                target_mode=TargetMode.INTERMEDIATE_POSITION,
            )
            test_dataloader = DataLoader(test_data, batch_size=batch_size)

        else:
            logging.error(
                f"feature source has to be either 'targets' or 'state', you chose: {feature_source}"
            )
            return None, None, None

        return train_dataloader, val_dataloader, test_dataloader

    def _load_criterion(self) -> Criterion:
        loss_config = copy(self._config["loss_func"])
        loss_config.pop("type")
        criterion = IKLoss(
            **loss_config, target_mode=self._train_data.dataset.target_mode
        )

        return criterion

    def _build_model(self) -> Regressor:
        post_processor = PostProcessor(**copy(self._config["post_processor_config"]))
        feature_source = self._config["feature_source"]
        num_joints = self._config["num_joints"]
        learning_rate = self._config["learning_rate"]
        action_radius = self._config["action_radius"]

        if feature_source == "state" or feature_source == "gaussian_target":
            logging.info("use 'state' as feature source")
            model = Regressor(
                4 + num_joints,
                num_joints,
                learning_rate,
                post_processor,
                action_radius,
            )
        elif feature_source == "targets":
            logging.info("use 'targets' as feature source")
            model = Regressor(
                2, num_joints, learning_rate, post_processor, action_radius
            )
        else:
            logging.error(
                f"feature source has to be either 'targets' or 'state', you chose: {feature_source}"
            )
            return

        return model.to(self._device)

    def _build_model_from_config(self) -> Regressor:
        """in this function we rely only on information provided by the config file

        Returns:
            Regressor: _description_
        """
        return self._build_model()

    def _build_trainer(self) -> Trainer:
        trainer = SupervisedTrainer(
            model=self._model,
            train_data=self._train_data,
            val_data=self._val_data,
            test_data=self._test_data,
            n_epochs=self._config["n_epochs"],
            criterion=self._criterion,
            logger=self._logger,
            val_interval=5,
            device=self._device,
            results_path=self._save_dir,
        )

        return trainer
