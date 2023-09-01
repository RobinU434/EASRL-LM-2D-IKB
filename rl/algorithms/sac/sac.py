# =====================================================================================================================
# This algorithm was adpated from:
# https://github.com/seungeunrho/minimalRL/blob/master/sac.py
# (date: 04.12.2022)
# =====================================================================================================================

import os
from typing import Any, Dict, List, Tuple, Union

import gym
import numpy as np
import torch
from gym import Env
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from envs.plane_robot_env.ikrlenv.env.plane_robot_env import PlaneRobotEnv
from logger.base_logger import Logger
from logger.fs_logger import FileSystemLogger
from rl.algorithms.algorithm import RLAlgorithm
from rl.algorithms.sac.actor.latent_actor import LatentActor
from rl.algorithms.sac.buffer import BufferDataset, ReplayBuffer
from rl.algorithms.sac.metrics import SACDistributionMetrics, SACScalarMetrics
from rl.algorithms.sac.policy_net import PolicyNet
from rl.algorithms.sac.q_net import QNet
from rl.algorithms.utils.helper import get_space_size
from rl.algorithms.utils.plot import kde_end_points, scatter_end_points
from utils.metrics import Metrics


class SAC(RLAlgorithm):
    def __init__(
        self,
        env: Env,
        logger: List[Union[Logger, SummaryWriter]],
        device: str = "cpu",
        n_epochs: int = 1000,
        lr_pi: float = 0.0005,
        lr_q: float = 0.001,
        init_alpha: float = 0.01,
        gamma: float = 0.98,
        batch_size: float = 32,
        buffer_limit: float = 50000,
        start_buffer_size: float = 1000,
        train_iterations: int = 20,
        tau: float = 0.01,  # for target network soft update,
        target_entropy: float = -1.0,  # for automated alpha update,
        lr_alpha: float = 0.001,  # for automated alpha update
        actor: Dict[str, Any] = {},
        log_dir: str = "results/sac",
        print_interval: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(env, logger, device, **kwargs)
        # Hyperparameters

        self._n_epochs = n_epochs
        self._print_interval = print_interval

        self._lr_pi = lr_pi
        self._lr_q = lr_q
        self._init_alpha = init_alpha
        self._gamma = gamma
        self._batch_size = batch_size
        self._buffer_limit = buffer_limit
        self._start_buffer_size = start_buffer_size
        self._train_iterations = train_iterations
        self._tau = tau  # for target network soft update
        self._target_entropy = target_entropy  # for automated alpha update
        self._lr_alpha = lr_alpha  # for automated alpha update

        # Replay-Buffer
        self._memory = ReplayBuffer(buffer_limit=buffer_limit)

        # Define networks
        input_dim = get_space_size(env.observation_space.shape)  # type: ignore
        output_dim = get_space_size(env.action_space.shape)  # type: ignore

        self._q1 = QNet(
            learning_rate=lr_q,
            input_dim_state=input_dim,
            input_dim_action=output_dim,
            device=device,
        ).to(self._device)
        self._q2 = QNet(
            learning_rate=lr_q,
            input_dim_state=input_dim,
            input_dim_action=output_dim,
            device=device,
        ).to(self._device)
        self._q1_target = QNet(
            learning_rate=lr_q,
            input_dim_state=input_dim,
            input_dim_action=output_dim,
            device=device,
        ).to(self._device)
        self._q2_target = QNet(
            learning_rate=lr_q,
            input_dim_state=input_dim,
            input_dim_action=output_dim,
            device=device,
        ).to(self._device)

        self._pi = PolicyNet(
            actor_config=actor,
            learning_rate=lr_pi,
            input_dim=input_dim,
            output_dim=output_dim,
            init_alpha=self._init_alpha,
            lr_alpha=self._lr_alpha,
            device=self._device,
        )

        self._log_dir = log_dir

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch

        with torch.no_grad():
            a_prime, log_prob = self._pi(s_prime)
            entropy = -self._pi.log_alpha.exp() * log_prob
            entropy = entropy.unsqueeze(dim=1).cpu()
            # TODO: make env easier
            # entropy = 0

            q1_val = self._q1_target(s_prime.to(self._device), a_prime.to(self._device))
            q2_val = self._q2_target(s_prime.to(self._device), a_prime.to(self._device))
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0].cpu()
            target = r + self._gamma * done * (min_q + entropy)
        return target

    def train(self):
        # target networks are initiated as copies from the actual q networks
        self._q1_target.load_state_dict(self._q1.state_dict().copy())
        self._q2_target.load_state_dict(self._q2.state_dict().copy())

        scalar_metrics = SACScalarMetrics()
        distribution_metrics = SACDistributionMetrics()
        for epoch_idx in range(self._n_epochs + 1):

            run_scalars, run_distributions = self._run_model()
            scalar_metrics.append(run_scalars)
            distribution_metrics.append(run_distributions)

            if len(self._memory) > self._start_buffer_size:
                train_metrics = self._train()
                scalar_metrics.append(train_metrics)

            # log metrics
            if epoch_idx % self._print_interval == 0:
                self._print_status(scalar_metrics, epoch_idx)
                # log metrics
                # in tensorboard
                # move to dedicated function

                self._log_scalars(scalar_metrics=scalar_metrics, epoch_idx=epoch_idx)
                self._log_distributions(
                    distr_metrics=distribution_metrics, epoch_idx=epoch_idx
                )

                # plot exploration
                """end_pos = np.array(end_pos)
                target_pos = np.array(target_pos)
                fig = kde_end_points(
                    end_pos[:, 0], end_pos[:, 1], target_pos[:, 0], target_pos[:, 1]
                )
                fig.savefig(
                    self._fs_logger._path + f"/polar_exploration_{epoch_idx}.png"
                )
                logger.add_figure("sac/polar_exploration", fig, epoch_idx)
                plt.close()"""
                # log vae stats

                # TODO: move to dedicated function
                # save model
                self._save(self._log_dir, scalar_metrics, epoch_idx)
                scalar_metrics = SACScalarMetrics()
                distribution_metrics = SACDistributionMetrics()

        # store metrics in a csv file
        self._dump()
        self._env.close()

    def load_checkpoint(self, path: str):
        """loads checkpoint from file system

        Args:
            path (str): path to checkpoint directory
        """
        epoch_idx = int(path.split("/")[-1]) 
        self._pi.load_checkpoint(path + f"/Actor_{epoch_idx}.pt")
        self._q1.load_checkpoint(path + f"/q1_{epoch_idx}.pt")
        self._q2.load_checkpoint(path + f"/q2_{epoch_idx}.pt")
        self._q1_target.load_checkpoint(path + f"/q1_target_{epoch_idx}.pt")
        self._q2_target.load_checkpoint(path + f"/q2_target_{epoch_idx}.pt")
        
    def inference(self, target_positions: ndarray) -> Dict[str, List[ndarray]]:
        """performs inference in a list of given target positions

        Args:
            target_positions (ndarray): shape (num_positions, 2)

        Returns:
            ndarray: trajectories from each arm shape: (num_positions, num_iterations, n_joints + 1, 2)
        """
        actions = []
        states = []
        trajectories = []
        self._env: PlaneRobotEnv
        for target_position in target_positions:
            s = self._env.reset(target_position)
            # .copy because of copy by value
            trajectory_actions = []
            trajectory_states = []
            trajectory = np.expand_dims(self._env._robot_arm.positions.copy(), axis=0)
            done = False
            while not done:
                # introduce batch size 1
                s_input = torch.from_numpy(s).float()
                s_input = s_input.unsqueeze(dim=0)
                a, log_prob = self._pi.forward(s_input)
                # detach grad from action to apply it to the environment where it is converted into a numpy.ndarray
                a = a.cpu().detach()
                trajectory_actions.append(a)
                trajectory_states.append(s)
                s_prime, r, done, info = self._env.step(a.numpy())
                trajectory = np.concatenate(
                    [
                        trajectory,
                        np.expand_dims(self._env._robot_arm.positions.copy(), axis=0),
                    ],
                    axis=0,
                )
                s = s_prime
            
            actions.append(np.stack(trajectory_actions))
            states.append(np.stack(trajectory_states))
            trajectories.append(trajectory)
        results = {"trajectories": trajectories, "actions": actions, "states": states}
        return results

    def print_model(self):
        print("===================================================")
        print("                     ACTOR")
        print("===================================================")
        print(self._pi.actor)
        print("===================================================")
        print("                     Q1")
        print("===================================================")
        print(self._q1)
        print("===================================================")
        print("                     Q2")
        print("===================================================")
        print(self._q2)
        print("===================================================")
        print("                     Q1-TARGET")
        print("===================================================")
        print(self._q1_target)
        print("===================================================")
        print("                     Q2-TARGET")
        print("===================================================")
        print(self._q2_target)

    def _run_model(self) -> Tuple[SACScalarMetrics, SACDistributionMetrics]:
        s = self._env.reset()
        done = False

        log_probs: List[float] = []
        actions: List[ndarray] = []
        scores = []
        while not done:
            # introduce batch size 1
            s_input = torch.from_numpy(s).float()
            s_input = s_input.unsqueeze(dim=0)

            a, log_prob = self._pi.forward(s_input)
            log_probs.append(log_prob.cpu().item())
            actions.append(a.cpu().detach().numpy())

            # detach grad from action to apply it to the environment where it is converted into a numpy.ndarray
            a = a.cpu().detach()
            s_prime, r, done, info = self._env.step(a.numpy())
            # squeeze action for memory
            a = a.squeeze()
            self._memory.put((s, a, r, s_prime, done))
            scores.append(r)
            s = s_prime

        scalar_metrics = SACScalarMetrics.from_dict(
            {
                "log_prob": np.stack(log_probs),
                "mean_score": np.array([np.mean(scores)]),
                "episode_len": np.array([self._env.num_steps]),  # type: ignore
            }
        )
        distribution_metrics = SACDistributionMetrics.from_dict(
            {"actions": np.stack(actions)}
        )

        return scalar_metrics, distribution_metrics

    def _train(self) -> SACScalarMetrics:
        """runs the train loop for policy net and q_nets"""
        critic_losses: List[float] = []
        entropies: List[ndarray] = []
        actor_losses: List[float] = []
        alpha_losses: List[float] = []
        alphas: List[float] = []
        for _ in range(self._train_iterations):
            mini_batch = self._memory.sample(self._batch_size)
            td_target = self.calc_target(mini_batch)

            critic_loss = self._q1.train(td_target.to(self._device), mini_batch)
            critic_losses.append(critic_loss.item())
            critic_loss = self._q2.train(td_target.to(self._device), mini_batch)
            critic_losses.append(critic_loss.item())

            entropy, actor_loss, alpha_loss = self._pi.train_net(
                self._q1, self._q2, mini_batch, self._target_entropy
            )

            entropies.append(entropy.cpu().detach().numpy())
            actor_losses.append(actor_loss.item())
            alpha_losses.append(alpha_loss.item())
            alphas.append(self._pi.alpha)

            self._q1.soft_update(self._q1_target, self._tau)
            self._q2.soft_update(self._q2_target, self._tau)

        scalar_metrics = SACScalarMetrics.from_dict(
            {
                "critic_losses": np.array(critic_losses),
                "entropy": np.stack(entropies).mean(axis=1),  # type: ignore
                "actor_loss": np.array(actor_losses),
                "alpha_loss": np.array(alpha_losses),
                "alpha": np.array(alphas)
            }
        )

        return scalar_metrics

    def _print_status(self, scalar_metrics: SACScalarMetrics, epoch_idx: int):
        print(
            "episode: {}, mean reward: {:.2f} alpha:{:.4f}".format(
                epoch_idx,
                scalar_metrics.mean_score.mean().item(),
                self._pi.log_alpha.exp(),
            )
        )

    def _log_scalars(self, scalar_metrics: SACScalarMetrics, epoch_idx: int):
        for key, value in scalar_metrics.mean().items():
            for logger in self._logger:
                logger.add_scalar(f"sac/{key}", value, epoch_idx)

    def _log_distributions(self, distr_metrics: SACDistributionMetrics, epoch_idx: int):
        for (
            key,
            value,
        ) in vars(distr_metrics).items():
            for logger in self._logger:
                logger.add_histogram(f"sac/{key}", value, epoch_idx)

    def _save(self, path: str, metrics: Metrics = Metrics(), epoch_idx: int = 0):
        path += f"/{epoch_idx}"
        os.makedirs(path)
        self._pi.save(path, metrics, epoch_idx)
        self._q1.save(path, metrics, epoch_idx, model_name="q1")
        self._q2.save(path, metrics, epoch_idx, model_name="q2")
        self._q1_target.save(path, metrics, epoch_idx, model_name="q1_target")
        self._q2_target.save(path, metrics, epoch_idx, model_name="q2_target")
