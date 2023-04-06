# =====================================================================================================================
# This algorithm was adpated from: 
# https://github.com/seungeunrho/minimalRL/blob/master/sac.py 
# (date: 04.12.2022)
# =====================================================================================================================

import gym
import torch
import numpy as np

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from envs.plane_robot_env import PlaneRobotEnv
from logger.fs_logger import FileSystemLogger
from algorithms.sac.q_net import QNet
from algorithms.sac.actor.latent_actor import LatentActor
from algorithms.sac.buffer import BufferDataset, ReplayBuffer
from algorithms.sac.policy_net import PolicyNet
from algorithms.helper.helper import get_space_size
from algorithms.helper.plot import kde_end_points, scatter_end_points


class SAC:
    def __init__(
        self, 
        env: gym.Env,
        logging_writer: SummaryWriter,
        fs_logger: FileSystemLogger,
        lr_pi: float = 0.0005, 
        lr_q: float = 0.001,
        init_alpha: float = 0.01,
        gamma: float = 0.98,
        batch_size: float = 32,
        buffer_limit: float = 50000,
        start_buffer_size: float = 1000,
        train_iterations: float = 20,
        tau: float = 0.01, # for target network soft update,
        target_entropy: float = -1.0, # for automated alpha update,
        lr_alpha: float = 0.001,  # for automated alpha update
        action_covariance_decay: float = 0.5,
        action_covariance_mode: str = "independent",
        action_magnitude: float = 1,
        actor_config: dict = None,
        ) -> None:
        
        self._env: PlaneRobotEnv = env 

        self._logger = logging_writer
        self._fs_logger = fs_logger
        # logging every print interval one trajectory and store it
        # x0 -> episode
        # x1 -> time step 

        if self._fs_logger is not None:
            self._trajectory_logger = FileSystemLogger(fs_logger._path)
        else:
            self._trajectory_logger = None

        #Hyperparameters
        self._lr_pi             = lr_pi
        self._lr_q              = lr_q
        self._init_alpha        = init_alpha
        self._gamma             = gamma
        self._batch_size        = batch_size
        self._buffer_limit      = buffer_limit
        self._start_buffer_size = start_buffer_size
        self._train_iterations  = train_iterations
        self._tau               = tau # for target network soft update
        self._target_entropy    = target_entropy # for automated alpha update
        self._lr_alpha          = lr_alpha  # for automated alpha update
        self._action_covariance_decay = action_covariance_decay
        self._action_covariance_mode = action_covariance_mode
        self._action_magnitude  = action_magnitude
        
        # Replay-Buffer
        self._memory = ReplayBuffer(buffer_limit=buffer_limit)

        # Define networks
        input_size = get_space_size(env.observation_space.shape)
        output_size = get_space_size(env.action_space.shape)

        self._q1 = QNet(lr_q, input_size_state=input_size, input_size_action=output_size)
        self._q2 = QNet(lr_q, input_size_state=input_size, input_size_action=output_size)
        self._q1_target = QNet(lr_q, input_size_state=input_size, input_size_action=output_size)
        self._q2_target = QNet(lr_q, input_size_state=input_size, input_size_action=output_size)

        self._pi = PolicyNet(
            actor_config,
            lr_pi,
            input_size,
            output_size,
            self._init_alpha,
            self._lr_alpha,
            self._env.observation_space._shape[0],
            self._action_covariance_mode,
            self._action_covariance_decay,
            self._action_magnitude,
            )

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch

        with torch.no_grad():
            a_prime, log_prob= self._pi(s_prime)
            entropy = -self._pi.log_alpha.exp() * log_prob
            entropy = entropy.unsqueeze(dim=1)
            # TODO: make env easier
            # entropy = 0
            
            q1_val = self._q1_target(s_prime, a_prime)
            q2_val = self._q2_target(s_prime, a_prime)
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = r + self._gamma * done * (min_q + entropy)
        return target

    def train(self, n_epochs, print_interval: int = 20):
        # target networks are initiated as copies from the actual q networks
        self._q1_target.load_state_dict(self._q1.state_dict().copy())
        self._q2_target.load_state_dict(self._q2.state_dict().copy())
        
        score = 0.0
        num_steps = 0
        
        end_pos = []
        target_pos = []
        for epoch_idx in range(n_epochs + 1):
            actions = torch.tensor([])
            log_probs = torch.tensor([])
            
            s = self._env.reset()
            done = False
            while not done:
                # introduce batch size 1
                s_input = torch.from_numpy(s).float()
                s_input = s_input.unsqueeze(dim=0)

                a, log_prob = self._pi.forward(s_input)
                log_probs = torch.cat([log_probs, torch.tensor([log_prob])])
                actions = torch.cat([actions, a.detach()])

                # detach grad from action to apply it to the environment where it is converted into a numpy.ndarray
                a = a.detach()
                s_prime, r, done, info = self._env.step(a)
                self._memory.put((s, a, r, s_prime, done))  # why is the reward divided by 10?????
                score += r
                s = s_prime

                # log trajectory
                if epoch_idx % print_interval == 0 and self._trajectory_logger is not None:
                    self._trajectory_logger.add_scalar("state", s, epoch_idx, self._env.num_steps)
                    self._trajectory_logger.add_scalar("action", a.tolist(), epoch_idx, self._env.num_steps)
                    self._trajectory_logger.add_scalar("reward", r, epoch_idx, self._env.num_steps)
                    self._trajectory_logger.add_scalar("s_prime", s_prime, epoch_idx, self._env.num_steps)
                    self._trajectory_logger.add_scalar("done", done, epoch_idx, self._env.num_steps)

                # append end positions for plotting exploration
                end_pos.append(s[2: 4])
                target_pos.append(s[0: 2])
                        
            num_steps += self._env.num_steps
            logging_entropy = []
            actor_losses = []
            critic_losses = []
            alpha_losses = []
            
            if len(self._memory) > self._start_buffer_size:
                for _ in range(self._train_iterations):
                    mini_batch = self._memory.sample(self._batch_size)
                    td_target = self.calc_target(mini_batch)
                    
                    critic_loss = self._q1.train_net(td_target, mini_batch)
                    critic_losses.append(critic_loss.item())
                    critic_loss = self._q2.train_net(td_target, mini_batch)
                    critic_losses.append(critic_loss.item())
                    
                    entropy, actor_loss, alpha_loss = self._pi.train_net(
                        self._q1, 
                        self._q2, 
                        mini_batch,
                        self._target_entropy)

                    logging_entropy.append(entropy.mean())
                    actor_losses.append(actor_loss)
                    alpha_losses.append(alpha_loss)

                    self._q1.soft_update(self._q1_target, self._tau)
                    self._q2.soft_update(self._q2_target, self._tau)

                if type(self._pi.actor) == LatentActor and self._pi.actor.vae_learning:
                    data = BufferDataset(self._memory)
                    data = DataLoader(data, batch_size=128, shuffle=True)
                    r_loss_mean, kl_loss_mean = self._pi.actor.train_vae(data, self._train_iterations) 
            
            # log metrics
            if epoch_idx % print_interval == 0 and epoch_idx != 0:
                avg_episode_len = num_steps / print_interval 
                mean_reward = score / num_steps
                print("# of episode: {}, mean reward / step : {:.2f} alpha:{:.4f}".format(epoch_idx, mean_reward, self._pi.log_alpha.exp()))
                # log metrics
                # in tensorboard
                if self._logger is not None:
                    self._logger.add_scalar("stats/mean_reward", mean_reward, epoch_idx)
                    self._logger.add_scalar("stats/mean_episode_len", avg_episode_len, epoch_idx)
                    self._logger.add_scalar("sac/alpha", self._pi.log_alpha.exp(), epoch_idx)
                    self._logger.add_scalar("sac/entropy", torch.tensor(logging_entropy).mean(), epoch_idx)
                    self._logger.add_scalar("sac/actor_loss", torch.tensor(actor_losses).mean(), epoch_idx)
                    self._logger.add_scalar("sac/critic_loss", torch.tensor(critic_losses).mean(), epoch_idx)
                    self._logger.add_scalar("sac/alpha_loss", torch.tensor(alpha_losses).mean(), epoch_idx)
                    self._logger.add_scalar("sac/log_prob", np.array(log_probs).mean(), epoch_idx)
                    #  self._logger.add_histogram("sac/action_distr", actions, epoch_idx)
                    # plot exploration
                    end_pos = np.array(end_pos)
                    target_pos = np.array(target_pos) 
                    fig = kde_end_points(end_pos[:, 0], end_pos[:, 1], target_pos[:, 0], target_pos[:, 1])
                    fig.savefig(self._fs_logger._path + f"/polar_exploration_{epoch_idx}.png")
                    self._logger.add_figure("sac/polar_exploration", fig, epoch_idx)
                    plt.close()
                    # log vae stats
                    if type(self._pi.actor) == LatentActor and self._pi.actor.vae_learning:
                        self._logger.add_scalar("vae/r_loss", r_loss_mean, epoch_idx)
                        self._logger.add_scalar("vae/kl_loss", kl_loss_mean, epoch_idx)

                # in file system
                if self._fs_logger is not None:
                    self._fs_logger.add_scalar("stats/mean_reward", float(mean_reward), epoch_idx)
                    self._fs_logger.add_scalar("stats/mean_episode_len", float(avg_episode_len), epoch_idx)
                    self._fs_logger.add_scalar("sac/alpha", float(self._pi.log_alpha.exp()), epoch_idx)
                    self._fs_logger.add_scalar("sac/entropy", torch.tensor(logging_entropy).mean(), epoch_idx)
                    self._fs_logger.add_scalar("sac/actor_loss", torch.tensor(actor_losses).mean(), epoch_idx)
                    self._fs_logger.add_scalar("sac/critic_loss", torch.tensor(critic_losses).mean(), epoch_idx)
                    self._fs_logger.add_scalar("sac/alpha_loss", torch.tensor(alpha_losses).mean(), epoch_idx)
                    self._fs_logger.add_scalar("sac/log_prob", np.array(log_probs).mean(), epoch_idx)
                    # log vae stats
                    if type(self._pi.actor) == LatentActor and self._pi.actor.vae_learning:
                        self._fs_logger.add_scalar("vae/r_loss", r_loss_mean, epoch_idx)
                        self._fs_logger.add_scalar("vae/kl_loss", kl_loss_mean, epoch_idx)

                # save model
                torch.save({
                    'epoch': epoch_idx,
                    'pi_model_state_dict': self._pi.state_dict(),
                    'pi_optimizer_state_dict': self._pi.optimizer.state_dict(),
                    'q1_model_state_dict': self._q1.state_dict(),
                    'q1_optimizer_state_dict': self._q1.optimizer.state_dict(),
                    'q2_model_state_dict': self._q2.state_dict(),
                    'q2_optimizer_state_dict': self._q2.optimizer.state_dict(),
                    'q1_target_model_state_dict': self._q1_target.state_dict(),
                    'q1_target_optimizer_state_dict': self._q1_target.optimizer.state_dict(),
                    'q2_target_model_state_dict': self._q2_target.state_dict(),
                    'q2_target_optimizer_state_dict': self._q2_target.optimizer.state_dict(),
                    'reward': float(mean_reward),
                }, self._fs_logger.path + f"/model_{epoch_idx}_reward_{mean_reward:.4f}.pt")     

                score = 0.0
                num_steps = 0
                end_pos = []
                target_pos = []
        
        # store metrics in a csv file
        if self._fs_logger is not None:
            self._fs_logger.dump()
        if self._trajectory_logger is not None:
            self._trajectory_logger.dump("trajectory.csv")

        self._env.close()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self._pi.load_state_dict(checkpoint["pi_model_state_dict"])
        self._q1.load_state_dict(checkpoint["q1_model_state_dict"])
        self._q2.load_state_dict(checkpoint["q2_model_state_dict"])
        self._q1_target.load_state_dict(checkpoint["q1_target_model_state_dict"])
        self._q2_target.load_state_dict(checkpoint["q2_target_model_state_dict"])

    def inference(self, target_positions: np.array):
        trajectories = list()
        for target_position in target_positions:
            s = self._env.reset(target_position)
            # .copy because of copy by value
            trajectory = np.expand_dims(self._env._robot_arm.positions.copy(), axis=0)
            done = False
            while not done:
                # introduce batch size 1
                s_input = torch.from_numpy(s).float()
                s_input = s_input.unsqueeze(dim=0)
                a, log_prob = self._pi.forward(s_input)
                # detach grad from action to apply it to the environment where it is converted into a numpy.ndarray
                a = a.detach()
                s_prime, r, done, info = self._env.step(a)
                trajectory = np.concatenate([trajectory, np.expand_dims(self._env._robot_arm.positions.copy(), axis=0)], axis=0)
                s = s_prime

            trajectories.append(trajectory)
        
        return np.array(trajectories)


if __name__ == '__main__':
    env = PlaneRobotEnv(4, 1)
    sac = SAC(env)