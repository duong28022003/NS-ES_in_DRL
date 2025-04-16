import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
import gymnasium as gym
from moviepy import *
import os
import tqdm
from tqdm import tqdm
import argparse
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self, env, device = device):
        super(NeuralNetwork, self).__init__()
        self.device = device
        
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, env.action_space.n)
        )
        
    def forward(self, state):
        return self.network(state)

    def choose_action(self, state):
        if type(state) == tuple:
            state = state[0]
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self(state.unsqueeze(0))
        best_action = torch.argmax(q_values, dim=1)[0]

        return best_action.detach().item()
    
class DQN():
    def __init__(self, env_id, device=device, max_epsilon=1.0, min_epsilon=0.001, max_num_steps=100000, 
                 max_step_per_episode = 500, convergence_window = 0, convergence_threshold = 10, min_episodes = 500,
                 epsilon_decay_intervals=10000, gamma=0.99, alpha=1e-3, 
                 memory_size=100000, min_replay_size=1000, batch_size=256, 
                 target_update_frequency=1000, capture_video=False, tqdm_flag = False, seed=1):
        self.env_id = env_id
        self.device = device
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.max_num_steps = max_num_steps
        self.max_step_per_episode = max_step_per_episode
        self.epsilon_decay_intervals = epsilon_decay_intervals
        self.gamma = gamma
        self.alpha = alpha
        self.memory_size = memory_size
        self.min_replay_size = min_replay_size
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.capture_video = capture_video
        self.tqdm_flag = tqdm_flag
        self.seed = seed
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.min_episodes = min_episodes
        self.save_path = "runs/DQN/" + self.env_id + '__' + str(seed)
        self.video_path = "videos/DQN/" + self.env_id + '__' + str(seed)
        
        # Set global seeds
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def fill_memory(self):
        env = gym.make(self.env_id)
        env.reset(seed=self.seed)
        memory = deque(maxlen=self.memory_size)
        state = env.reset(seed=self.seed)
        for _ in range(self.min_replay_size):
            if type(state) == tuple:
                state = state[0]
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            experience = (state, action, reward, done, next_state)
            memory.append(experience)
            state = next_state
            if done:
                state = env.reset(seed=self.seed)
        return memory

    @staticmethod
    def play(env_id, q_net, seed=1, max_step=500, capture_video=False, video_output_path='output_video.mp4'):
        env = gym.make(env_id, render_mode='rgb_array')
        action_network = NeuralNetwork(env, device=torch.device('cpu'))
        action_network.load_state_dict(q_net.state_dict())
        action_network.eval()
        record_frames = []
        env.reset(seed=seed)
        state = env.reset(seed=seed)
        done = False
        total_rewards = 0
        total_steps = 0
        with torch.no_grad():
            while not done:
                frame = env.render()
                record_frames.append(frame)
                if total_steps == max_step:
                    break
                if type(state) == tuple:
                    state = state[0]
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = action_network.choose_action(state_tensor)
                next_state, reward, done, truncated, info = env.step(action)
                total_rewards += reward
                state = next_state
                total_steps += 1
            env.close()
            
        if capture_video:
            clip = ImageSequenceClip(record_frames, fps=30) 
            video_output_file = video_output_path
            clip.write_videofile(video_output_file, codec="libx264")
        return total_rewards, total_steps

    @staticmethod
    def play_multiple_times(env_id, q_net, seed_list, max_step=500):
        rewards, steps = [], []
        for seed in seed_list:
            episode_reward, episode_step = DQN.play(env_id, q_net, seed, max_step, capture_video=False)
            rewards.append(episode_reward)
            steps.append(episode_step)
        return rewards, steps
            
    def train(self):
        env = gym.make(self.env_id)
        device = self.device
        env.reset(seed=self.seed)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.video_path):
            os.makedirs(self.video_path)
        
        q_net = NeuralNetwork(env, device).to(device)
        target_net = NeuralNetwork(env, device).to(device)
        target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.Adam(q_net.parameters(), lr=self.alpha)

        memory = self.fill_memory()
        reward_buffer = deque(maxlen=100)
        
        reward_per_episode = 0.0
        state = env.reset(seed=self.seed)
        all_rewards = []
        training_times = []
        
        training_iters = tqdm(range(self.max_num_steps)) if self.tqdm_flag else range(self.max_num_steps)
        
        for step in training_iters:
            early_stop = False
            start_time = time.time()
            epsilon = np.interp(step, [0, self.epsilon_decay_intervals], [self.max_epsilon, self.min_epsilon])
            random_number = np.random.uniform(0, 1)
            if random_number <= epsilon:
                action = env.action_space.sample()
            else:
                if type(state) == tuple:
                    state = state[0]
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                action = q_net.choose_action(state_tensor)
            next_state, reward, done, truncated, info = env.step(action)
            experience = (state, action, reward, done, next_state)
            memory.append(experience)
            reward_per_episode += reward

            state = next_state
            
            if done or step % self.max_step_per_episode == 0:
                state = env.reset(seed=self.seed)
                reward_buffer.append(reward_per_episode)
                all_rewards.append((step, reward_per_episode))
                reward_per_episode = 0.0
                
            # Check convergence condition
            if (self.convergence_window and 
                len(all_rewards) > self.convergence_window and
                len(all_rewards) >= self.min_episodes):  
                episode_rewards_buffer = [reward[1] for reward in all_rewards[-self.convergence_window:]]
                if all(abs(x-episode_rewards_buffer[0]) <= self.convergence_threshold for x in episode_rewards_buffer):
                    early_stop = True

            if not early_stop:
            
                # Take a batch of experiences from the memory
                experiences = random.sample(memory, self.batch_size)
                states = [ex[0] for ex in experiences]
                actions = [ex[1] for ex in experiences]
                rewards = [ex[2] for ex in experiences]
                dones = [ex[3] for ex in experiences]
                next_states = [ex[4] for ex in experiences]
                
                # Debug for gym environment
                for i in range(len(states)):
                    if type(states[i]) == tuple:
                        states[i] = states[i][0]

                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device) 
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

                # Compute targets using the formulation sample = r + gamma * max q(s',a')
                target_q_values = target_net(next_states)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
                targets = rewards + self.gamma * (1-dones) * max_target_q_values

                # Compute loss
                q_values = q_net(states)

                action_q_values = torch.gather(input=q_values, dim=1, index=actions)
                loss = torch.nn.functional.mse_loss(action_q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                end_time = time.time()
                training_times.append(end_time - start_time)
            
            # Update target network and save the results
            if (step+1) % self.target_update_frequency == 0:
                target_net.load_state_dict(q_net.state_dict())
                
            if (step+1) % (10*self.target_update_frequency) == 0 or early_stop:
                q_net_path = self.save_path + '/Step_' + str(step+1) + '.pth'
                video_record_path = self.video_path + '/Step_' + str(step+1) + '.mp4'
                torch.save(q_net.state_dict(), q_net_path)
                np.savez_compressed(self.save_path + '/results.npz', all_rewards=all_rewards, training_times = training_times)
                DQN.play(self.env_id, q_net, self.seed, 500, self.capture_video, video_record_path)
                
            # Print training results
            if (step+1) % 1000 == 0:
                average_reward = np.mean(reward_buffer)
                print(f'Episode: {len(all_rewards)} Step: {step+1} Average reward: {average_reward:.2f} Total training time: {sum(training_times):.2f} seconds')
            
            if early_stop:
                print(f'Early stopping training at step {step+1}')
                break
                
def main():
    parser = argparse.ArgumentParser(description="Deep Q-Network Training")
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generators')
    parser.add_argument('--env-id', type=str, default='CartPole-v0', help='Gym environment ID')
    parser.add_argument('--total-timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--capture-video', action='store_true', help='Capture video of the agent')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help='Device to run the training on (cpu or cuda)')
    parser.add_argument('--tqdm_flag', action='store_true',help='Training process with tqdm')
    parser.add_argument('--max-epsilon', type=float, default=1.0, help='Starting epsilon for exploration')
    parser.add_argument('--min-epsilon', type=float, default=0.001, help='Minimum epsilon for exploration')
    parser.add_argument('--epsilon-decay-intervals', type=int, default=10000, help='Epsilon decay intervals')
    parser.add_argument('--max-step-per-episode', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
    parser.add_argument('--alpha', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--memory-size', type=int, default=100000, help='Replay memory size')
    parser.add_argument('--min-replay-size', type=int, default=1000, help='Minimum replay buffer size before training')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--target-update-frequency', type=int, default=1000, help='Target network update frequency')
    parser.add_argument('--c-window', type=int, default=0, help='Convergence window')
    parser.add_argument('--c-threshold', type=float, default=10, help='Convergence threshold value')
    parser.add_argument('--min-episodes', type=int, default=500, help='Min number of episodes before checking convergence condition')
    
    args = parser.parse_args()

    dqn = DQN(
        env_id=args.env_id,
        device=torch.device(args.device),
        max_epsilon=args.max_epsilon,
        min_epsilon=args.min_epsilon,
        max_num_steps=args.total_timesteps,
        max_step_per_episode=args.max_step_per_episode,
        epsilon_decay_intervals=args.epsilon_decay_intervals,
        gamma=args.gamma,
        alpha=args.alpha,
        memory_size=args.memory_size,
        min_replay_size=args.min_replay_size,
        batch_size=args.batch_size,
        target_update_frequency=args.target_update_frequency,
        convergence_window=args.c_window,
        convergence_threshold=args.c_threshold,
        min_episodes=args.min_episodes,
        capture_video=args.capture_video,
        tqdm_flag=args.tqdm_flag,
        seed=args.seed
    )
    
    dqn.train()

if __name__ == "__main__":
    main()