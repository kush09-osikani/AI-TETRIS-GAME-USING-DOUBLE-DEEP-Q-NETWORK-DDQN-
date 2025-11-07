
# train_ddqn.py
# Training driver for DDQN on the Tetris environment
import os
import sys
# Ensure parent directory (project root) is on sys.path so imports like `env_tetris` resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env_tetris import TetrisApp
# NOTE: this file references two different DDQNAgent symbols in different modules.
#from DDQN_training import DDQNAgent

from ddqn_agent import DDQNAgent
import matplotlib.pyplot as plt
import numpy as np

# training loop (copied from your script, minimal edits for imports)
def train_ddqn(env, agent, num_episodes=500, max_steps_per_episode=100):
    scores = []
    loss_v = []
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        loss = 0.0
        while not done and step < max_steps_per_episode:
            action = agent.act(state)
            # env.step may return (next_state, reward, done) or
            # (next_state, reward, done, info). Use a starred target
            # so we gracefully accept either form and ignore extra info.
            next_state, reward, done, *rest = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_on_batch()
            state = next_state
            total_reward += reward
            step += 1

        # soft target update at episode end (same as inside agent but applied here too to match original script)
        target_weights = agent.target_model.get_weights()
        online_weights = agent.model.get_weights()
        new_weights = [agent.tau * ow + (1 - agent.tau) * tw for ow, tw in zip(online_weights, target_weights)]
        agent.target_model.set_weights(new_weights)

        scores.append(total_reward)
        loss_v.append(loss)
        print(f"Ep:{ep:03d} Steps:{step:03d} Reward:{total_reward:.3f} Epsilon:{agent.epsilon:.3f} Loss:{loss:.5f}")

    return scores, loss_v, agent

# main training script (keeps your hyperparameters)
if __name__ == "__main__":
    env = TetrisApp(render=False)
    # Use the actual observation returned by env.reset() to determine state shape
    init_obs = env.reset()
    state_shape = np.array(init_obs).shape
    # Use environment's declared action size to avoid mismatches
    action_size = getattr(env, 'action_size', 26)
    agent = DDQNAgent(state_shape=state_shape, action_size=action_size,
                      lr=1e-4, batch_size=64, target_update_freq=100)

    # runs training (same defaults as your script)
    scores, loss, trained_agent = train_ddqn(env, agent, num_episodes=5000)

    # plot reward and loss curves exactly as your script did
    def smooth_curve(values, window=50):
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode='valid')

    plt.figure()
    plt.plot(scores, color='lightblue', alpha=0.2)
    plt.plot(smooth_curve(scores, window=50), color='blue', linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DDQN Learning Curve: Rewards per Episode")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(loss, color='lightcoral', alpha=0.2)
    plt.plot(smooth_curve(loss, window=50), color='red', linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("DDQN Learning Curve: Loss per Episode")
    plt.grid(True)
    plt.show()
