
# ddqn_agent.py
# Double DQN network and agent implementation (extracted from your training script)

import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers

class DDQN(tf.keras.Model):
    """
    Neural network that maps flattened state -> Q-values for actions.
    Architecture (from your script):
      Dense(256) -> Dense(256) -> Dense(128) -> Dense(action_size)
    """
    def __init__(self, state_shape, action_size):
        super(DDQN, self).__init__()
        self.flat_dim = int(np.prod(state_shape))

        self.fc1 = layers.Dense(256, activation='relu', kernel_initializer='he_normal')
        self.fc2 = layers.Dense(256, activation='relu', kernel_initializer='he_normal')
        self.fc3 = layers.Dense(128, activation='relu', kernel_initializer='he_normal')
        self.fc4 = layers.Dense(action_size, activation='linear', kernel_initializer='he_normal')

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, (-1, self.flat_dim))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class DDQNAgent:
    """
    DDQN agent with:
      - online model: self.model
      - target model: self.target_model
      - experience replay: deque(self.memory)
      - soft target updates (tau) applied in train_on_batch and at episode end
    All hyperparameters are taken from your script.
    """
    def __init__(self,
                 state_shape,
                 action_size,
                 lr=1e-3,
                 tau=0.05,
                 gamma=0.99,
                 batch_size=64,
                 memory_size=10000,
                 epsilon_start=1.0,
                 epsilon_min=0.001,
                 epsilon_decay=0.995,
                 target_update_freq=1000):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # build models
        self.model = DDQN(state_shape, action_size)
        self.target_model = DDQN(state_shape, action_size)

        # call once with dummy input so variables are created
        dummy = tf.zeros((1,) + state_shape, dtype=tf.float32)
        _ = self.model(dummy)
        _ = self.target_model(dummy)
        self.update_target_model()

        self.optimizer = optimizers.Adam(learning_rate=lr)
        self.train_step_count = 0
        self.target_update_freq = target_update_freq

    def update_target_model(self):
        """Copy online weights to target (hard copy)."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, s, a, r, s_next, done):
        """Store experience tuple in replay memory."""
        self.memory.append((np.array(s, dtype=np.float32),
                            int(a),
                            float(r),
                            np.array(s_next, dtype=np.float32),
                            bool(done)))

    def act(self, state):
        """Epsilon-greedy action selection using the online model."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        q_vals = self.model(state_tensor)
        return int(tf.argmax(q_vals[0]).numpy())

    def sample_batch(self):
        """Sample a random minibatch from replay memory."""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones.astype(np.float32)

    def train_on_batch(self):
        """
        Train on one sampled minibatch using Double DQN target:
          1. select best next action using online model (self.model)
          2. evaluate that action using target model (self.target_model)
        Then update online model by minimizing MSE(target_q, predicted_q).
        Applies epsilon decay and soft target updates per your training script.
        Returns scalar loss (float) or 0.0 if not enough samples.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.sample_batch()

        # Double DQN target computation
        q_next_vals = self.model(next_states)           # online model predicts next-state Qs
        best_act = tf.argmax(q_next_vals, axis=1).numpy()  # indices of best next actions
        q_next_target = self.target_model(next_states)  # target model Qs for next states
        indices = tf.stack([tf.range(self.batch_size), best_act], axis=1)
        next_q = tf.gather_nd(q_next_target, indices)   # pick Q_target(s', argmax_a Q_online(s',a))
        target_q = rewards + self.gamma * next_q * (1 - dones)

        with tf.GradientTape() as tape:
            q_vals = self.model(states)
            indices_ = tf.stack([tf.range(self.batch_size), actions], axis=1)
            pred_q = tf.gather_nd(q_vals, indices_)
            loss = tf.reduce_mean(tf.square(target_q - pred_q))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # soft target update triggered at intervals
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            target_weights = self.target_model.get_weights()
            online_weights = self.model.get_weights()
            new_weights = [self.tau * ow + (1 - self.tau) * tw for ow, tw in zip(online_weights, target_weights)]
            self.target_model.set_weights(new_weights)

        return float(loss.numpy())
