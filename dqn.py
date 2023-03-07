import tensorflow as tf
import gym
import random

from dc import *
from hospital import *

# Define the deep reinforcement learning algorithm
class DQN:
    def __init__(self, SETTINGS, env):                      # Constructor method for the DQN class
        self.env = env                                      # Set the environment for the class
        self.memory = []                                    # Initialize memory
        self.epsilon = SETTINGS.epsilon                     # Set exploration rate epsilon
        self.epsilon_min = SETTINGS.epsilon_min
        self.epsilon_decay = SETTINGS.epsilon_decay
        self.learning_rate = SETTINGS.learning_rate         # Set learning rate
        self.model = self.build_model()                     

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)   # Create optimizer using Adam
        self.gamma = SETTINGS.gamma                         # Set discount factor gamma
        self.batch_size = SETTINGS.batch_size               # Set batch size for replay buffer
    

    def build_model(self):                                  # Method to build the model
        # Get the size of the input state and output action spaces
        input_size = len(np.ravel(self.env.state))          
        output_size = self.env.action_space.shape[0]*self.env.action_space.shape[1]  

        # Create the input layer and two hidden layers with relu activation and output layer with sigmoid activation
        inputs = tf.keras.layers.Input(shape=(input_size,))
        layer_0 = tf.keras.layers.Dense(128, activation='relu')(inputs)
        layer_1 = tf.keras.layers.Dense(64, activation='relu')(layer_0)
        outputs = tf.keras.layers.Dense(output_size, activation='sigmoid')(layer_1)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

        return model


    def select_action(self, state):                         # Method to select the action to take
        if np.random.rand() <= self.epsilon:                # Choose a random action with probability epsilon
            Q_values = self.env.action_space.sample()
        else:                                               # Choose the action with the highest predicted Q-value
            # Predict the action given the state.
            Q_values = self.model.predict(np.ravel(state).reshape(1,-1))

        # For each row in the Q_matrix (for each blood group), put a 1 in the cell with the highest Q-value.
        # action = np.eye(self.env.action_space.shape[1])[np.argmax(np.reshape(Q_values, self.env.action_space.shape), axis=1)]
        action = np.argmax(np.reshape(Q_values, self.env.action_space.shape), axis=1)
        
        return action


    def update(self):

        batch = random.sample(self.memory, self.batch_size)

        states = []
        Q_tables = []

        for sample in batch:

            state, action, reward, next_state, _ = sample

            # Compute the target Q-values
            Q_next = np.reshape(self.model.predict(np.ravel(next_state).reshape(1,-1)), self.env.action_space.shape)             # Predict the Q-values for the next states
            max_Q_next = np.max(Q_next, axis=1)
            Q_target = reward + (self.gamma * max_Q_next)  # Compute the target Q-values using the Bellman equation

            # Compute the current Q-values
            Q_table = np.reshape(self.model.predict(np.ravel(state).reshape(1,-1)), self.env.action_space.shape)             # Predict the Q-values for the next states
            Q_table[:,action] = Q_target # Update the target Q-values for the actions taken

            states.append(np.ravel(state).reshape(1,-1))
            Q_tables.append(np.ravel(Q_table).reshape(1,-1))

        # Train the model on the batch using the target Q-values as the target output
        self.model.train_on_batch(np.concatenate(states, axis=0), np.concatenate(Q_tables, axis=0))


    def train(self, SETTINGS, PARAMS, htype):
           
        # Run the simulation for the given range of episodes.
        for e in range(SETTINGS.episodes[0], SETTINGS.episodes[1]):
            print(f"\nEpisode: {e}")

            # Initialize the hospital. A distribution center is also initialized to provide the hospital with random supply.
            dc = Distribution_center(SETTINGS, e)
            hospital = Hospital(SETTINGS, htype, e)

            self.env.reset(PARAMS, dc, hospital)    # Reset the environment
            state = self.env.state
            day = 0
            total_reward = 0
            
            while day < (SETTINGS.init_days + SETTINGS.test_days):
                print(f"Day: {day}")
                
                action = self.select_action(state)    # Select an action using the Q-network's epsilon-greedy policy
                next_state, reward, next_day = self.env.step(SETTINGS, PARAMS, action, dc, hospital)   # Take the action and receive the next state, reward and next day
                self.memory.append([state, action, reward, next_state, day])    # Store the experience tuple in memory
                total_reward += reward

                # Update epsilon
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                state = next_state
                day = next_day
                
                if len(self.memory) >= self.batch_size:
                    self.update()    # Update the Q-network using a batch of experiences from memory
                
            print("Episode:", e+1, "Total Reward:", total_reward)
            self.memory = []    # Clear the memory buffer after each episode