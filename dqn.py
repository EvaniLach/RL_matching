import tensorflow as tf
import gym
import random
import os

from dc import *
from hospital import *
from log import *

# Define the Deep Q-Learning algorithm.
class DQN:
    def __init__(self, SETTINGS, env):
        self.env = env                          # learning environment
        self.alpha = SETTINGS.alpha             # learning rate
        self.gamma = SETTINGS.gamma             # discount factor
        self.batch_size = SETTINGS.batch_size   # batch size for replay buffer
        
        # Initialize the NN for generating Q-matrices.
        self.model = self.build_model(SETTINGS)                  


    def load(self, SETTINGS, e, name=""):
        self.model = tf.keras.models.load_model(SETTINGS.generate_filename("models", e)+f"{name}")

    
    # Initialize the NN for generating Q-matrices.
    def build_model(self, SETTINGS):

        # Get the size of the input state and output action spaces
        input_size = len(np.ravel(self.env.state))          
        # output_size = self.env.action_space.shape[0]*self.env.action_space.shape[1]   # DAY-BASED
        output_size = self.env.action_space.shape[0]                                      # REQUEST-BASED  

        if SETTINGS.model_name == "M1":
            inputs = tf.keras.layers.Input(shape=(input_size,))
            layer_0 = tf.keras.layers.Dense(units=512, activation='relu')(inputs)
            layer_1 = tf.keras.layers.Dense(units=256, activation='relu')(layer_0)
            layer_2 = tf.keras.layers.Dense(units=128, activation='relu')(layer_1)
            outputs = tf.keras.layers.Dense(units=output_size, activation='sigmoid')(layer_2)
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        if SETTINGS.model_name == "M2":
            inputs = tf.keras.layers.Input(shape=(input_size,))
            layer_0 = tf.keras.layers.Dense(units=64, activation='relu')(inputs)
            layer_1 = tf.keras.layers.Dense(units=32, activation='relu')(layer_0)
            layer_2 = tf.keras.layers.Dense(units=16, activation='relu')(layer_1)
            outputs = tf.keras.layers.Dense(units=output_size, activation='sigmoid')(layer_2)
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        # elif SETTINGS.model_name == "M2":
        #     inputs = tf.keras.layers.Input(shape=(input_size,))
        #     layer_0 = tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu')(inputs)
        #     layer_1 = tf.keras.layers.MaxPooling1D(pool_size=2)(layer_0)
        #     layer_2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(layer_1)
        #     layer_3 = tf.keras.layers.MaxPooling1D(pool_size=2)(layer_2)
        #     layer_4 = tf.keras.layers.Dense(units=512, activation='relu')(layer_3)
        #     outputs = tf.keras.layers.Dense(units=output_size, activation='sigmoid')(layer_4)
        #     model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        # elif SETTINGS.model_name == "M3":
        #     inputs = tf.keras.layers.Input(shape=(input_size,))
        #     layer_0 = tf.keras.layers.LSTM(units=256, activation='sigmoid')(inputs)
        #     outputs = tf.keras.layers.Dense(units=output_size, activation='sigmoid')(layer_0)
        #     model = tf.keras.models.Model(inputs=inputs, outputs=outputs)


        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha))

        return model


    def select_action(self, state):                         # Method to select the action to take
        if np.random.rand() <= self.epsilon:                # Choose a random action with probability epsilon
            Q_values = self.env.action_space.sample()
            e = "exploration"
        else:                                               # Choose the action with the highest predicted Q-value
            # Predict the action given the state.
            Q_values = self.model.predict(np.ravel(state).reshape(1,-1), verbose=0)
            e = "exploitation"

        # For each row in the Q_matrix (for each blood group), put a 1 in the cell with the highest Q-value.
        # action = np.argmax(np.reshape(Q_values, self.env.action_space.shape), axis=1)   # DAY-BASED
        action = np.argmax(Q_values)        # REQUEST-BASED

        # if e == "exploitation":
        #     # print(f"requested: {binarray(state[np.where(state[:,-1]),-2], len(antigens))}")
        #     print(f"\nrequested: {np.where(state[:,-1])[0]}")
        #     print(f"Q-values: {Q_values}")
        #     print(f"action: {action}")
        #     print(f"num {action} in inventory: {sum(state[action,:35])}")
        
        return action


    # REQUEST-BASED
    def update(self):
        
        # Sample a batch of experiences from the model's memory.
        batch = random.sample(self.memory, self.batch_size)

        states = []
        Q_matrices = []

        for sample in batch:

            # Unpack the experience tuple.
            state, action, reward, next_state, _ = sample

            # Predict the Q-values for the next state.
            Q_next = self.model.predict(np.ravel(next_state).reshape(1,-1), verbose=0)
            # Get the maximum Q-value for the next state.
            max_Q_next = np.max(Q_next, axis=1)
            # Compute the target Q-values using the Bellman equation.
            Q_target = reward + (self.gamma * max_Q_next)

            # Predict the Q-values for the current state.
            Q_matrix = self.model.predict(np.ravel(state).reshape(1,-1), verbose=0)
            # Update the target Q-values for the actions taken.
            Q_matrix[:,action] = Q_target

            # Add the state and Q-matrix to the lists for training the model.
            states.append(np.ravel(state).reshape(1,-1))
            Q_matrices.append(np.ravel(Q_matrix).reshape(1,-1))

        # Train the model on the batch using the target Q-values as the target output.
        self.model.train_on_batch(np.concatenate(states, axis=0), np.concatenate(Q_matrices, axis=0))

    # REQUEST-BASED
    def test(self, SETTINGS, PARAMS):

        # Calculate the total number of days for the simulation.
        n_days = SETTINGS.init_days + SETTINGS.test_days

        # Run the simulation for the given range of episodes.
        for e in range(SETTINGS.episodes[0], SETTINGS.episodes[1]):
            print(f"\nEpisode: {e}")

            # Get the highest directory name
            dir_path = SETTINGS.generate_filename("models")
            highest_dir = max([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))], key=int)
            self.load(SETTINGS, highest_dir)

            # Start with an empty memory and initial epsilon.
            # self.epsilon = SETTINGS.epsilon
            self.epsilon = 0.0

            # Reset the environment.
            self.env.reset(SETTINGS, PARAMS, e, max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i]))
            # Initialize a dataframe to store the output of the simulation.
            df = initialize_output_dataframe(SETTINGS, PARAMS, self.env.hospital, e)

            # Set the current state and day to the environment's initial state and day.
            state = self.env.state
            day = self.env.day

            # Loop through each day in the simulation.
            while day < n_days:

                done = False
                todays_reward = 0

                # Write information about today's state to log file.
                df = self.env.log_state(PARAMS, df)

                # If there are no requests for today, proceed to the next day.
                if sum(self.env.state[:,-1]) == 0:
                    self.env.next_day(PARAMS)
                    done = True

                # If there are requests for today, loop through each request.
                while not done:
                    # Select an action using the model's epsilon-greedy policy.
                    action = self.select_action(state)
                    # Calculate the reward and update the dataframe.
                    reward, df = self.env.calculate_reward(SETTINGS, PARAMS, action, day, df)
                    todays_reward += reward
                    # Get the next state and whether the episode is done.
                    next_state, done = self.env.next_request(PARAMS)
                    # Update the current state to the next state.
                    state = next_state

                # Update the dataframe with the current day's information.
                df.loc[day,"logged"] = True
                print(f"Day {day}, reward {todays_reward}")

                # Update the model's epsilon value.
                df.loc[day,"epsilon current"] = self.epsilon

                # Save model and log file on predifined days.
                df.to_csv(SETTINGS.generate_filename("results", e)+".csv", sep=',', index=True)

                # Set the current day to the environment's current day.
                day = self.env.day


    def train_minrar(self, SETTINGS, PARAMS):

        minrar_dir = "C:/Users/Merel/Documents/Sanquin/Projects/RBC matching/Paper patient groups/blood_matching/"
        htype = max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i])[:3]

        # Run the simulation for the given range of episodes.
        for e in range(SETTINGS.episodes[0], SETTINGS.episodes[1]):
            print(f"\nEpisode: {e}")

            for part in range(5):
                print(f"part {part}")
                
                states = SETTINGS.unpickle(minrar_dir + f"NN_training_data/{htype}_{''.join(PARAMS.minor)}/states_{e}_{part}.pickle")
                Q_matrices = SETTINGS.unpickle(minrar_dir + f"NN_training_data/{htype}_{''.join(PARAMS.minor)}/Q_matrices_{e}_{part}.pickle")
            
                # Train the model on the batch using the target Q-values as the target output.
                # batch_indices = random.sample(range(len(states)), self.batch_size)
                # self.model.train_on_batch(states[np.ix_(batch_indices)], Q_matrices[np.ix_(batch_indices)])
                self.model.train_on_batch(states, Q_matrices)

                self.model.save(SETTINGS.generate_filename("models", e))


    def train(self, SETTINGS, PARAMS):
        # Calculate the total number of days for the simulation.
        n_days = SETTINGS.init_days + SETTINGS.test_days
        # Determine the days at which to save the model.
        model_saving_days = [day for day in range(n_days) if day % 100 == 0] + [n_days-1]

        # Run the simulation for the given range of episodes.
        for e in range(SETTINGS.episodes[0], SETTINGS.episodes[1]):
            print(f"\nEpisode: {e}")
            # If this isn't the first episode, load the previous episode's saved model.
            if e > 0:
                self.load(SETTINGS, e-1)

            # Start with an empty memory and initial epsilon.
            self.memory = []
            self.epsilon = SETTINGS.epsilon

            # Reset the environment.
            self.env.reset(SETTINGS, PARAMS, e, max(SETTINGS.n_hospitals, key = lambda i: SETTINGS.n_hospitals[i]))
            # Initialize a dataframe to store the output of the simulation.
            df = initialize_output_dataframe(SETTINGS, PARAMS, self.env.hospital, e)

            # Set the current state and day to the environment's initial state and day.
            state = self.env.state
            day = self.env.day

            # Loop through each day in the simulation.
            while day < n_days:

                # REQUEST-BASED
                done = False
                todays_reward = 0

                # Write information about today's state to log file.
                df = self.env.log_state(PARAMS, df, day)

                # If there are no requests for today, proceed to the next day.
                if sum(self.env.state[:,-1]) == 0:
                    self.env.next_day(PARAMS)
                    done = True

                # # DAY-BASED
                # action = self.select_action(state)    # Select an action using the Q-network's epsilon-greedy policy
                # next_state, reward, df = self.env.step(SETTINGS, PARAMS, action, dc, hospital, day, df)   # Take the action and receive the next state, reward and next day
                # self.memory.append([state, action, reward, next_state, day])    # Store the experience tuple in memory
                # state = next_state
                # # Update the Q-network using a batch of experiences from memory
                # if len(self.memory) >= self.batch_size: 
                #     self.update()

                # REQUEST-BASED 
                # If there are requests for today, loop through each request.
                while not done:
                    # Select an action using the model's epsilon-greedy policy.
                    action = self.select_action(state)
                    # Calculate the reward and update the dataframe.
                    reward, df = self.env.calculate_reward(SETTINGS, PARAMS, action, day, df)
                    todays_reward += reward
                    # Get the next state and whether the episode is done.
                    next_state, done = self.env.next_request(PARAMS)
                    # Store the experience tuple in memory.
                    if day >= SETTINGS.init_days:
                        self.memory.append([state, action, reward, next_state, day])
                    # Update the current state to the next state.
                    state = next_state

                # If there are enough experiences in memory, update the model.
                if len(self.memory) >= self.batch_size:
                    self.update()

                # Update the dataframe with the current day's information.
                df.loc[day,"logged"] = True
                print(f"Day {day}, reward {todays_reward}")

                # Update the model's epsilon value.
                df.loc[day,"epsilon current"] = self.epsilon
                self.epsilon = max(self.epsilon * SETTINGS.epsilon_decay, SETTINGS.epsilon_min)

                # Save model and log file on predifined days.
                if day in model_saving_days:
                    df.to_csv(SETTINGS.generate_filename("results", e)+".csv", sep=',', index=True)
                    self.model.save(SETTINGS.generate_filename("models", e))

                # Set the current day to the environment's current day.
                day = self.env.day
                


    # # DAY-DASED
    # def update(self):

    #     batch = random.sample(self.memory, self.batch_size)

    #     states = []
    #     Q_tables = []

    #     for sample in batch:

    #         state, action, reward, next_state, _ = sample

    #         # Compute the target Q-values
    #         Q_next = np.reshape(self.model.predict(np.ravel(next_state).reshape(1,-1), verbose=0), self.env.action_space.shape)             # Predict the Q-values for the next states
    #         max_Q_next = np.max(Q_next, axis=1)
    #         Q_target = reward + (self.gamma * max_Q_next)  # Compute the target Q-values using the Bellman equation

    #         # Compute the current Q-values
    #         Q_table = np.reshape(self.model.predict(np.ravel(state).reshape(1,-1), verbose=0), self.env.action_space.shape)             # Predict the Q-values for the next states
    #         Q_table[:,action] = Q_target # Update the target Q-values for the actions taken

    #         states.append(np.ravel(state).reshape(1,-1))
    #         Q_tables.append(np.ravel(Q_table).reshape(1,-1))

    #     # Train the model on the batch using the target Q-values as the target output
    #     self.model.train_on_batch(np.concatenate(states, axis=0), np.concatenate(Q_tables, axis=0))
