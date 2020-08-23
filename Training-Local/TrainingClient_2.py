import sys
from DQN import DQN # A class of creating a deep q-learning model
from MinerEnv import MinerEnv # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
from Memory import Memory # A class of creating a batch in order to store experiences for the training process
import tensorflow as tf

import pandas as pd
import datetime
import numpy as np

#import matplotlib.pyplot as plt

from random import random

HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

# Create header for saving DQN learning file
now = datetime.datetime.now() #Getting the latest datetime
header = ["Ep", "Step", "Score", "Reward", "Episode_Reward", "Action", "Epsilon", "Done", "Termination_Code"] #Defining header for the save file
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)

# Parameters for training a DQN model
# N_EPISODE = 10000  # The number of episodes for training
N_EPISODE = 1000000  # The number of episodes for training
# MAX_STEP = 1000   #The number of steps for each episode
BATCH_SIZE = 256  #128 # or 256  #The number of experiences for each replay
MEMORY_SIZE = 1000000  # tang dan -->>>>  # The size of the batch for storing experiences
SAVE_NETWORK = 500  # After this number of episodes, the DQN model is saved for testing later.
INITIAL_REPLAY_SIZE = 21*9*6*3  # The number of experiences are stored in the memory batch before starting replaying
INPUT_SHAPE_1 = (21, 9, 7)  # The number of input values for the DQN model
INPUT_SHAPE_2 = (60,)
ACTION_NUM = 6  # The number of actions output from the DQN model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map

my_tensor = tf.Variable(0, dtype=tf.float32)  # initial value = 0

#tf.summary.scalar('episode avg_loss', my_tensor)
tf.summary.scalar('episode reward', my_tensor)
tf.summary.scalar('episode avg_reward', my_tensor)
tf.summary.scalar('episode goal', my_tensor)
tf.summary.scalar('episode total_steps', my_tensor)
tf.summary.scalar('episode epsilon', my_tensor)
merged_summary_op = tf.summary.merge_all()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'Logs/' + current_time
summary_writer = tf.summary.FileWriter(log_dir)

# Initialize a DQN model and a memory batch for storing experiences
DQNAgent = DQN(INPUT_SHAPE_1, INPUT_SHAPE_2, ACTION_NUM)
DQNAgent.update_target_model()
memory = Memory(MEMORY_SIZE)

# Initialize environment
minerEnv = MinerEnv(HOST, PORT) #Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
minerEnv.start()  # Connect to the game

train = False #The variable is used to indicate that the replay starts, and the epsilon starts decrease.
#Training Process
#the main part of the deep-q learning agorithm

for episode_i in range(0, N_EPISODE):
    try:
        # Choosing a map in the list
        mapID = np.random.randint(1, 6)  # Choosing a map ID from 5 maps in Maps folder randomly
        posID_x = np.random.randint(MAP_MAX_X)  # Choosing a initial position of the DQN agent on X-axes randomly
        posID_y = np.random.randint(MAP_MAX_Y)  # Choosing a initial position of the DQN agent on Y-axes randomly
        #Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
        request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100")
        #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
        minerEnv.send_map_info(request)

        # Getting the initial state
        minerEnv.reset()  # Initialize the game environment
        state_map, state_users = minerEnv.get_state()  # Get the state after reseting.
        # This function (get_state()) is an example of creating a state for the DQN model
        episode_reward = 0  # The amount of rewards for the entire episode
        terminate = False  # The variable indicates that the episode ends
        maxStep = minerEnv.state.mapInfo.maxStep  # Get the maximum number of steps for each episode in training
        score = 0  # Khoi added
        # episode_loss = 0
        step = 0
        # Start an episode for training
        for step in range(0, maxStep):
            if random() < DQNAgent.epsilon and minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) > 0:
                action = 5
            else:
                action = DQNAgent.act(state_map, state_users)  # Getting an action from the DQN model from the state (s)
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            reward = minerEnv.get_reward()  # Getting a reward
            new_state_map, new_state_users = minerEnv.get_state()  # Getting a new state
            terminate = minerEnv.check_terminate()  # Checking the end status of the episode

            # Add this transition to the memory batch
            memory.append(state_map, state_users, action, reward, new_state_map, new_state_users, terminate)

            # Sample batch memory to train network
            if memory.size > INITIAL_REPLAY_SIZE:
                # If there are INITIAL_REPLAY_SIZE experiences in the memory batch
                # then start replaying
                batch = memory.sample(BATCH_SIZE)  # Get a BATCH_SIZE experiences for replaying
                DQNAgent.replay(batch, BATCH_SIZE)  # Do relaying
                train = True  # Indicate the training starts
            episode_reward += reward  # Plus the reward to the total reward of the episode
            state_map = new_state_map  # Assign the next state for the next step.
            state_users = new_state_users  # Assign the next state for the next step.

            score = minerEnv.state.score

            # check again, when we need to save ?????????????????????????????????????????????????????
            # Saving data to file
            save_data = np.hstack(
                [episode_i + 1, step + 1, score, reward, episode_reward, action, DQNAgent.epsilon, terminate]).reshape(1, 8)
            with open(filename, 'a') as f:
                pd.DataFrame(save_data).to_csv(f, encoding='utf-8', index=False, header=False)

            if terminate:
                # If the episode ends, then go to the next episode
                break

        summary = tf.Summary()
        #summary.value.add(tag='episode avg_loss', simple_value=episode_loss/(step+1))
        summary.value.add(tag='episode reward', simple_value=episode_reward)
        summary.value.add(tag='episode agv_reward', simple_value=episode_reward/(step + 1))
        summary.value.add(tag='episode goal', simple_value=score)
        summary.value.add(tag='episode total steps', simple_value=step + 1)
        summary.value.add(tag='episode epsilon', simple_value=DQNAgent.epsilon)
        summary_writer.add_summary(summary, episode_i)
        summary_writer.flush()

        # check again ??????????????????????????????????????????????????????????
        # Iteration to save the network architecture and weights
        if np.mod(episode_i + 1, SAVE_NETWORK) == 0 and train == True:
            DQNAgent.update_target_model()  # Replace the learning weights for target model with soft replacement
            # Save the DQN model
            now = datetime.datetime.now()  # Get the latest datetime
            DQNAgent.save_model("TrainedModels/",
                                "DQNmodel_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode_i + 1))

        # Print the training information after the episode
        print('Episode %d ends. Number of steps is: %d. Accumulated Reward = %.4f. Score = %d. Epsilon = %.2f .Termination code: %d' % (
            episode_i + 1, step + 1, episode_reward, score, DQNAgent.epsilon, terminate))

        # Decreasing the epsilon if the replay starts
        if train is True and DQNAgent.epsilon > DQNAgent.epsilon_min:
            DQNAgent.update_epsilon()

    except Exception as e:
        import traceback

        traceback.print_exc()
        # print("Finished.")
        break
