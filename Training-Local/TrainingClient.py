import sys
from DQN import DQN # A class of creating a deep q-learning model
from MinerEnv import MinerEnv # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
from Memory import Memory # A class of creating a batch in order to store experiences for the training process
import tensorflow as tf

import pandas as pd
import datetime
import numpy as np
from random import random
#import time

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
N_EPISODE = 8000000  # The number of episodes for training
# MAX_STEP = 1000   #The number of steps for each episode
BATCH_SIZE = 64000  #128 # or 256  #The number of experiences for each replay
MEMORY_SIZE = 1000000  # tang dan -->>>>  # The size of the batch for storing experiences
SAVE_NETWORK = 5000  # After this number of episodes, the DQN model is saved for testing later.
INITIAL_REPLAY_SIZE = 64000 * 2  # The number of experiences are stored in the memory batch before starting replaying
INPUT_SHAPE_1 = (21, 9, 15)  # The number of input values for the DQN model
INPUT_SHAPE_2 = ((2 + 8 + 6) * 4 + 1,)
ACTION_NUM = 6  # The number of actions output from the DQN model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map

my_tensor = tf.Variable(0, dtype=tf.float32)  # initial value = 0
my_tensor_int = tf.Variable(0, dtype=tf.int32)  # initial value = 0


#tf.summary.scalar('time_append', my_tensor)
#tf.summary.scalar('time_take_samples', my_tensor)
#tf.summary.scalar('time_train', my_tensor)
tf.summary.scalar('episode avg_loss1', my_tensor)
tf.summary.scalar('episode avg_loss2', my_tensor)
tf.summary.scalar('episode reward', my_tensor)
tf.summary.scalar('episode avg_reward', my_tensor)
tf.summary.scalar('episode goal', my_tensor)
tf.summary.scalar('episode total_steps', my_tensor_int)
tf.summary.scalar('episode epsilon', my_tensor)
tf.summary.scalar('episode num_act left', my_tensor_int)
tf.summary.scalar('episode num_act right', my_tensor_int)
tf.summary.scalar('episode num_act up', my_tensor_int)
tf.summary.scalar('episode num_act down', my_tensor_int)
tf.summary.scalar('episode num_act relax', my_tensor_int)
tf.summary.scalar('episode num_act mine', my_tensor_int)
tf.summary.scalar('episode num of wrong relax', my_tensor_int)
tf.summary.scalar('episode num of wrong mining', my_tensor_int)
tf.summary.scalar('episode terminate playing', my_tensor_int)
tf.summary.scalar('episode terminate out of map', my_tensor_int)
tf.summary.scalar('episode terminate out of energy', my_tensor_int)
tf.summary.scalar('episode terminate others', my_tensor_int)
tf.summary.scalar('episode terminate out of golds', my_tensor_int)
tf.summary.scalar('episode terminate end step', my_tensor_int)
tf.summary.scalar('total step at episodes', my_tensor_int)
merged_summary_op = tf.summary.merge_all()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'Logs/' + current_time
summary_writer = tf.summary.FileWriter(log_dir)

#log_dir_2 = 'Logs/check_time_' + current_time
#summary_writer_time = tf.summary.FileWriter(log_dir_2)

# Initialize a DQN model and a memory batch for storing experiences
DQNAgent = DQN(INPUT_SHAPE_1, INPUT_SHAPE_2, ACTION_NUM, epsilon_decay=0.99999, epsilon_min=0.1)
DQNAgent.update_target_model()
memory = Memory(MEMORY_SIZE)
current_memory = Memory(32000)

# Initialize environment
minerEnv = MinerEnv(HOST, PORT) #Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
minerEnv.start()  # Connect to the game

train = False #The variable is used to indicate that the replay starts, and the epsilon starts decrease.
#Training Process
#the main part of the deep-q learning agorithm

total_step = 0
loss1 = 0
loss2 = 0

for episode_i in range(0, N_EPISODE):
    try:
        # Choosing a map in the list
        #mapID = np.random.randint(1, 13)  # Choosing a map ID from 12 maps in Maps folder randomly
        mapID = 1  # Choosing a map ID from 12 maps in Maps folder randomly

        posID_x = np.random.randint(MAP_MAX_X)  # Choosing a initial position of the DQN agent on X-axes randomly
        posID_y = np.random.randint(MAP_MAX_Y)  # Choosing a initial position of the DQN agent on Y-axes randomly
        #Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
        request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100")
        #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
        minerEnv.send_map_info(request)

        # Getting the initial state
        minerEnv.reset()  # Initialize the game environment
        state_map, state_users = minerEnv.get_state(100, initial_flag=True)  # Get the state after reseting.
        # This function (get_state()) is an example of creating a state for the DQN model
        episode_reward = 0  # The amount of rewards for the entire episode
        terminate = False  # The variable indicates that the episode ends
        maxStep = minerEnv.state.mapInfo.maxStep  # Get the maximum number of steps for each episode in training
        score = 0  # Khoi added
        # episode_loss = 0
        step = 0
        num_of_acts = [0, 0, 0, 0, 0, 0]
        terminate_list = [0, 0, 0, 0, 0, 0]
        num_of_wrong_relax = 0
        num_of_wrong_mining = 0

        # Start an episode for training
        for step in range(0, maxStep):
            total_step += 1
            #if random() < 0.8 \
            action = DQNAgent.act(state_map, state_users)  # Getting an action from the DQN model from the state (s)
            # stay at gold
            if minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) > 0:
                if random() < DQNAgent.epsilon:
                    if minerEnv.state.energy <= 5:
                        action = 4
                    else:
                        action = 5
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            reward, num_of_wrong_relax, num_of_wrong_mining = minerEnv.get_reward(num_of_wrong_relax, num_of_wrong_mining)  # Getting a reward
            new_state_map, new_state_users = minerEnv.get_state(100 - step - 1)  # Getting a new state
            terminate = minerEnv.check_terminate()  # Checking the end status of the episode
            
            #t1=0
            #t2=0
            #t3=0

            # Add this transition to the memory batch
            #tmp_t1 = time.time()
            current_memory.append(state_map, state_users, action, reward, new_state_map, new_state_users, terminate)
            memory.append(state_map, state_users, action, reward, new_state_map, new_state_users, terminate)
            #t1 = time.time() - tmp_t1

            num_of_acts[action] += 1

            episode_reward += reward  # Plus the reward to the total reward of the episode
            state_map = new_state_map  # Assign the next state for the next step.
            state_users = new_state_users  # Assign the next state for the next step.

            score = minerEnv.state.score
            """
            summary_2 = tf.Summary()
            summary_2.value.add(tag='time_append', simple_value=t1)
            summary_2.value.add(tag='time_take_samples', simple_value=t2)
            summary_2.value.add(tag='time_train', simple_value=t3)
            summary_writer_time.add_summary(summary_2, total_step)
            summary_writer_time.flush()
            """
            # check again, when we need to save ?????????????????????????????????????????????????????
            # Saving data to file
            #save_data = np.hstack(
            #    [episode_i + 1, step + 1, score, reward, episode_reward, action, DQNAgent.epsilon, terminate]).reshape(1, 8)
            #with open(filename, 'a') as f:
            #    pd.DataFrame(save_data).to_csv(f, encoding='utf-8', index=False, header=False)

            # Sample batch memory to train network
            if memory.size >= INITIAL_REPLAY_SIZE and np.mod(total_step, 32000) == 0:
                # If there are INITIAL_REPLAY_SIZE experiences in the memory batch
                # then start replaying
                #for i in range(2):
                batch1 = current_memory.sample(32000)
                hist1 = DQNAgent.replay(batch1, 32000)  # Do relaying
                loss1 = hist1.history['loss'][0]
                #loss1 = sum(loss1) / float(len(loss1))
                batch2 = memory.sample(BATCH_SIZE)  # Get a BATCH_SIZE experiences for replaying
                hist2 = DQNAgent.replay(batch2, BATCH_SIZE)  # Do relaying
                loss2 = hist2.history['loss'][0]
                #loss2 = sum(loss2) / float(len(loss2))
                train = True  # Indicate the training starts

            # check again ??????????????????????????????????????????????????????????
            # Iteration to save the network architecture and weights
            # if np.mod(episode_i + 1, SAVE_NETWORK) == 0 and train == True:
            if np.mod(total_step, 640000) == 0 and train == True:
                DQNAgent.update_target_model()  # Replace the learning weights for target model with soft replacement
                # Save the DQN model
                now = datetime.datetime.now()  # Get the latest datetime
                DQNAgent.save_model("TrainedModels/",
                                    "DQNmodel_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode_i + 1) + f'_map{mapID}')

            if terminate:
                # If the episode ends, then go to the next episode
                break

        terminate_list[minerEnv.state.status] = 1

        summary = tf.Summary()
        #summary.value.add(tag='episode avg_loss', simple_value=episode_loss/(step+1))
        summary.value.add(tag='episode reward', simple_value=episode_reward)
        summary.value.add(tag='episode agv_reward', simple_value=episode_reward/(step + 1))
        summary.value.add(tag='episode goal', simple_value=score)
        summary.value.add(tag='episode total steps', simple_value=step + 1)
        summary.value.add(tag='episode epsilon', simple_value=DQNAgent.epsilon)
        summary.value.add(tag='episode num_act left', simple_value=num_of_acts[0])
        summary.value.add(tag='episode num_act right', simple_value=num_of_acts[1])
        summary.value.add(tag='episode num_act up', simple_value=num_of_acts[2])
        summary.value.add(tag='episode num_act down', simple_value=num_of_acts[3])
        summary.value.add(tag='episode num_act relax', simple_value=num_of_acts[4])
        summary.value.add(tag='episode num_act mine', simple_value=num_of_acts[5])
        summary.value.add(tag='episode num of wrong relax', simple_value=num_of_wrong_relax)
        summary.value.add(tag='episode num of wrong mining', simple_value=num_of_wrong_mining)
        summary.value.add(tag='episode terminate playing', simple_value=terminate_list[0])
        summary.value.add(tag='episode terminate out of map', simple_value=terminate_list[1])
        summary.value.add(tag='episode terminate out of energy', simple_value=terminate_list[2])
        summary.value.add(tag='episode terminate others', simple_value=terminate_list[3])
        summary.value.add(tag='episode terminate out of golds', simple_value=terminate_list[4])
        summary.value.add(tag='episode terminate end step', simple_value=terminate_list[5])
        summary.value.add(tag='total step at episodes', simple_value=total_step)
        summary.value.add(tag='episode avg_loss1', simple_value=loss1)
        summary.value.add(tag='episode avg_loss2', simple_value=loss2)

        summary_writer.add_summary(summary, episode_i)
        summary_writer.flush()

        # Print the training information after the episode
        print('Episode %d ends. Number of steps is: %d. Accumulated Reward = %.4f. Score = %d. Epsilon = %.2f .Termination code: %d . Total step: %d' % (
            episode_i + 1, step + 1, episode_reward, score, DQNAgent.epsilon, terminate, total_step))

        # Decreasing the epsilon if the replay starts
        if train is True and DQNAgent.epsilon > DQNAgent.epsilon_min:
            DQNAgent.update_epsilon()

    except Exception as e:
        import traceback

        traceback.print_exc()
        # print("Finished.")
        break
