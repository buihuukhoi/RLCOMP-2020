from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import sys
from keras.models import model_from_json
from MinerEnv import MinerEnv
import numpy as np

import tensorflow as tf
import logging
import os

from GMiner_Renderer import GMinerRenderer
import copy

trajectory = []

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5

HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

# load json and create model
json_file = open('DQNmodel_Test.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
DQNAgent = model_from_json(loaded_model_json)
# load weights into new model
DQNAgent.load_weights("DQNmodel_Test.h5")
print("Loaded model from disk")
status_map = {0: "STATUS_PLAYING", 1: "STATUS_ELIMINATED_WENT_OUT_MAP", 2: "STATUS_ELIMINATED_OUT_OF_ENERGY",
                  3: "STATUS_ELIMINATED_INVALID_ACTION", 4: "STATUS_STOP_EMPTY_GOLD", 5: "STATUS_STOP_END_STEP"}
try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game
    minerEnv.reset()
    state_map, state_users = minerEnv.get_state(initial_flag=True)  ##Getting an initial state
    while not minerEnv.check_terminate():
        try:
            if minerEnv.state.mapInfo.gold_amount(minerEnv.state.x, minerEnv.state.y) > 0:
                if minerEnv.state.energy <= 5:
                    action = 4
                else:
                    action = 5
            else:
                action = np.argmax(DQNAgent.predict({"state_map": state_map.reshape(1, 21, 9, 7),
                                                "state_users": state_users.reshape(1, (2 + 8 + 6) * 4)}))  # Getting an action from the trained model
            print("next action = ", action)
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            trajectory.append(copy.deepcopy(minerEnv.state))
            new_state_map, new_state_users = minerEnv.get_state()  # Getting a new state

            state_map = new_state_map  # Assign the next state for the next step.
            state_users = new_state_users  # Assign the next state for the next step.

        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Finished.")
            break
    print(status_map[minerEnv.state.status])
    print('Score = %d' % (minerEnv.state.score))
except Exception as e:
    import traceback
    traceback.print_exc()
print("End game.")
render = GMinerRenderer()
render.replay(trajectory, 0.1)
