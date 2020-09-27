from GMiner_Renderer import GMinerRenderer

from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
import copy

import sys
from MinerEnv import MinerEnv
import numpy as np
from Graph import Graph
from random import randrange


HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5


def valid (x,y):
    return (x>=0 and x <= 8 and y>=0 and y <=20)

def next_action(stateX):
    countPlayerAtGoldMine = 0
    x, y= stateX.x,stateX.y
    returnAction = ACTION_FREE #for safe
    if (valid(y,x)):
        
        goldOnGround =  stateX.mapInfo.gold_amount(x, y)

        if (goldOnGround > 0):
            countPlayerAtGoldMine = 1
            for player in stateX.players:
                px,py = player['posx'],player['posy']
                if (px==x and py==y and stateX.id != player['playerId']):
                    countPlayerAtGoldMine+= 1

            goldPerPlayer = goldOnGround/countPlayerAtGoldMine
            
            if (goldPerPlayer >= 0 and stateX.energy > 5):
                returnAction = ACTION_CRAFT

        elif( stateX.energy <= 20):
            returnAction = ACTION_FREE            
        else :

            returnAction = randrange(4) #random action
            

    return returnAction
    


status_map = {0: "STATUS_PLAYING", 1: "STATUS_ELIMINATED_WENT_OUT_MAP", 2: "STATUS_ELIMINATED_OUT_OF_ENERGY",
               3: "STATUS_ELIMINATED_INVALID_ACTION", 4: "STATUS_STOP_EMPTY_GOLD", 5: "STATUS_STOP_END_STEP"}

action_map = {0: "GO_LEFT", 1: "GO_RIGHT", 2: "GO_UP",
                  3: "GO_DOWN", 4: "FREE", 5: "CRAFT", 6: "DIE CMNR"}



trajectory = []

try:
        # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game
    
    mapID = np.random.randint(1,6)
    posID_x = np.random.randint(21-1) #Choosing a initial position of the DQN agent on X-axes randomly
    posID_y = np.random.randint(9-1) #Choosing a initial position of the DQN agent on Y-axes randomly
    request = ("map" + str(mapID%5+1) + "," + str(posID_x) + "," + str(posID_y) + ",50,100" ) 
    print (request)
    minerEnv.send_map_info(request)
  
    originalState = minerEnv.reset()
    while not minerEnv.check_terminate():
        try:
            action = next_action(originalState)
            originalState = minerEnv.step(action)
            trajectory.append(copy.deepcopy(originalState))

        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Finished.")
            break
            
    print(minerEnv.state.score)
except Exception as e:
    import traceback
    traceback.print_exc()
print("End game.")

# replay game
renderer = GMinerRenderer(isSkipFrame= True)
renderer.replay(trajectory)