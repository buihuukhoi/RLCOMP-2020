from MINER_STATE import State
import numpy as np
#from Graph import Graph
from random import randrange

def valid (x,y):
    return (x>=0 and x <= 8 and y>=0 and y <=20)

class PlayerInfo:
    def __init__(self, id):
        self.playerId = id
        self.score = 0
        self.energy = 0
        self.posx = 0
        self.posy = 0
        self.lastAction = -1
        self.status = 0
        self.freeCount = 0 
        

class Bot3:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

    def __init__(self, id):
        self.state = State()
        self.info = PlayerInfo(id)

        self.isMovingInc = False

    def next_action(self):

        if (self.info.status!=0 and self.state.stepCount < 100):
            print ("WTF",self.info.status)

        countPlayerAtGoldMine = 0
        x, y= self.info.posx,self.info.posy
        #print (x,y)
        r_Action = self.ACTION_FREE #for safe

        if (self.isKeepFree ):
            self.isKeepFree = False
            return r_Action
        
        # 1st rule. Heighest Priority. Craft & Survive 
        if (valid(y,x)):
            
            goldOnGround =  self.state.mapInfo.gold_amount(x, y)
            countPlayerAtGoldMine = 1
            for player in self.state.players:
                px,py,pId = player['posx'],player['posy'],player['playerId']
                if (pId!=self.info.playerId):
                    if (px==x and py==y):
                        countPlayerAtGoldMine += 1


            if ( goldOnGround > 0 and countPlayerAtGoldMine >0):
                if ( goldOnGround > 0 and self.info.energy > 5):
                    r_Action = self.ACTION_CRAFT
            else :
                if (self.state.mapInfo.is_row_has_gold(y)):
                    ty = y
                    tx = -1
                    for dx in range (0,21,1):
                        if (self.state.mapInfo.gold_amount(dx,y) > 0):
                            tx = dx
                            break
                    if (tx>x):
                        r_Action = self.ACTION_GO_RIGHT
                    else :
                        r_Action = self.ACTION_GO_LEFT

                    #print ("found Gold ",tx,ty,self.state.mapInfo.gold_amount(tx,ty))
                else :
                    if (y == 8) :
                        self.isMovingInc = False
                    if (y == 0) :
                        self.isMovingInc = True

                    if (self.isMovingInc):
                        r_Action = self.ACTION_GO_DOWN
                    else :
                        r_Action = self.ACTION_GO_UP



        else :
            print ("INVALID WTF")

        if (r_Action == self.ACTION_CRAFT and self.info.energy < 5):
            r_Action = self.ACTION_FREE

        safeEnergy = 20*(1+randrange(0,1))
        if (self.info.energy < safeEnergy):
            r_Action = self.ACTION_FREE
        
        return r_Action
    
    def new_game(self, data):
        try:
            self.isKeepFree = False
            self.state.init_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def new_state(self, data):
        # action = self.next_action();
        # self.socket.send(action)
        try:
            self.state.update_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def printInfo(self):
        print ("G_BOT",self.info.playerId,self.estWood,self.pEnergyToStep,self.pStepToGold,self.info.score,self.info.energy)