from MINER_STATE import State
import numpy as np
from Graph import Graph
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
        

class Bot1:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

    def __init__(self, id,estWood=-1,pEnergyToStep=-1,pStepToGold=-1):
        self.state = State()
        self.info = PlayerInfo(id)

        if (estWood==-1): #random strenght 
            estWood         = (5 + randrange(16))
            pEnergyToStep   = (2 + randrange(9))* 5
            pStepToGold     = (1 + randrange(6) )*50

        self.estWood = estWood
        self.pEnergyToStep = pEnergyToStep
        self.pStepToGold  = pStepToGold
        #print ("AddG_BOT",estWood,pEnergyToStep,pStepToGold)

    def next_action(self):

        if (self.info.status!=0 and self.state.stepCount < 100):
            print ("WTF",self.info.status)

        countPlayerAtGoldMine = 0
        x, y= self.info.posx,self.info.posy
        r_Action = self.ACTION_FREE #for safe

        if (self.isKeepFree ):
            self.isKeepFree = False
            return r_Action
        
        # 1st rule. Heighest Priority. Craft & Survive 
        if (valid(y,x)):
            
            goldOnGround =  self.state.mapInfo.gold_amount(x, y)
            countPlayerAtGoldMine = 0
            for player in self.state.players:
                px,py = player['posx'],player['posy']
                if (px==x and py==y):
                    countPlayerAtGoldMine+= 1
            
            if ( goldOnGround > 0):
                if ( goldOnGround//countPlayerAtGoldMine > 0 and self.info.energy > 5):
                    r_Action = self.ACTION_CRAFT
            else :
                g = Graph(9,21)
                g.convertToMap(state = self.state, estWood =self.estWood, botInfo = self.info , isBot = True)
                g.BFS()
                target = g.getBFSResult(self.pEnergyToStep,self.pStepToGold)
                
                if (target==-1):
                   print ("NO TARGET")
                   return self.ACTION_FREE

                
                ny,nx = g.traceBack(target)
                ny,nx = int(ny), int (nx)
                
                typeOb = self.state.mapInfo.get_obstacle(nx,ny)
                
                nextTrap = g.boardMap[ny,nx]
                if (typeOb ==  1 ):    # WOOOD
                    nextTrap = 20

                if (  nextTrap >= self.info.energy  ):
                    r_Action = self.ACTION_FREE
                else:
                    if (ny == y):
                        if (nx > x):
                            r_Action=  self.ACTION_GO_RIGHT
                        elif (nx<x):
                            r_Action=  self.ACTION_GO_LEFT
                    else: #nx==x
                        if (ny > y):
                            r_Action=  self.ACTION_GO_DOWN
                        elif (ny < y):
                            r_Action=  self.ACTION_GO_UP

        else :
            print ("INVALID WTF")

        if (r_Action < 4 and self.info.energy <= 13  and self.state.stepCount < 90):
            self.isKeepFree = True
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
