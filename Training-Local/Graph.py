import numpy as np
import sys

INF = 999999999

def valid (x,y):
    return (x>=0 and x <= 8 and y>=0 and y <=20)

delta = [[-1,0],[0,-1],[1,0],[0,1]]

LandID = 0 # -1
WoodID = 1 # [5,20]
TrapID = 2 ## 10
SwampID = 3 # 5 20 40 100
GoldID = -1 # 4

class Graph(): 
   
    def __init__(self, N,M): 

        #enegy
        self.boardMap = np.zeros((N,M))
        self.goldMap = np.zeros((N,M))

        self.V = N*M

        self.graph = [[0 for column in range(M)]  
                    for row in range(N)] 

        self.listTarget = []
        
        self.start = None
        self.state = None
        self.BFS_Value = []
        self.totalGold = 0
        self.currentEnergy = 0
        self.BFSFeatureMap = np.zeros((9,21))

    def convertToMap(self,state,estWood = 15,botInfo=None, isBot = False):

        #reset
        self.totalGold  = 0
        #reset 
        self.listTarget = []
        self.start = None
        self.state = None
        self.BFS_Value = []

        self.boardMap = np.zeros((9,21))
        self.goldMap = np.zeros((9,21))
        self.BFSFeatureMap = np.zeros((9,21))

        self.state = state
        
        if (isBot == True):
            posx, posy = botInfo.posy,botInfo.posx
            self.start = [botInfo.posy,botInfo.posx]
            self.currentEnergy = botInfo.energy
        else:
            posx, posy = self.state.y,self.state.x
            self.start = [self.state.y,self.state.x]
            self.currentEnergy = self.state.energy

        for ob in state.mapInfo.obstacles:
            x,y,value,typeX = ob["posx"],ob["posy"],ob["value"], ob["type"]
            self.boardMap[y,x] = - value

            if (value==0) :
                self.boardMap[y,x] = estWood

            
        #init boardMap
        for i in range(state.mapInfo.max_x + 1):
            for j in range(state.mapInfo.max_y + 1):
                goldAmount = state.mapInfo.gold_amount(i, j)
                if goldAmount > 0:
                    mahatanDist = (abs(posx-j) + abs(posy-i))
                    scoring = (((21*9)-mahatanDist) * goldAmount)//100
                    self.listTarget.append ([j,i,goldAmount,mahatanDist,scoring])
                    self.BFS_Value.append([INF,INF])

                    self.goldMap[j,i] = goldAmount
                    self.totalGold  += goldAmount
                    self.boardMap[j,i] = 4


        
        #print ("start", self.start)
        # print ("listTarget")
        # print (len(self.listTarget))
        # print(np.array(self.listTarget))
        #print ("BOardmap")
        #print (self.boardMap)

    def printSolution(self, dist): 
        print ("Vertex tDistance from Source") 
        for node in range(self.V): 
            print (node, "t", dist[node]) 
   
    
    def minDistance(self, dist, sptSet): 
        minDist = INF
        for v in range(self.V): 
            if dist[v] < minDist and sptSet[v] == False: 
                minDist = dist[v] 
                min_index = v 
   
        return min_index 


    def BFS(self):
        #for each target
        visited = np.full((9,21),False,dtype=bool) 
        #print (visited)
        steps = np.zeros((9,21))
        energy = np.ones((9,21)) * INF
        self.parent = np.ones((9,21,2)) * -1
        queue = [] 

        queue.append(self.start)
        visited[self.start[0],self.start[1]] = True #step
        energy[self.start[0],self.start[1]] = 0 #starting Energy
        energy[self.start[0],self.start[1]] = 0 #starting Energy
        self.parent[self.start[0],self.start[1]] = np.array([self.start[0],self.start[1]])
        count = 0
        while queue:
            current = queue.pop(0) 
            #print (current, end = " ") 

            cx = current[0]
            cy = current[1]
            visited[cx,cy] = True
            #print  ("Check ",cx,cy,self.boardMap[cx,cy])

            for d in delta:
                nx = (int)(cx + d[0])
                ny = (int) (cy + d[1])
                if (valid(nx,ny)):
                    #print ("tempCheck",nx,ny,visited[nx,ny], self.boardMap[nx,ny])
                    if ( visited[nx,ny] == False and self.boardMap[nx,ny] <= 40):
                        #print ("Add",nx,ny,visited[nx,ny])
                        
                        wastedEnergy = energy[cx,cy] + self.boardMap[nx,ny]

                        if (wastedEnergy <  energy[nx,ny]):
                            energy[nx,ny] = wastedEnergy
                            steps[nx,ny] = steps[cx,cy] + 1
                            visited[nx,ny] = True
                            self.parent[nx,ny] = np.array([cx,cy])
                            queue.append([nx,ny])
                        

                            for i in range (len(self.listTarget)):
                                if ( self.listTarget[i][0] == nx and self.listTarget[i][1] == ny ): # reach target
                                    if (wastedEnergy < self.BFS_Value[i][1]): #update by energy
                                        self.BFS_Value[i][0] = steps[nx,ny] #step
                                        self.BFS_Value[i][1] = wastedEnergy
                                        self.parent[nx,ny] = np.array([cx,cy])
                                        count +=1
                                #     print("Update",i,nx,ny)
                                # else :
                                #     print ("Ignore Update", wastedEnergy , energy[nx,ny])
                                #     print (energy[cx,cy] , self.boardMap[nx,ny])
                    # else:
                    #      print ("Ignore",nx,ny)
                    #      print ("Ignore by step", (int)(steps[nx,ny]))
                    #      print ("Ignore by boardMap", self.boardMap[nx,ny])
        #if (count==0):
        #    print (self.BFS_Value)
        #    print (self.listTarget)

    def getBFSResult(self, pEnergyToStep = 15, pStepToGold = 200):
        #alpha: coonvert enerrgy to free step
        #beta: convert step to gold
        #print ("")
        bestTarget = -1
        maxDist = -INF
        self.BFSFeatureMap = np.zeros((9,21))
        bx,by = -1,-1
        for i in range (len(self.listTarget)):
            x,y = self.listTarget[i][0],self.listTarget[i][1]
            
            energyToStep = self.BFS_Value[i][0] + ((self.BFS_Value[i][1]-self.currentEnergy)/pEnergyToStep)
            #energyToStep = self.BFS_Value[i][0] + ((self.BFS_Value[i][1])/pEnergyToStep)
            countPlayerAtGoldMine = 0
            for player in self.state.players:
                px,py = player['posx'],player['posy']
                if (px==y and py==x):
                    countPlayerAtGoldMine+= 1

            #HURISTIC FUNCTION
            #scoring  = ((-energyToStep*pStepToGold) + self.listTarget[i][2] - countPlayerAtGoldMine*energyToStep*50 )
            
            scoring  = ((-(4-countPlayerAtGoldMine)*energyToStep*pStepToGold)*0.2 + self.listTarget[i][2] - countPlayerAtGoldMine*energyToStep*50 )
            #scoring  = ((-energyToStep*pStepToGold) + self.listTarget[i][2] )
            #print ("Scoring",i, scoring)
            self.BFSFeatureMap[x,y] = scoring

            if (scoring > maxDist):
                maxDist = scoring
                bestTarget = i
                bx,by = x,y
        
        # print ("Starting",self.start)
        #print (maxDist, bx,by)
        #print (self.normalized(BFSFeatureMap)+1)
        # print ("---")
        
        #print ("Gold Target:",self.state.mapInfo.gold_amount(by, bx))

        return bestTarget


    def normalized(self, a, axis = -1, order = 2):

        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2==0] = 1
        return a / np.expand_dims(l2, axis)

    def getBFSFeatures(self):
        
        return self.normalized(self.BFSFeatureMap)


    def traceBack(self,targetID):
        cx, cy = self.listTarget[targetID][0],self.listTarget[targetID][1]
        '''
        if ( cx < 0 ):
            print (cx)
            print (err)
            print ("NEVER")
        '''
        while (  (int)(cx) != self.start[0] or (int)(cy) != self.start[1]):
            
            _cx, _cy = self.parent[(int)(cx),int(cy)]
            #print ( _cx, _cy)

            if (_cx == self.start[0] and _cy==self.start[1]):
                return cx,cy
            else :
                cx,cy = _cx,_cy
    
            '''
            if (cx < 0):
                print (cx)
                print (err)
                print ("NEVER")
            '''
         
        return -1,-1

    def dijkstra(self, src): 
       
        dist = [INF] * self.V 
        dist[src] = 0
        sptSet = [False] * self.V 
        for cout in range(self.V): 
            u = self.minDistance(dist, sptSet)
            sptSet[u] = True
            for v in range(self.V): 
                if self.graph[u][v] > 0 and  sptSet[v] == False and  dist[v] > dist[u] + self.graph[u][v]: 
                    dist[v] = dist[u] + self.graph[u][v] 
   
        self.printSolution(dist) 

