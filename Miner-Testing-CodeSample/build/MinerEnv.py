import sys
import numpy as np
#from GAME_SOCKET_DUMMY import GameSocket  # in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from GAME_SOCKET import GameSocket
from MINER_STATE import State

TreeID = 1
TrapID = 2
SwampID = 3


class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()

        self.score_pre = self.state.score  # Storing the last score for designing the reward function
        self.energy_pre = self.state.energy
        #self.x_pre = self.state.x
        #self.y_pre = self.state.y

    def start(self):  # connect to server
        self.socket.connect()

    def end(self):  # disconnect server
        self.socket.close()

    def send_map_info(self, request):  # tell server which map to run
        self.socket.send(request)

    def reset(self):  # start new game
        try:
            message = self.socket.receive()  # receive game info from server
            self.state.init_state(message)  # init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action):  # step process
        self.socket.send(action)  # send action to server
        try:
            message = self.socket.receive()  # receive new state from server
            self.state.update_state(message)  # update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    def get_state(self, initial_flag=False):
        # update pre position, score, energy
        #self.x_pre = self.state.x
        #self.y_pre = self.state.y
        self.score_pre = self.state.score
        self.energy_pre = self.state.energy

        # depth = 3  # goal, min_energy, max_energy
        depth = 14  # goal, min_energy, max_energy, 4 player position
        goal_depth = 0
        min_energy_depth = 1
        max_energy_depth = 2
        my_agent_depth = 3
        bot1_depth = 4
        bot2_depth = 5
        bot3_depth = 6
        goal_pos = 7
        tree_pos = 8
        trap_pos = 9
        swamp_pos_5 = 10
        swamp_pos_20 = 11
        swamp_pos_40 = 12
        swamp_pos_100 = 13

        # len_player_infor = 6 * 4
        len_player_infor = 2 + 8 + 6

        # max_goal = 67 * 50 * 4  # assume 67 steps for mining and 33 steps for relaxing
        max_goal = 2000
        max_energy = 100

        # max_x = self.state.mapInfo.max_x
        # max_y = self.state.mapInfo.max_y
        max_player_energy = 50
        max_score = 3000
        # max_score = 67 * 50
        max_last_action = 6 + 1  # 1 because of None
        max_status = 5

        # Building the map
        view_1 = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1, depth], dtype=float)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                view_1[i, j, min_energy_depth] = -1 / max_energy
                view_1[i, j, max_energy_depth] = -1 / max_energy

                goal = self.state.mapInfo.gold_amount(i, j)
                if goal > 0:
                    view_1[i, j, goal_pos] = 1
                    view_1[i, j, min_energy_depth] = -4 / max_energy
                    view_1[i, j, max_energy_depth] = -4 / max_energy
                    view_1[i, j, goal_depth] = goal / max_goal

        for obstacle in self.state.mapInfo.obstacles:
            i = obstacle["posx"]
            j = obstacle["posy"]
            if obstacle["type"] == TreeID:  # Tree
                view_1[i, j, tree_pos] = 1
                view_1[i, j, min_energy_depth] = -5 / max_energy  # -5 ~ -20
                view_1[i, j, max_energy_depth] = -20 / max_energy  # -5 ~ -20
            elif obstacle["type"] == TrapID:  # Trap
                if obstacle["value"] != 0:
                    view_1[i, j, trap_pos] = 1
                view_1[i, j, min_energy_depth] = obstacle["value"] / max_energy
                view_1[i, j, max_energy_depth] = obstacle["value"] / max_energy
            elif obstacle["type"] == SwampID:  # Swamp
                view_1[i, j, min_energy_depth] = obstacle["value"] / max_energy  # -5, -20, -40, -100
                view_1[i, j, max_energy_depth] = obstacle["value"] / max_energy  # -5, -20, -40, -100
                if obstacle["value"] == -5:
                    view_1[i, j, swamp_pos_5] = 1
                elif obstacle["value"] == -20:
                    view_1[i, j, swamp_pos_20] = 1
                elif obstacle["value"] == -40:
                    view_1[i, j, swamp_pos_40] = 1
                elif obstacle["value"] == -100:
                    view_1[i, j, swamp_pos_100] = 1

        """
        for goal in self.state.mapInfo.golds:
            i = goal["posx"]
            j = goal["posy"]
            view_1[i, j, min_energy_depth] = 4 / max_energy
            view_1[i, j, max_energy_depth] = 4 / max_energy
            view_1[i, j, goal_depth] = goal["amount"] / max_goal
        """

        # Add player's information
        view_2 = np.zeros([len_player_infor * 4], dtype=float)

        index_player = 0

        if (0 <= self.state.x <= self.state.mapInfo.max_x) and \
                (0 <= self.state.y <= self.state.mapInfo.max_y):
            view_1[self.state.x, self.state.y, my_agent_depth] = 1
            view_2[index_player * len_player_infor + 0] = self.state.energy / max_player_energy
            view_2[index_player * len_player_infor + 1] = self.state.score / max_score
            if self.state.lastAction is None:  # 0 step
                view_2[index_player * len_player_infor + 2 + max_last_action] = 1
            else:  # > 1 step
                view_2[index_player * len_player_infor + 2 + self.state.lastAction] = 1
            view_2[index_player * len_player_infor + 2 + max_last_action + 1 + self.state.status] = 1

        bot_depth = my_agent_depth
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                index_player += 1
                bot_depth += 1
                if (0 <= player["posx"] <= self.state.mapInfo.max_x) and \
                        (0 <= player["posy"] <= self.state.mapInfo.max_y):
                    if "energy" in player:  # > 1 step
                        if player["status"] == self.state.STATUS_PLAYING:
                            view_1[player["posx"], player["posy"], bot_depth] = 1
                            view_2[index_player * len_player_infor + 0] = player["energy"] / max_player_energy
                            view_2[index_player * len_player_infor + 1] = player["score"] / max_score
                            view_2[index_player * len_player_infor + 2 + player["lastAction"]] = 1  # one hot
                            view_2[index_player * len_player_infor + 2 + max_last_action + 1 + player["status"]] = 1
                    elif initial_flag:  # 0 step, initial state
                        view_1[player["posx"], player["posy"], bot_depth] = 1
                        view_2[index_player * len_player_infor + 0] = 50 / max_player_energy
                        view_2[index_player * len_player_infor + 1] = 0 / max_score
                        view_2[index_player * len_player_infor + 2 + max_last_action] = 1  # one hot
                        view_2[index_player * len_player_infor + 2 + max_last_action + 1 + self.state.STATUS_PLAYING] = 1

        # Convert the DQNState from list to array for training
        DQNState_map = np.array(view_1)
        DQNState_users = np.array(view_2)

        return DQNState_map, DQNState_users

    def get_reward(self, num_of_wrong_relax, num_of_wrong_mining):
        # return -0.01 ~ 0.01
        # reward must target to mine goal

        max_reward = 50
        reward_died = -50  # ~ double max reward
        # reward_died = -25  # let a try

        reward_enter_goal = max_reward / 20  # 5

        # Calculate reward
        reward = 0  # moving, because agent will die at the max step

        energy_action = self.state.energy - self.energy_pre  # < 0 if not relax
        score_action = self.state.score - self.score_pre  # >= 0

        if score_action > 0:
            reward = score_action / 2500  # max ~2500 / episode
        else:
            # moving
            if int(self.state.lastAction) < 4:
                # enter gold
                if self.state.mapInfo.gold_amount(self.state.x, self.state.y) > 0:
                    reward = reward_enter_goal / 2500
            # mining but cannot get gold
            elif (int(self.state.lastAction) == 5) and (score_action == 0):
            #    reward = reward_died / 10 / max_reward
                num_of_wrong_mining += 1
            # relax when energy > 40 or cannot get more energy
            elif int(self.state.lastAction) == 4:
                if self.energy_pre > 40 or energy_action == 0:
            #        reward = reward_died / 10 / max_reward
                    num_of_wrong_relax += 1

            # at gold but move to ground
            # if (int(self.state.lastAction) < 4) and (self.state.mapInfo.gold_amount(self.x_pre, self.y_pre) > 0) \
            #        and (self.state.mapInfo.gold_amount(self.state.x, self.state.y) == 0):
            #    reward = reward_died

            # relax when energy > 40
            #elif self.energy_pre > 40 and int(self.state.lastAction) == 4:
            #    reward = reward_died / 4

            # relax but cannot get more energy
            #elif int(self.state.lastAction) == 4 and energy_action == 0:
            #    reward = reward_died / 4

        # If out of the map, then the DQN agent should be punished by a larger negative reward.
        #if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP or self.state.status == State.STATUS_ELIMINATED_INVALID_ACTION:
        #    reward = reward_died / max_reward

        #elif self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY or self.state.status == State.STATUS_STOP_EMPTY_GOLD \
        #        or self.state.status == State.STATUS_STOP_END_STEP:

        if self.state.status != State.STATUS_PLAYING:
            if self.state.score == 0:
                reward = reward_died / max_reward  # -1

        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP or self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward = reward_died / max_reward  # -1

        # print ("reward",reward)
        #return reward / max_reward / self.state.mapInfo.maxStep  # 100 steps
        return reward, num_of_wrong_relax, num_of_wrong_mining

    def check_terminate(self):
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
