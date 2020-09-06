from MINER_STATE import State
import numpy as np
from random import randrange


def valid(x, y):
    return (x >= 0 and x <= 8 and y >= 0 and y <= 20)


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


class Bot5:
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
        self.initial_flag = True

    def myGetGoldAmount(self, x, y, initial_flag=False, are_we_here=False):
        gold_on_ground = self.state.mapInfo.gold_amount(x, y)
        if gold_on_ground == 0:
            return 0

        count_players = 0
        for player in self.state.players:
            if player["posx"] == x and player["posy"] == y:
                if "energy" in player:
                    if player["status"] == self.state.STATUS_PLAYING:
                        count_players += 1
                elif initial_flag:  # 0 step, initial state
                    count_players += 1
        if are_we_here:
            return gold_on_ground / count_players
        else:
            return gold_on_ground / (count_players + 1)  # +1 because assuming that we will come here

    def findLargestGold(self):
        my_bot_x, my_bot_y = self.info.posx, self.info.posy
        largest_gold_x = -1
        largest_gold_y = -1

        max_gold = -1
        for goal in self.state.mapInfo.golds:
            i = goal["posx"]
            j = goal["posy"]
            gold_amount = goal["amount"]
            if gold_amount > max_gold:
                largest_gold_x = i
                largest_gold_y = j
                max_gold = gold_amount
            elif gold_amount == max_gold:
                prev_distance = (largest_gold_x - my_bot_x) * (largest_gold_x - my_bot_x) + \
                                (largest_gold_y - my_bot_y) * (largest_gold_y - my_bot_y)
                new_distance = (i - my_bot_x) * (i - my_bot_x) + (j - my_bot_y) * (j - my_bot_y)
                if new_distance < prev_distance:
                    largest_gold_x = i
                    largest_gold_y = j
                    max_gold = gold_amount

        return largest_gold_x, largest_gold_y

    def findLargestGoldInSmallMap(self, des_x, des_y):
        x, y = self.info.posx, self.info.posy
        largest_gold_x = None
        largest_gold_y = None
        next_step_x = 0
        next_step_y = 0
        if x < des_x:
            next_step_x = 1
        else:
            next_step_x = -1
        if y < des_y:
            next_step_y = 1
        else:
            next_step_y = -1

        max_gold = 0
        while x != des_x + next_step_x:
            while y != des_y + next_step_y:
                if x != des_x or y != des_y:
                    gold_amount = self.myGetGoldAmount(x, y)
                    if gold_amount > 0:
                        if gold_amount > max_gold:
                            largest_gold_x = x
                            largest_gold_y = y
                            max_gold = gold_amount
                        elif gold_amount == max_gold:
                            prev_distance = (largest_gold_x - self.info.posx) * (largest_gold_x - self.info.posx) + \
                                            (largest_gold_y - self.info.posy) * (largest_gold_y - self.info.posy)
                            new_distance = (x - self.info.posx) * (x - self.info.posx) + (y - self.info.posy) * (
                                        y - self.info.posy)
                            if new_distance < prev_distance:
                                largest_gold_x = x
                                largest_gold_y = y
                                max_gold = gold_amount
                y += next_step_y
            y = self.info.posy
            x += next_step_x

        return largest_gold_x, largest_gold_y

    def getActionBaseOnEnergy(self, action_option_1, action_option_2):
        my_bot_x, my_bot_y = self.info.posx, self.info.posy
        n_action = action_option_1
        require_energy = 100

        if action_option_1 == self.ACTION_GO_RIGHT:
            next_x = my_bot_x + 1
        else:
            next_x = my_bot_x - 1

        if action_option_2 == self.ACTION_GO_DOWN:
            next_y = my_bot_y + 1
        else:
            next_y = my_bot_y - 1

        energy_1 = 1
        energy_2 = 1
        gold = self.state.mapInfo.gold_amount(next_x, my_bot_y)
        if gold > 0:
            energy_1 = 4
        gold = self.state.mapInfo.gold_amount(my_bot_x, next_y)
        if gold > 0:
            energy_2 = 4

        for obstacle in self.state.mapInfo.obstacles:
            i = obstacle["posx"]
            j = obstacle["posy"]
            if i == next_x and j == my_bot_y:
                if obstacle["type"] == 1:  # Tree
                    energy_1 = 20
                elif obstacle["type"] == 2:  # Trap
                    if obstacle["value"] == -10:
                        energy_1 = 10
                elif obstacle["type"] == 3:  # Swamp
                    energy_1 = -obstacle["value"]
            if i == my_bot_x and j == next_y:
                if obstacle["type"] == 1:  # Tree
                    energy_2 = 20
                elif obstacle["type"] == 2:  # Trap
                    if obstacle["value"] == -10:
                        energy_2 = 10
                elif obstacle["type"] == 3:  # Swamp
                    energy_2 = -obstacle["value"]

        if energy_1 < energy_2:
            n_action = action_option_1
            require_energy = energy_1
        else:
            n_action = action_option_2
            require_energy = energy_2

        if self.info.energy <= require_energy:
            n_action = self.ACTION_FREE

        # print("require_energy = {0}".format(require_energy))
        # print("choose action = {0}".format(n_action))

        return n_action

    def goToTarget(self, des_x, des_y):
        n_action = self.ACTION_FREE
        require_energy = 100
        my_bot_x, my_bot_y = self.info.posx, self.info.posy
        next_my_bot_x = my_bot_x
        next_my_bot_y = my_bot_y
        if my_bot_x == des_x:
            if my_bot_y < des_y:
                n_action = self.ACTION_GO_DOWN
                next_my_bot_y += 1
            else:
                n_action = self.ACTION_GO_UP
                next_my_bot_y -= 1
        elif my_bot_y == des_y:
            if my_bot_x < des_x:
                n_action = self.ACTION_GO_RIGHT
                next_my_bot_x += 1
            else:
                n_action = self.ACTION_GO_LEFT
                next_my_bot_x -= 1
        else:
            if my_bot_x < des_x:
                action_option_1 = self.ACTION_GO_RIGHT
            else:
                action_option_1 = self.ACTION_GO_LEFT

            if my_bot_y < des_y:
                action_option_2 = self.ACTION_GO_DOWN
            else:
                action_option_2 = self.ACTION_GO_UP

            n_action = self.getActionBaseOnEnergy(action_option_1, action_option_2)
            return n_action

        require_energy = 1
        gold_amount = self.state.mapInfo.gold_amount(next_my_bot_x, next_my_bot_y)
        if gold_amount > 0:
            require_energy = 4
        for obstacle in self.state.mapInfo.obstacles:
            i = obstacle["posx"]
            j = obstacle["posy"]
            if i == next_my_bot_x and j == next_my_bot_y:
                if obstacle["type"] == 1:  # Tree
                    require_energy = 20
                elif obstacle["type"] == 2:  # Trap
                    if obstacle["value"] == -10:
                        require_energy = 10
                elif obstacle["type"] == 3:  # Swamp
                    require_energy = -obstacle["value"]
        if self.info.energy <= require_energy:
            n_action = self.ACTION_FREE

        return n_action

    def next_action(self, initial_flag=False):
        #my_bot_x, my_bot_y = self.state.x, self.state.y
        my_bot_x, my_bot_y = self.info.posx, self.info.posy
        n_action = self.ACTION_FREE
        gold_on_ground = self.myGetGoldAmount(my_bot_x, my_bot_y, self.initial_flag, are_we_here=True)
        if self.initial_flag:
            self.initial_flag = False
        energy = self.info.energy

        if gold_on_ground > 0:
            if energy <= 5:
                n_action = self.ACTION_FREE
            elif energy > 37.5:
                n_action = self.ACTION_CRAFT
            elif energy > (gold_on_ground / 50) * 5:
                n_action = self.ACTION_CRAFT
            else:
                n_action = self.ACTION_FREE
        else:
            largest_gold_x, largest_gold_y = self.findLargestGold()
            target_x = largest_gold_x
            target_y = largest_gold_y
            while True:
                tmp_x, tmp_y = self.findLargestGoldInSmallMap(target_x, target_y)
                if (tmp_x is None) or (tmp_y is None):
                    break
                target_x = tmp_x
                target_y = tmp_y

            n_action = self.goToTarget(target_x, target_y)

        return n_action

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
        print("G_BOT", self.info.playerId, self.estWood, self.pEnergyToStep, self.pStepToGold, self.info.score,
              self.info.energy)
