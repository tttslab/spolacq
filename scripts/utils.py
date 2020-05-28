# -*- coding: utf-8 -*-

import numpy as np
import torch
import random

seed = 3

random.seed(seed)

# Environment simulator class
class Env:
    def __init__(self, res_dict):
        self.res_dict = res_dict
        pass

    def reset(self):
        #self.potion_num = 9
        pass
    
    def feedback(self, action):
        """
        Action List
        "up", "down", "left", "right", "forward", "backward"
        """
        accum_num = [self.res_dict['up'],
                    self.res_dict['up'] + self.res_dict['down'],
                    self.res_dict['up'] + self.res_dict['down'] + self.res_dict['left'],
                    self.res_dict['up'] + self.res_dict['down'] + self.res_dict['left'] + self.res_dict['right'],
                    self.res_dict['up'] + self.res_dict['down'] + self.res_dict['left'] + self.res_dict['right'] + self.res_dict['forward'],
                    self.res_dict['up'] + self.res_dict['down'] + self.res_dict['left'] + self.res_dict['right'] + self.res_dict['forward'] + self.res_dict['backward']]

        x_change = 0
        y_change = 0
        z_change = 0
        
        
        probability = random.uniform(0, 1)

        if probability < 0.9:
            if action >= 0 and action < accum_num[0]:
                z_change += 1
            elif action >= accum_num[0] and action < accum_num[1]:
                z_change -= 1
            elif action >= accum_num[1] and action < accum_num[2]:
                x_change -= 1
            elif action >= accum_num[2] and action < accum_num[3]:
                x_change += 1
            elif action >= accum_num[3] and action < accum_num[4]:
                y_change -= 1
            elif action >= accum_num[4] and action < accum_num[5]:
                y_change += 1

        # rand 200
        # probability = random.uniform(0, 1)
        # print(probability)
        # if probability < 0.827:
        #     if action >= 0 and action < 9:
        #         z_change += 1
        #     elif action >= 9 and action < 57:
        #         z_change -= 1
        #     elif action >= 57 and action < 104:
        #         x_change -= 1
        #     elif action >= 104 and action < 189:
        #         x_change += 1
        #     elif action >= 189 and action < 215:
        #         y_change -= 1
        #     elif action >= 215 and action < 236:
        #         y_change += 1
            

        # if action >= 0 and action < 2:
        #     z_change += 1
        # elif action >= 2 and action < 10:
        #     z_change -= 1
        # elif action >= 10 and action < 17:
        #     x_change -= 1
        # elif action >= 17 and action < 25:
        #     x_change += 1
        # elif action >= 25 and action < 33:
        #     y_change -= 1
        # elif action >= 33 and action < 37:
        #     y_change += 1
        return x_change, y_change, z_change


# Agent class
class Agent:
    def __init__(self):
        #self.action_space = ["attack", "potion", "durian", "dragonfruit", "lemon", "watermelon", "pear", "orange", "banana", "apple"]
        self.action_space = ["up", "up"]
        self.random_range = 22
        self.x = random.randint(-self.random_range, self.random_range)
        self.y = random.randint(-self.random_range, self.random_range)
        self.z = random.randint(-self.random_range, self.random_range)
        pass
    
    def reset(self):
        self.x = random.randint(-self.random_range, self.random_range)
        self.y = random.randint(-self.random_range, self.random_range)
        self.z = random.randint(-self.random_range, self.random_range)
        pass

    def get_state(self):
        state = np.asarray([self.x, self.y, self.z], dtype=np.float32)
        state = torch.from_numpy(state).reshape(1, 3)
        return state
    
    def evaluate_reward(self, x_change, y_change, z_change):
        # satisfaction level
        old_sl = -(self.x ** 2 + self.y ** 2 + self.z ** 2)
        self.x += x_change
        self.y += y_change
        self.z += z_change
        new_sl = -(self.x ** 2 + self.y ** 2 + self.z ** 2)
        reward = 1.0 * (new_sl - old_sl)
        if reward == 0:
            reward -= 10
        if self.x == 0 and self.y == 0 and self.z == 0:
            done = 1
        else:
            done = 0
        return reward, done

'''
# Environment simulator class
class Env:
    def __init__(self):
        #self.potion_num = 9
        pass

    def reset(self):
        #self.potion_num = 9
        pass
    
    def feedback(self, action):
        """
        Action List
        0: Cat
        1: Bed
        2: Tree
        3: Go
        4: Marvin
        5: House
        """
        house_change = 0
        w_change = 0
        emo_change = 0
        # Energy always decreases by 1.
        ener_change = 0

        if action == 0:
            # Cat
            w_change = -1
            emo_change = 2
            ener_change = -1
        elif action == 1:
            # Bed
            emo_change = 1
            ener_change = 3
        elif action == 2:
            # Tree
            w_change = 2
            emo_change = -1
            ener_change = -2
        elif action == 3:
            # Go
            emo_change = 1
            ener_change = -1
        elif action == 4:
            # Marvin
            w_change = -2
            emo_change = 1
            ener_change = 2
        elif action == 5:
            # House
            w_change = -3
            emo_change = 1
            house_change = 1
        
        ener_change -= 1
        day_change = 1
        return w_change, emo_change, ener_change, house_change, day_change


# Agent class
class Agent:
    def __init__(self):
        #self.action_space = ["attack", "potion", "durian", "dragonfruit", "lemon", "watermelon", "pear", "orange", "banana", "apple"]
        self.action_space = ["cat", "bed", "tree", "go", "marvin", "house", "aaa", "bbb", "Asdjaiodjoh"]
        self.emotion = 5
        self.wealth = 0
        self.energy = 5
        self.day = 0

        self.house = 0
        self.can_buy_house = 0
        #self.hp         = random.randint(1, 10)
        #self.coins      = 0
        pass
    
    def reset(self):
        #self.hp     = random.randint(1, 10)
        #self.coins  = 0
        self.emotion = 5
        self.wealth = 0
        self.energy = 5
        self.day = 0

        self.house = 0
        self.can_buy_house = 0
        pass

    def get_state(self):
        state = np.asarray([self.wealth, self.emotion, self.energy, self.can_buy_house], dtype=np.float32)
        state = torch.from_numpy(state).reshape(1, 4)
        return state
    
    def evaluate_reward(self, w_change, emo_change, ener_change, house_change, day_change):
        self.emotion += emo_change
        # The maximum value shouldn't exceed 5
        self.emotion = min(self.emotion, 5)

        self.wealth += w_change
        self.wealth = min(self.wealth, 5)

        self.energy += ener_change
        self.energy = min(self.energy, 5)

        self.day += day_change
        
        if self.wealth > 3:
            self.can_buy_house = 5
        else:
            self.can_buy_house = 0

        if self.emotion > 0 and self.energy > 0 and self.wealth >= 0 and self.day < 100:
            # The reward depends on if the agent is alive.
            done  = 0
            #reward = self.wealth / 3
            if house_change == 1:
                self.house += house_change
                done = 1
                #reward += done
        else:
            done = -1
            #reward = done

        reward = 100.0 * self.house + 10.0 * (3 * w_change + emo_change + ener_change) / 5 + 10.0 * done - self.day
        #if done == 0:
        #    reward += 0.5 * self.wealth
        #if done == 1 and house_change == 1:
        #    reward += (1000 - self.day) * 0.01 # Propotion of the importance of the time
        
        return reward, done
'''