# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
#lambda for poisson distribution
lamda_loc_A = 2
lamda_loc_B = 12
lamda_loc_C = 4
lamda_loc_D = 7
lamda_loc_E = 8


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [[0,0]]
        self.action_space += [list(ele) for ele in list(permutations([i for i in range(m)], 2))]  
        self.state_space = [[a,b,c] for a in range(m) for b in range(t) for c in range(d)]
        
        self.city = np.random.choice(np.arange(0,m-1))
        self.time = np.random.choice(np.arange(0,t-1))
        self.day = np.random.choice(np.arange(0,d-1))
        self.state_init = [self.city,self.time,self.day]

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        m_vector = [0 if i != state[0] else 1 for i in range(0,m)]
        t_vector = [0 if i != state[1] else 1 for i in range(0,t)]
        d_vector = [0 if i != state[2] else 1 for i in range(0,d)]
        #return np.array(m_vector+t_vector+d_vector)
        return m_vector+t_vector+d_vector


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(lamda_loc_A)
        if location == 1:
            requests = np.random.poisson(lamda_loc_B)
        if location == 2:
            requests = np.random.poisson(lamda_loc_C)
        if location == 3:
            requests = np.random.poisson(lamda_loc_D)
        if location == 4:
            requests = np.random.poisson(lamda_loc_E)



        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m ), requests) + [0]# (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        #Reward to move from current place to action start place + Reward to move from start place of action to end place
        current_city = state[0]
        current_time = state[1]
        current_day = state[2]
        start_city = action[0]
        end_city = action[1]
        '''
        Three Possible action Can be taken :
        1. No Ride taken - [0,0] 
        2. Current location/city is same as the pickup location/city 
        3. Current location/city is different from the pickup location/city
        ''' 
        if (start_city == 0 and end_city == 0):
            #Now calculate the reward for the ride: (just cost for 1Hr)
            reward = -C * 1
            
        elif (current_city == start_city) :
            hrs_to_drop =  Time_matrix[start_city][end_city][current_time][current_day]
            #Now calculate the reward for the ride: (revenue for hrs - cost for hrs)
            reward = (R * hrs_to_drop) - (C * hrs_to_drop)
            
        else:
            hrs_to_pick = Time_matrix[current_city][start_city][current_time][current_day]
            current_time = current_time + int(hrs_to_pick)
            
            if current_time > 23:
                current_day += current_time//24
                current_time = current_time%24
                current_day = current_day%7
            hrs_to_drop = Time_matrix[start_city][end_city][current_time][current_day]
            
            reward = (R * hrs_to_drop) - (C* (hrs_to_pick + hrs_to_drop))
            
        reward = int(reward)
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        current_city = state[0]
        current_time = state[1]
        current_day = state[2]
        start_city = action[0]
        end_city = action[1]
        total_hrs = 0
        reward = 0
        '''
        Three Possible action Can be taken :
        1. No Ride taken - [0,0] 
        2. Current location/city is same as the pickup location/city 
        3. Current location/city is different from the pickup location/city
        '''
        
        if (start_city == 0 and end_city == 0):
            next_state = [current_city,current_time,current_day]
            current_time = current_time +1
            if current_time > 23:
                current_day += current_time//24
                current_time = current_time%24
                current_day = current_day%7
            next_state = [current_city,current_time,current_day]
            total_hrs = 1
            reward = -C * 1
            
        elif (current_city == start_city) :
            hrs_to_drop = Time_matrix[start_city][end_city][current_time][current_day]
            total_hrs += int(hrs_to_drop)
            current_time = current_time + int(hrs_to_drop)
            if current_time > 23:
                current_day += current_time//24
                current_time = current_time%24
                current_day = current_day%7
            #current_city = end_city
            next_state = [end_city,current_time,current_day]
            reward = (R * hrs_to_drop) - (C * hrs_to_drop)
            
        else:
            hrs_to_pick = Time_matrix[current_city][start_city][current_time][current_day]
            #print("Hours to Pick is : ", hrs_to_pick)
            total_hrs += int(hrs_to_pick)
            current_time = current_time + int(hrs_to_pick)
            if current_time > 23:
                current_day += current_time//24
                current_time = current_time%24
                current_day = current_day%7
            #print("After PickUP time : ", current_time)
            hrs_to_drop = Time_matrix[start_city][end_city][current_time][current_day]
            total_hrs += int(hrs_to_drop)
            #print("Hours took to Drop : " , hrs_to_drop)
            current_time = current_time + int(hrs_to_drop)
            if current_time > 23:
                current_day += current_time//24
                current_time = current_time%24
                current_day = current_day%7
            next_state = [end_city,current_time,current_day]
            reward = (R * hrs_to_drop) - (C* (hrs_to_pick + hrs_to_drop))
            
        return next_state, reward ,total_hrs
    
    
    def reset(self):
        #self.state_init = random.choice(self.state_space)
        return self.action_space, self.state_space, self.state_init
