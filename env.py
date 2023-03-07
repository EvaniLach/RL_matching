import gym
import numpy as np
import collections

from action_to_matches import *

# Define the environment for the reinforcement learning algorithm
class MatchingEnv(gym.Env):
    def __init__(self, SETTINGS, PARAMS, htype):

        self.num_bloodgroups = 2**len(PARAMS.major + PARAMS.minor)

        # Each state is a matrix of 2**len(antigens) × (35 + 7)
        # Vertical: all considered blood groups, each row index representing the integer representation of a blood group.
        # Horizontal: product age (0,1,...,34) + number of days until issuing (6,5,...,0)
        # Each cell contains the number of requests/products of that type
        self.state = np.zeros([self.num_bloodgroups, PARAMS.max_age + PARAMS.max_lead_time])

        # Each action is a matrix of 2**len(antigens) × inventory size
        # Vertical: all considered blood groups
        # Horizontal: number of products issued from that blood group (possibly in binary notation)
        I_size = SETTINGS.inv_size_factor_hosp * SETTINGS.avg_daily_demand[htype]
        # self.action_space = gym.spaces.Box(low=0, high=1, shape=[self.num_bloodgroups, np.ceil(np.log2(I_size))])   # binary
        self.action_space = gym.spaces.Box(low=0, high=1, shape=[self.num_bloodgroups, I_size])                       # real number, one-hot encoded

        # States whether the time horizon has been reached.
        # self.done = False
        self.day = 0


    def reset(self, PARAMS, dc, hospital):
        self.day = 0

        I = np.zeros([self.num_bloodgroups, PARAMS.max_age])
        I[:,0] = dc.sample_supply_single_day(PARAMS, len(I), hospital.inventory_size)

        R = hospital.sample_requests_single_day(PARAMS, [self.num_bloodgroups, PARAMS.max_lead_time], self.day)

        self.state = np.concatenate((I, R), axis=1)


    # action = |bloodgroups| × (|I| + 1)
    def step(self, SETTINGS, PARAMS, action, dc, hospital):

        ######################
        ## STATE AND ACTION ##
        ######################

        # List of all considered bloogroups in integer representation.
        bloodgroups = list(range(self.num_bloodgroups))

        # The inventory is represented by the left part of the state -> matrix of size |bloodgroups| × max age
        I = self.state[:,:PARAMS.max_age]
        
        # Create lists of blood groups in integer representation.
        inventory, issued_action, R = [], [], []
        for bg in bloodgroups:
            # All blood groups present in the inventory.
            inventory.extend([bg] * int(sum(I[bg])))

            # All inventory products from the action that should be issued today.
            # issued_action.extend([bg] * int(np.where(action[bg]==1)[0]))
            issued_action.extend([bg] * action[bg])

            # All requests from the state that need to be satisfied today.
            R.extend([bg] * int(self.state[bg,-1]))

        #####################
        ## GET ASSIGNMENTS ##
        #####################
        
        # Divide the 'issued' list in two, depending on whether they are actually present in the inventory.
        inv = collections.Counter(inventory)
        iss = collections.Counter(issued_action)
        nonexistent = list((iss - inv).elements())      # Products attempted to issue, but not present in the inventory.
        issued = list((inv & iss).elements())           # Products issued and available for issuing.
        # print("inventory:", inventory)
        # print("issued action:", issued_action)
        # print("action nonexistent:", nonexistent)
        # print("action issued:", issued)
        # print("requested:", self.state[:,PARAMS.max_age:])

        if len(R) > 0:
            # Assign all issued products to today's requests, first minimizing shortages, then minimizing the mismatch penalty.
            shortages, mismatches, assigned, discarded = action_to_matches(PARAMS, issued, R)
        else:
            shortages, mismatches, assigned = 0, 0, 0
            discarded = issued.copy()
        # print("shortages:", shortages)
        # print("mismatches:", mismatches)
        # print("assigned:", assigned)
        # print("discarded:", discarded)

        ######################
        ## CALCULATE REWARD ##
        ######################

        reward = 0
        reward -= 50 * len(nonexistent)              # Penalty of 50 for each nonexistent product that was attempted to issue.
        reward -= 10 * shortages                # Penalty of 10 for each shortage.
        reward -= 5 * mismatches                # Penalty of 5 multiplied by the total mismatch penalty, which is weighted according to relative immunogenicity.
        reward -= sum(I[:,PARAMS.max_age-1])    # Penalty of 1 for each outdated product.
        reward -= len(discarded)                     # Penalty of 1 for each discarded product (issued but not assigned).

        ######################
        ## GO TO NEXT STATE ##
        ######################

        # Increase the day count.
        self.day += 1

        # Remove all issued products from the inventory, where the oldest products are removed first.
        for bg in issued:
            I[bg, np.where(I[bg] > 0)[0][-1]] -= 1

        # Increase the age of all products (also removing all outdated products).
        I[:,1:PARAMS.max_age] = I[:,:PARAMS.max_age-1]

        # Return the number of products to be supplied, in order to fill the inventory upto its maximum capacity.
        I[:,0] = dc.sample_supply_single_day(PARAMS, len(I), max(0, hospital.inventory_size - int(sum(sum(I)))))

        # Remove all today's requests and increase lead time of all other requests.
        R = self.state[:,PARAMS.max_age:]
        R = np.insert(R[:, :-1], 0, values=0, axis=1)

        # Sample new requests
        R += hospital.sample_requests_single_day(PARAMS, R.shape, self.day)

        # Update the state with the updated inventory and requests.
        self.state = np.concatenate((I, R), axis=1)

        return self.state, reward, self.day