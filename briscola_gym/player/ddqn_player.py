from random import randint, random
import torch
from briscola_gym.player.base_player import BasePlayer
from briscola_gym.game_rules import select_winner
from briscola_gym.player.network import Network

class DDQN_Player(BasePlayer):

    def __init__(self, epsilon, gamma):
        super().__init__()
        self.epsilon = epsilon
        self.name = 'DDQN_Player'

        # Define all the variables
        self.minibatch = 32
        self.budget = 500_000

        # Add some exploration/exploitation using epsilon
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.target_epsilon = 0.01
        self.use_epsilon = False

        # Initial exploration don't change epsilon for the first steps
        self.exploration_start = 25_000

        # For the TD-error
        self.gamma = gamma

        # Keep track of the last 4 frames to stack them together tp use 
        # as input to the network
        self.state_stacked = [torch.zeros((84, 84)) for _ in range(4)]

        # Initialize both the target and policy networks
        self.q_net = Network(output_size=self.n_actions, lr=1e-5)
        self.target_net = Network(output_size=self.n_actions)


    def act(self, state):
        # TODO
        '''
        Given the state choose an action to preform. \n
        Note that. If we are in the evaluation phase, I don't want to 
        use epilon but just take the actions from the network, instead if 
        we are training I want to use epsilon
        '''

        # If we are not using epsilon this means we are in the evaluation phase
        # and I need to stack the last frames to have the correct state for 
        # the network
        if not self.use_epsilon:
            # Transform and append
            self.state_stacked = self.state_stacked[1:] + \
                                    [self.preprocess_states(state).squeeze()]
            # Stack into a tensor
            state = torch.stack(self.state_stacked)
            # Add batch dimension for the network
            state = torch.tensor(state).float().unsqueeze(0)

            # take action according to the network
            predicted_q = self.forward(state)
            action = torch.argmax(predicted_q).item()

            return action

        # If we are training and I sample a number under epsilon 
        if np.random.random() < self.epsilon:
            # take a random action
            action = np.random.choice(self.n_actions)
        else:
            # take action according to the network
            predicted_q = self.forward(state)
            action = torch.argmax(predicted_q).item()
        
        # return the action
        return action

