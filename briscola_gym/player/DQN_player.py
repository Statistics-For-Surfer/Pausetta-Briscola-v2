from random import randint, random
import numpy as np
import torch
from briscola_gym.player.base_player import BasePlayer
from briscola_gym.game_rules import select_winner
from briscola_gym.player.network import Network
from game import BriscolaCustomEnemyPlayer

class DQN_Player(BriscolaCustomEnemyPlayer):

    def __init__(self, epsilon, gamma):
        super().__init__()
        self.epsilon = epsilon
        self.name = 'DQN_Player'
        self.n_actions = 3

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


        # Initialize both the target and policy networks
        self.q_net = Network(output_size=self.n_actions, lr=1e-5)

    def forward(self, x):
        # TODO
        '''
        Pass x in the qnetwork and get the output
        '''
        predicted_q = self.q_net(x)
        return predicted_q


    def act(self, state):
        '''
        Given the state choose an action to preform.
        Note that. If we are in the evaluation phase, I don't want to 
        use epilon but just take the actions from the network, instead if 
        we are training I want to use epsilon.
        '''

        # If we are not using epsilon this means we are in the evaluation phase
        # and I need to stack the last frames to have the correct state for 
        # the network
        if not self.use_epsilon:
            state = torch.tensor(state).float()

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
    
    

    def train(self, env, num_episodes):
        # Define the optimizer
        optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-5)
            
        for episode in range(num_episodes):
            # Reset the environment
            state = env.reset()
            done = False
            total_reward = 0
                
            while not done:
                # Choose an action
                action = self.act(state)
                # Take the action in the environment
                next_state, reward, done, _ = env.step(action)
                    
                # Update the total reward
                total_reward += reward

                # Store the transition in replay memory
                self.replay_memory.append((state, action, reward, next_state, done))
                    
                # Sample a minibatch from replay memory
                minibatch = random.sample(self.replay_memory, self.minibatch)
                    
                # Extract the components of the minibatch
                states, actions, rewards, next_states, dones = zip(*minibatch)
    
                # Convert the components to tensors
                states = torch.tensor(states).float()
                actions = torch.tensor(actions).long()
                rewards = torch.tensor(rewards).float()
                next_states = torch.tensor(next_states).float()
                dones = torch.tensor(dones).float()
                    
                # Compute the target Q-values
                target_q = rewards + self.gamma * torch.max(self.q_net(next_states), dim=1)[0] * (1 - dones)
                    
                # Compute the predicted Q-values
                predicted_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                    
                # Compute the TD error
                td_error = target_q - predicted_q
                    
                # Update the Q-network
                loss = torch.mean(td_error ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                # Update the state
                state = next_state
                
            # Decay epsilon
            if self.epsilon > self.target_epsilon:
                self.epsilon *= self.epsilon_decay
                
            # Print the episode results
            print(f"Episode {episode+1}: Total Reward = {total_reward}")
            # Train the DQN agent
agent = DQN_Player(epsilon=1, gamma=0.99)
env = BriscolaCustomEnemyPlayer()  # Replace with your environment
num_episodes = 2  # Replace with the desired number of episodes

agent.train(env, num_episodes)