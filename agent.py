############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import collections
from matplotlib import pyplot as plt



# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=150)
        self.layer_2 = torch.nn.Linear(in_features=150, out_features=150)
        self.output_layer = torch.nn.Linear(in_features=150, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output
    
    
# Creating a replay buffer
class ReplayBuffer:

    # Class initialiser
    def __init__(self):
        
        # Creating the buffer
        self.buffer = collections.deque(maxlen=15000)
        
        # Creating transaction utility buffer
        self.utility = collections.deque(maxlen=15000)
        
       
        
    # Function which returns the length of the buffer
    def get_length(self):
        return len(self.buffer)
    
    
    # Function that adds a transition to the buffer      
    def add_transition(self,transition,utility):
        
        self.buffer.append(transition)
        
        # Adding some probability to have a chance of picking any action
        self.utility.append(abs(utility)+0.0000001)
        
    
    # Sampling a minibatch from the buffer
    def get_minibatch(self,minibatch_size=100):
        
        # Calculating sampling probabilities
        utilities = list(self.utility)
        probs = utilities / np.sum(utilities)
        
        # Sampling the minibatch
        minibatch_indices = np.random.choice(self.get_length(),minibatch_size,replace=False,p=probs)
        
        minibatch = [self.buffer[i] for i in minibatch_indices]
        
        return minibatch
    
    # Checking if the buffer has enough transitions ready to be sampled
    def buffer_ready(self,num):
        
        if num <= self.get_length():
            return True
        
        else:
            
            return False
        
 

# The DQN class determines how to train the above neural network.
class DQN(ReplayBuffer):

    # The class initialisation function.
    def __init__(self,gamma=0.9,lr=0.001,minibatch_size=100):
        # Initialising ReplayBuffer
        ReplayBuffer.__init__(self)
        
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=3)
           
        # Target network
        self.target_network = Network(input_dimension=2, output_dimension=3)
        
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialising discount factor parameter
        self.gamma = gamma
        
        # Setting minibatch size
        self.minibatch_size = minibatch_size

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self,stop_training,testing_period):
        
        # If the minibatch is full enough to be sampled
        if self.buffer_ready(self.minibatch_size):
        
            # Set all the gradients stored in the optimiser to zero.
            self.optimiser.zero_grad()
            
            # Generate a minibatch of transitions
            minibatch = self.get_minibatch(self.minibatch_size)  
            
            # Calculate the loss for this transition.
            loss = self._calculate_loss(minibatch)
            
            # While the training is not stopped
            if((not stop_training)&(not testing_period)):
                
                # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
                loss.backward()
                
                # Take one gradient step to update the Q-network.
                self.optimiser.step()
            
            # Return the loss as a scalar
            return loss.item()
        
        else:
                       
            return np.NaN
        
        
      
    # Function that is called when we want to predict the Q values on the trained network. Inputs and outputs are numpy type
    def predict_q_network(self,X):
        
        # Convert input into pytorch tensor
        X_tensor = torch.tensor(X,dtype=torch.float32)
            
        # Do a forward pass given the input
        Q = self.q_network.forward(X_tensor)            
        
        # Converting predictions to numpy array
        Q_np = Q.detach().numpy()
            
        # Return the Q values
        return Q_np
    
    
    
    # Function that updates target network weights from the Q-network
    def update_target_weights(self):
        
        # Getting Q-network weights
        q_network_weights = self.q_network.state_dict()
        
        # Upload the weights to the target network
        self.target_network.load_state_dict(q_network_weights)
        
       

    # Function to calculate the loss for a minibatch
    def _calculate_loss(self, minibatch):
            
        # Calculating loss based on Bellman's equation
            
                
        # Acquiring all the data sets in seperate variables
        states,actions,rewards,next_states,d_actions = zip(*minibatch)
                    
                
        # Converting data sets into torch tensors
        next_states_tensor = torch.tensor(next_states,dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards,dtype=torch.float32).unsqueeze(1)
        states_tensor = torch.tensor(states,dtype=torch.float32)
        d_actions_tensor = torch.tensor(d_actions,dtype=torch.int64)
                   
        
        # Predicting for the current state on Q network and selecting Q values of the action taken
        predicted_Q_values = self.q_network.forward(states_tensor)
        network_prediction = predicted_Q_values.gather(dim=1,index=d_actions_tensor.unsqueeze(-1)).squeeze(0)
                    
        # Predicting Q values for the next state on target network
        predicted_Q_values_next_state = self.target_network.forward(next_states_tensor)
                    
        # Getting maximum Q values of the next state
        maxQ_next_state = predicted_Q_values_next_state.max(1)[0].unsqueeze(1)
                    
        # Calculating the loss
        loss = torch.nn.MSELoss()(rewards_tensor + self.gamma * maxQ_next_state , network_prediction)
            
        return loss
                
 

class Agent():

    # Function to initialise the agent
    def __init__(self):
        
        ####################
        # Hyper-Parameters #
        ####################
        
        
        # Initial episode length (decays to 100)
        self.episode_length = 850
        # Setting initial epsilon for e-greedy policy
        self.epsilon = 0.9
        # Setting gamma values for loss discount
        self.gamma = 0.97
        # Setting minibatch size
        self.minibatch_size = 100
        # Defining update rate of the target network
        self.K = 100
        # Number of exploration epochs
        self.exploration_period = 6000
        # Learning rate
        self.lr = 0.001
        # Plot the training info
        self.plot_data = True
        
        
        ####################
        #     Variables    #
        ####################
        
        # Initialising the Q-Network
        self.network = DQN(self.gamma,lr=self.lr,minibatch_size=self.minibatch_size)
        # Initialising possible continuous actions
        self.actions = [[0.0, 0.02],[0.02, 0.0],[0.0, -0.02]]
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # Number of steps in the episode
        self.current_step = 0
        # Current episode length
        self.current_episode_length = self.episode_length - (self.episode_length - 100)*(1-self.get_epsilon())
        # Flag to stop training
        self.stop_training = False
        # Num of episodes
        self.epochs = 0
        # Setting a testing epoch value
        self.testing_epoch = 0
        # Timer for the training termination
        self.testing_period = False
        
        ####################
        #   For Plotting   #
        ####################
        
        self.losses = [[]]
        self.rewards = [0]
        self.epsilons = [self.get_epsilon()]
        
        
       

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        
        # If adaptable epoch lenght was reached (finished)
        if (self.current_step > self.current_episode_length):
            
            
            # Plotting training info
            self.plot_info(self.losses,self.epsilons,self.rewards,self.plot_data)
            
            # Appending reward, epsilon and loss value FOR PLOTTING ONLY
            self.epsilons.append(self.get_epsilon())
            self.rewards.append(0)
            self.losses.append([])
            
            # Printing the message
            if(self.epochs == 0):
                print("Started training!")
            
            # Increasing the epoch number
            self.epochs += 1
            print("Epoch: {}".format(self.epochs))
            
                
            # Recalculating the next episode lenght
            self.current_episode_length = self.episode_length - (self.episode_length - 100)*(1-self.get_epsilon())
            # Reseting the loss and the step in the episode counter
            self.current_step = 0

            
            if(not self.stop_training):
                if(self.testing_period & ((self.testing_epoch + 2) == self.epochs)):
                    print("Testing period off!")
                    self.testing_period = False
            
            return True
        
        else:
            
            return False
    
    
 
    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        
        # Getting discrete action from e-greedy policy
        d_action = self.e_greedy_policy(state)
        # Converting to continuous action
        action = self.discrete_to_continuous_action(d_action)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state;
        self.state = state
        # Store the action;
        self.action = action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):    
        
        # If epsilon decayed, agent reached the goal, testing period is on and it is the next episode from the testing start
        if((self.get_epsilon() <= 0.005) & (distance_to_goal < 0.03) & self.testing_period & ((self.testing_epoch + 1) == self.epochs)):
            if(not self.stop_training):
                print("Training stopped!")
            # Stop training    
            self.stop_training = True 
            # Update the target network
            self.network.update_target_weights()
            
        # If epsilon decayed almost to zero and the agent is close to the goal
        if((self.get_epsilon() <= 0.005) & (distance_to_goal < 0.03) & (not self.testing_period)):
            # Save when the testing was started
            self.testing_epoch = self.epochs
            print("Testing period initiated!")
            # Begin testing period on full greedy policy
            self.testing_period = True
           
        
        # Not penalising unless the conditions below are met
        step_penalization = 0
        goal_bonus = 0
        
        # If the agent didnt change the state
        if (np.array_equal(self.state,next_state)):
            step_penalization = 0.05
            
        # Adding a bonus for reaching the end state
        if distance_to_goal < 0.03:
            goal_bonus = 0.1
        
        # Calculating the reward
        reward = 0.1*(1 - distance_to_goal) - step_penalization + goal_bonus
        
        # Appending reward FOR PLOTTING
        self.rewards[self.epochs] += (reward / self.current_episode_length)
        
        # Update current step in epoch
        self.current_step += 1
        # Create a transition
        transition = (self.state, self.action, reward, next_state, self.continuous_to_descrete_action(self.action))
        # Calculating the utility
        utility = abs(self.network._calculate_loss([transition])) 
        # Adding transition to the buffer
        self.network.add_transition(transition,utility.item())
        #Train the network
        loss = self.network.train_q_network(self.stop_training,self.testing_period)
        # Appending this episode's losses FOR PLOTTING
        self.losses[self.epochs].append(loss)
        
        # Update target network
        
        if ((self.num_steps_taken)%self.K == 0):
            self.network.update_target_weights()
           

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Getting discrete action from e-greedy policy
        d_action = self.e_greedy_policy(state,True)
        # Converting to continuous action
        action = self.discrete_to_continuous_action(d_action)
        return action
    
    
    # E-greedy policy
    def e_greedy_policy(self,state,greedy=False):
        
        # Performing full exploration during exploration period
        if (self.num_steps_taken < self.exploration_period):
             
             return  np.random.randint(0,3,1,dtype=int)[0]
                
        # Predicting Q_values based on the Q network
        Q_values = self.network.predict_q_network(state)
        
        # Selecting action with the highest Q value
        best_action = np.argmax(Q_values)
        
        # If greedy argument on or training stopped or epsilon very small
        if (greedy|self.stop_training|(self.get_epsilon()<0.005)):
            
            # Perform epsilon=0
            return best_action
            
        else:
            
            
            
            # Calculating the epsilon given the epoch to satisfy Robin-Monroe
            eps = self.get_epsilon()
        
            # Defining all equal probabilities
            greedy_policy = [eps/3,]*3
            
            # Defining best action probability
            greedy_policy[best_action] += 1 - eps
            
            # Normalising the probabilities
            greedy_policy /= np.sum(greedy_policy)
            
            # Calculating cumulative sum
            cumulative_prob_sum = np.cumsum(greedy_policy)
            
            # Picking a random number between 0 and 1
            random_number = np.random.random(1)
            
            # initialising action
            action = -1
            
            # Checking which action is picked
            for i in range(2,-1,-1):
            
                if random_number < cumulative_prob_sum[i]:
                    action = i
        
            return action
    
    
    ####################
    # Helper Functions #
    ####################
    
    # Function converting discrete action to continuous
    def discrete_to_continuous_action(self,d_action):
        
        # Initialising action as stay in the same place
        action = np.array([0.0, 0.0], dtype=np.float32)
        
        # Go up
        if (d_action == 0):
            action = np.array(self.actions[0], dtype=np.float32)
            
        # Go right    
        elif (d_action == 1):
            action = np.array(self.actions[1], dtype=np.float32)
            
        # Go down    
        elif (d_action == 2):
            action = np.array(self.actions[2], dtype=np.float32)
       
        return action
    
    
    # Function converting continuous action to discrete 
    def continuous_to_descrete_action(self,action):
        
        for d_action,c_action in enumerate(self.actions):
            
            if np.array_equiv(np.array(c_action,dtype=np.float32),action):
                return d_action
            
        d_action = np.random.randint(0,2,1,dtype=int)[0]
        
        return d_action
    
    
    # Function that calculates the adaptable epsilon value
    def get_epsilon(self):
        
        eps = self.epsilon**(1+0.0025*(self.num_steps_taken-self.exploration_period))
        eps = min(1,eps)
        eps = max(0,eps)
        
        return eps
    
    # Function plotting the training data
    def plot_info(self, losses, epsilons, rewards, show = False):
        plt.close()
        
        if show:
        
            fig, ax = plt.subplots(1,3, figsize = (15, 5))
    
            # Average loss of the episode plot
            mean, std = [], []
            for loss in losses:
                mean.append(np.nanmean(loss))
                std.append(np.nanstd(loss))
    
            ax[0].errorbar(range(len(mean)), mean, yerr = std, c='r', ecolor='b')
            ax[0].plot([], [], 'r', label='Mean')
            ax[0].plot([], [], 'b', label='STD')
            ax[0].set_xticks(range(0, len(mean), max(int(len(mean)/10), 1)))
            ax[0].set_xlabel('Episode')
            ax[0].set_ylabel("MSE value")
            ax[0].set_yscale("log")
            ax[0].set_title("Average network loss per episode")
            ax[0].legend()
    
            
    
            # Epsilon decay plot
            ax[1].plot(range(len(epsilons)), epsilons)
            ax[1].set_xticks(range(0, len(mean), max(int(len(mean)/10), 1)))
            ax[1].set_xlabel("Episode")
            ax[1].set_ylabel("Epsilon value")
            ax[1].set_title("Epsilon decay")
    
            # Total reward per episode plot
            ax[2].plot(range(len(rewards)), rewards)
            ax[2].set_xticks(range(0, len(mean), max(int(len(mean)/10), 1)))
            ax[2].set_xlabel('Episode')
            ax[2].set_ylabel('Average reward')
            ax[2].set_title('Average reward per episode')
    
            plt.tight_layout()
           
            plt.show()
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




    







































