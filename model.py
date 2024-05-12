import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

#nn.Module means neural network model
class Linear_QNet(nn.Module):
    #This creates two linear layer
    #It represent the dimensions of the input, hidden, and output layers of the neural network
    def __init__(self, input_size, hidden_size, output_size):
        #This line calls the constructor of the parent class (nn.Module) 
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    #This function performs linear transformation on the created linear layer
    #Here x is the tensor.
    def forward(self, x):
        #Rectified Linear Unit (ReLU) 
        #This line applies a linear transformation to the input x using the first linear layer (self.linear1)
        x = F.relu(self.linear1(x))
        #This layer typically produces the final output of the network, which may represent predictions, scores, probabilities, etc.
        x = self.linear2(x)
        return x
    
    #Saving the model's state dictionary allows you to persist the trained model's parameters to disk.
    #Saving the model enables you to reload it later without having to retrain from scratch.
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        #state_dict()==state dictionary
        torch.save(self.state_dict(), file_name)

#Necessary for training the neural network model , based on Q-learning algorithm
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        #The optimizer is responsible for updating the model's parameters during training to minimize the loss function.
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        #The loss function measures the discrepancy between the predicted outputs of the model and the ground truth labels or targets.
        self.criterion = nn.MSELoss()
        
    # Preprocessing on the input data 
    def train_step(self, state, action, reward, next_state, done):
        #Multi dimensional sample in the state
        # Converts the state variable into a PyTorch tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        #Usually Q-Learning deals with multiple samples
        #The code is designed to handle cases where the input tensors (state, next_state, action, reward) have a one-dimensional shape
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        #Neural model takes current state as input and predicts the q-value
        #This consists of predicted q values
        pred = self.model(state)
        
        # This line creates a copy of the predicted Q-values (pred) to serve as the target Q-values
        target = pred.clone()
        #This loop iterates over each sample in the batch.
        for idx in range(len(done)):
            # Initially, Q_new is set to the immediate reward received for the current action.
            Q_new = reward[idx]
            # If the next state is not a terminal state 
            # In the Q-learning algorithm, a state is considered "terminal" if it's the final state of an episode, 
            # meaning that no further actions can be taken from that state. 
            # If its not a terminal state , then we're extracting more values from to state .
            # the Q-value is updated based on the reward and the estimated future rewards.
            if not done[idx]:
                #Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) [Bellman Equation]
                # The updated Q-value (Q_new) is computed by adding the immediate reward (reward[idx]) 
                # to the discounted maximum predicted Q-value for the next state (self.model(next_state[idx])). 
                # The discount factor (self.gamma) is used to weigh the importance of future rewards.
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                
            #Finally, the target Q-value for the selected action in the current state is updated to the computed Q_new
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        
        #Loss function calculation
        #The goal is to minimize this loss, which represents the discrepancy between the predicted and target Q-values.
        
        #Setting the gradient to zero
        self.optimizer.zero_grad()
        #Calculation of loss between the new Q(predicted) and previous Q(target)
        loss = self.criterion(target, pred)
        #Back Propagation (calculation of the gradient of the loss)
        #Gradient = It provides information about the rate of change of the function at a specific point in parameter space.
        loss.backward()
        
        # It is a method call that updates the parameters of the model using the gradients 
        # computed during the backward pass (backpropagation).
        self.optimizer.step()



