"""
@author: Xiao Ning
 Date: 2022 年 10 月 15 日
"""

# import libraries
import os
import random
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch.autograd import grad
import torch.nn as nn
from numpy import genfromtxt
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

seed = 43
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import pandas as pd
import itertools

PATH = 'sird_norm_simple' 


#Load the data
covid_data = genfromtxt('sird.csv', delimiter=',') #in the form of [t,S,I,D,R] #in the form of [t,S,I,D,R]


class DINN(nn.Module):
    def __init__(self, t, S_data, I_data, R_data, D_data): #[t,S,I,D,R]
        super(DINN, self).__init__()
        
        self.N = 59e6 #population size
        
        #for the time steps, we need to convert them to a tensor, a float, and eventually to reshape it so it can be used as a batch
        self.t = torch.tensor(t, requires_grad=True)
        self.t_float = self.t.float()
        self.t_batch = torch.reshape(self.t_float, (len(self.t),1)) #reshape for batch 

        #for the compartments we just need to convert them into tensors
        self.S = torch.tensor(S_data)
        self.I = torch.tensor(I_data)
        self.R = torch.tensor(R_data)
        self.D = torch.tensor(D_data)
    
        self.losses = [] # here I saved the model's losses per epoch

        #setting the parameters
        # self.beta_tilda = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        # self.gamma_tilda = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        # self.mu_tilda = torch.nn.Parameter(torch.randn(1, requires_grad=True))

        #find values for normalization
        self.S_max = max(self.S)
        self.I_max = max(self.I)
        self.R_max = max(self.R)
        self.D_max = max(self.D)
        
        self.S_min = min(self.S)
        self.I_min = min(self.I)
        self.R_min = min(self.R)
        self.D_min = min(self.D)
        
        #normalize
        self.S_hat = (self.S - self.S_min) / (self.S_max - self.S_min)
        self.I_hat = (self.I - self.I_min) / (self.I_max - self.I_min)
        self.R_hat = (self.R - self.R_min) / (self.R_max - self.R_min)
        self.D_hat = (self.D - self.D_min) / (self.D_max - self.D_min)
                

        #matrices (x4 for S,I,D,R) for the gradients
        self.m1 = torch.zeros((len(self.t), 4)); self.m1[:, 0] = 1
        self.m2 = torch.zeros((len(self.t), 4)); self.m2[:, 1] = 1
        self.m3 = torch.zeros((len(self.t), 4)); self.m3[:, 2] = 1
        self.m4 = torch.zeros((len(self.t), 4)); self.m4[:, 3] = 1

        #NN
        self.net_sird = self.Net_sird()
        # self.params = list(self.net_sird.parameters())
        # self.params.extend(list([self.beta_tilda, self.gamma_tilda, self.mu_tilda]))
        self.net_params = self.Net_params()
        self.params = itertools.chain(self.net_sird.parameters(), self.net_params.parameters())

    #force parameters to be in a range
    # @property
    # def beta(self):
    #     return 0.5 * (torch.tanh(self.beta_tilda + 1))

    # @property
    # def gamma(self):
    #     return 0.5 * (torch.tanh(self.gamma_tilda + 1))
    
    # @property
    # def mu(self):
    #     return 0.05 * (torch.tanh(self.mu_tilda + 1))  

    class Net_sird(nn.Module): # input = [[t1], [t2]...[t100]] -- that is, a batch of timesteps 
        def __init__(self):
            super(DINN.Net_sird, self).__init__()

            self.fc1=nn.Linear(1, 20) #takes 100 t's
            self.fc2=nn.Linear(20, 20)
            self.fc3=nn.Linear(20, 20)
            self.fc4=nn.Linear(20, 20)
            self.out=nn.Linear(20, 4) #outputs S, I, D, R (100 S, 100 I, 100 D, 100 R --- since we have a batch of 100 timesteps)

        def forward(self, t_batch):
            sird=F.relu(self.fc1(t_batch))
            sird=F.relu(self.fc2(sird))
            sird=F.relu(self.fc3(sird))
            sird=F.relu(self.fc4(sird))
            sird=self.out(sird)
            return sird

    class Net_params(nn.Module): # input = [[t1], [t2]...[t100]] -- that is, a batch of timesteps 
        def __init__(self):
            super(DINN.Net_params, self).__init__()

            self.fc1=nn.Linear(1, 20) #takes 100 t's
            self.fc2=nn.Linear(20, 20)
            self.fc3=nn.Linear(20, 20)
            self.out=nn.Linear(20, 3) #outputs S, I, D, R (100 S, 100 I, 100 D, 100 R --- since we have a batch of 100 timesteps)

        def forward(self, t_batch):
            sird = F.relu(self.fc1(t_batch))
            sird = F.relu(self.fc2(sird))
            sird = 0.5 * (torch.tanh(self.fc3(sird)) + 1)
            # sird=self.out(sird)
            return sird


    def net_f(self, t_batch):
            
            #pass the timesteps batch to the neural network
            sird_hat = self.net_sird(t_batch)
            params_hat = self.net_params(t_batch)

            beta = params_hat[:,0]
            gamma = params_hat[:,1]
            mu = params_hat[:,2]
            
            #organize S,I,D,R from the neural network's output -- note that these are normalized values -- hence the "hat" part
            S_hat, I_hat, R_hat, D_hat = sird_hat[:,0], sird_hat[:,1], sird_hat[:,2], sird_hat[:,3]

            #S_t
            sird_hat.backward(self.m1, retain_graph=True)
            S_hat_t = self.t.grad.clone()
            self.t.grad.zero_()

            #I_t
            sird_hat.backward(self.m2, retain_graph=True)
            I_hat_t = self.t.grad.clone()
            self.t.grad.zero_()

            #R_t
            sird_hat.backward(self.m3, retain_graph=True)
            R_hat_t = self.t.grad.clone()
            self.t.grad.zero_() 

             #D_t
            sird_hat.backward(self.m4, retain_graph=True)
            D_hat_t = self.t.grad.clone()
            self.t.grad.zero_()

            #unnormalize
            S = self.S_min + (self.S_max - self.S_min) * S_hat
            I = self.I_min + (self.I_max - self.I_min) * I_hat
            R = self.R_min + (self.R_max - self.R_min) * R_hat
            D = self.D_min + (self.D_max - self.D_min) * D_hat      
                    
            f1_hat = S_hat_t - (-(beta / self.N) * S * I)  / (self.S_max - self.S_min)
            f2_hat = I_hat_t - ((beta / self.N) * S * I - gamma * I - mu * I ) / (self.I_max - self.I_min)
            f3_hat = R_hat_t - (gamma * I ) / (self.R_max - self.R_min)   
            f4_hat = D_hat_t - (mu * I) / (self.D_max - self.D_min)
     
            return f1_hat, f2_hat, f3_hat, f4_hat, S_hat, I_hat, R_hat, D_hat, beta, gamma, mu

	
    def train(self, n_epochs):
        # train
        print('\nstarting training...\n')
        
        for epoch in range(n_epochs):
            # lists to hold the output (maintain only the final epoch)
            S_pred_list = []
            I_pred_list = []
            R_pred_list = []
            D_pred_list = []
    
            # we pass the timesteps batch into net_f
            f1, f2, f3, f4, S_pred, I_pred, R_pred, D_pred, beta, gamma, mu = self.net_f(self.t_batch) # net_f outputs f1_hat, f2_hat, f3_hat, f4_hat, S_hat, I_hat, D_hat, R_hat
            
            self.optimizer.zero_grad() #zero grad
            
            #append the values to plot later (note that we unnormalize them here for plotting)
            S_pred_list.append(self.S_min + (self.S_max - self.S_min) * S_pred)
            I_pred_list.append(self.I_min + (self.I_max - self.I_min) * I_pred)
            R_pred_list.append(self.R_min + (self.R_max - self.R_min) * R_pred)
            D_pred_list.append(self.D_min + (self.D_max - self.D_min) * D_pred)

            #calculate the loss --- MSE of the neural networks output and each compartment
            loss = (torch.mean(torch.square(self.S_hat - S_pred))+ 
                    torch.mean(torch.square(self.I_hat - I_pred))+
                    torch.mean(torch.square(self.D_hat - D_pred))+
                    torch.mean(torch.square(self.R_hat - R_pred))+
                    torch.mean(torch.square(f1))+
                    torch.mean(torch.square(f2))+
                    torch.mean(torch.square(f3))+
                    torch.mean(torch.square(f4))
                    ) 

            loss.backward()
            self.optimizer.step()
            self.scheduler.step() 

            # append the loss value (we call "loss.item()" because we just want the value of the loss and not the entire computational graph)
            self.losses.append(loss.item())

            if epoch % 1000 == 0:          
                print('\nEpoch ', epoch, loss.item())

                print('#################################')          

        print('beta: (goal 0.25 ', beta)
        print('gamma: (goal 0.05 ', gamma)
        print('mu: (goal 0.003 ', mu)      

        return S_pred_list, I_pred_list, R_pred_list, D_pred_list
# time

dinn = DINN(covid_data[0], covid_data[1], covid_data[2], covid_data[3], 
            covid_data[4]) #in the form of [t,S,I,R,D]

learning_rate = 1e-6
optimizer = optim.Adam(dinn.params, lr = learning_rate)
dinn.optimizer = optimizer

scheduler = torch.optim.lr_scheduler.CyclicLR(dinn.optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=1000, mode="exp_range", gamma=0.85, cycle_momentum=False)

dinn.scheduler = scheduler

S_pred_list, I_pred_list, R_pred_list, D_pred_list = dinn.train(50000) #train

plt.plot(dinn.losses[0:], color = 'teal')
plt.xlabel('Epochs')
plt.ylabel('Loss'),

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.set_facecolor('xkcd:white')

ax.plot(covid_data[0], covid_data[1], 'blue', alpha=0.95, lw=2, label='Susceptible')
ax.plot(covid_data[0], S_pred_list[0].detach().numpy(), 'blue', alpha=0.9, lw=2, label='Susceptible Prediction', linestyle='dashed')

ax.plot(covid_data[0], covid_data[2], 'red', alpha=0.95, lw=2, label='Infected')
ax.plot(covid_data[0], I_pred_list[0].detach().numpy(), 'red', alpha=0.9, lw=2, label='Infected Prediction', linestyle='dashed')

ax.plot(covid_data[0], covid_data[3], 'green', alpha=0.95, lw=2, label='Recovered')
ax.plot(covid_data[0], R_pred_list[0].detach().numpy(), 'green', alpha=0.9, lw=2, label='Recovered Prediction', linestyle='dashed')

ax.plot(covid_data[0], covid_data[4], 'black', alpha=0.95, lw=2, label='Death')
ax.plot(covid_data[0], D_pred_list[0].detach().numpy(), 'black', alpha=0.9, lw=2, label='Death Prediction', linestyle='dashed')


ax.set_xlabel('Time /days')
ax.set_ylabel('Number')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='black', lw=0.2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()