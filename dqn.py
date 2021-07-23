import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

#定義神經網路框架
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(N_STATES,10) #將現在狀態(N_STATE)當作輸入值
        self.fc1.weight.data.normal_(0,0.1) #初始化
        self.out=nn.Linear(10,N_ACTIONS) #依輸入值決定後續動作
        self.out.weight.data.normal_(0,0.1) #初始化
    
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        action_value=self.out(x)
        return action_value

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.eval_net, self.target_net=Net(),Net() #使用同一神經網路,但參數不同，兩者間參數轉換產生更新效果
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # 記憶庫
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    #選取action
    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy(選擇較大的)
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]     # return the argmax
        else:   # random(隨機選取動作)
            action = np.random.randint(0, N_ACTIONS)
        return action #真正選去的action

    #存入記憶庫
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 更新記憶庫
        index = self.memory_counter % MEMORY_CAPACITY #超過記憶庫容量，蓋掉前面的重新給index
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        if self.learn_step_counter%TARGET_REPLACE_ITER==0: #走到一定步數，將eval_net的參數更新到target_net
            self.target_net.load_state_dict(self.eval_net.state_dict()) #eval_net是每一步都在更新，target_net在特定步樹更新
        
        # 批次訓練
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        q_eval=self.eval_net(b_s).gather(1,b_a) #輸入現在的狀態，就會生成所有動作的價值，並照當初所選的價值當作現在的q_eval
        q_next=self.target_net(b_s_).detach()
        q_target=b_r+GAMMA*q_next.max(1)[0] #下一步的價值加上reward，並選擇最大值(其index是0)
        loss=self.loss_func(q_eval,q_target)

        #將誤差反向傳遞回去
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn=DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_
    

