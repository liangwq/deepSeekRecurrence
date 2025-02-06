import torch  
import torch.nn as nn  
import torch.optim as optim  

class PolicyNetwork(nn.Module):  
    """策略网络：输入状态，输出动作概率"""  
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=3):  
        super().__init__()  
        self.net = nn.Sequential(  
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),  
            nn.Linear(hidden_dim, output_dim)  
        )  
    
    def forward(self, x):  
        return torch.softmax(self.net(x), dim=-1)  

class ValueNetwork(nn.Module):  
    """价值网络：评估状态价值"""  
    def __init__(self, input_dim=256, hidden_dim=128):  
        super().__init__()  
        self.net = nn.Sequential(  
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),  
            nn.Linear(hidden_dim, 1)  
        )  
    
    def forward(self, x):  
        return self.net(x)  

class PPOAgent:  
    """PPO算法智能体"""  
    def __init__(self, state_dim=256, action_dim=3, lr=3e-4, gamma=0.99, clip_epsilon=0.2):  
        self.policy_net = PolicyNetwork(state_dim, action_dim)  
        self.value_net = ValueNetwork(state_dim)  
        self.optimizer = optim.Adam(  
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),  
            lr=lr  
        )  
        self.gamma = gamma          # 折扣因子  
        self.clip_epsilon = clip_epsilon  
    
    def get_action(self, state):  
        """根据策略网络选择动作"""  
        with torch.no_grad():  
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  
            probs = self.policy_net(state_tensor)  
            dist = torch.distributions.Categorical(probs)  
            action = dist.sample()  
        return action.item()  
    
    def update(self, batch):  
        """PPO算法更新策略和价值网络"""  
        # 解包批次数据（假设 batch 是一个包含 states, actions, rewards, next_states, dones 的元组）  
        states, actions, rewards, next_states, dones = batch  
        
        # 转换为张量  
        states = torch.FloatTensor(states)  
        actions = torch.LongTensor(actions)  
        rewards = torch.FloatTensor(rewards)  
        next_states = torch.FloatTensor(next_states)  
        dones = torch.FloatTensor(dones)  
        
        # 计算价值估计  
        values = self.value_net(states).squeeze()  
        next_values = self.value_net(next_states).squeeze()  
        
        # 计算Advantage  
        advantages = rewards + self.gamma * next_values * (1 - dones) - values  
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  
        
        # 旧策略概率  
        old_probs = self.policy_net(states).detach()  
        old_action_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze()  
        
        # 新策略概率  
        new_probs = self.policy_net(states)  
        new_action_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze()  
        
        # PPO损失  
        ratio = new_action_probs / (old_action_probs + 1e-8)  
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)  
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()  
        
        # 价值损失  
        value_loss = nn.MSELoss()(values, rewards + self.gamma * next_values * (1 - dones))  
        
        # 总损失  
        total_loss = policy_loss + 0.5 * value_loss  
        
        # 反向传播  
        self.optimizer.zero_grad()  
        total_loss.backward()  
        self.optimizer.step()  
        
        return total_loss.item()