import numpy as np  
import torch  
from collections import deque  
import random  
import json  
from datetime import datetime  


class ReplayBuffer:  
    def __init__(self, capacity=10000, save_path="data_collected.jsonl"):  
        self.buffer = deque(maxlen=capacity)  
        self.save_path = save_path  
        self.stats = {  
            "total_steps": 0,  
            "total_reward": 0.0,  
            "episode_rewards": []  
        }  
        # 初始化时清空旧数据  
        with open(self.save_path, "w") as f:  
            f.write("")  
    def __len__(self):  
        """返回缓冲区中当前存储的数据量"""  
        return len(self.buffer)  
    
    def add(self, state, action, reward, next_state, done):  
        """添加数据并实时保存到文件"""  
        # 确保状态是可序列化的  
        if isinstance(state, np.ndarray):  
            state = state.tolist()  
        elif hasattr(state, "__dict__"):  # 如果是自定义对象（如 ToTNode），将其转换为字典  
            state = vars(state)  
        
        if isinstance(next_state, np.ndarray):  
            next_state = next_state.tolist()  
        elif hasattr(next_state, "__dict__"):  # 如果是自定义对象（如 ToTNode），将其转换为字典  
            next_state = vars(next_state)  
        
        self.buffer.append((state, action, reward, next_state, done))  
        
        # 更新统计  
        self.stats["total_steps"] += 1  
        self.stats["total_reward"] += reward  
        if done:  
            self.stats["episode_rewards"].append(self.stats["total_reward"])  
            self.stats["total_reward"] = 0.0  
        
        # 转换为可读格式  
        data = {  
            "timestamp": datetime.now().isoformat(),  
            "state": state,  
            "action": int(action),  
            "reward": float(reward),  
            "next_state": next_state,  
            "done": bool(done)  
        }  
        
        with open(self.save_path, "a") as f:  
            f.write(json.dumps(data, ensure_ascii=False) + "\n") 
    
    def sample(self, batch_size):  
        """从缓存中随机采样一批数据"""  
        if len(self.buffer) < batch_size:  
            raise ValueError("缓冲区中的数据不足以采样指定大小的批次")  
        batch = random.sample(self.buffer, batch_size)  
        states, actions, rewards, next_states, dones = zip(*batch)  
        # 转换为 NumPy 数组  
        states = np.array(states)  
        actions = np.array(actions)  
        rewards = np.array(rewards)  
        next_states = np.array(next_states)  
        dones = np.array(dones)  
        return states, actions, rewards, next_states, dones  
    
    def print_stats(self):  
        """打印数据统计"""  
        avg_reward = np.mean(self.stats["episode_rewards"]) if self.stats["episode_rewards"] else 0  
        print(f"[统计] 总步数: {self.stats['total_steps']}, 平均每回合奖励: {avg_reward:.2f}")