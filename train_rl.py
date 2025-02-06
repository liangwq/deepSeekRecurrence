import torch  
import os  
import json  
from core.replay_buffer import ReplayBuffer  
from agents.rl_agent import PPOAgent  
from tasks.math_solver import MathSolver  
from tqdm import tqdm  
from colorama import Fore, Style, init  

init(autoreset=True)  # 自动重置颜色  


def load_data_from_file(buffer, file_path="data_collected.jsonl"):  
    """从文件中加载数据到缓冲区"""  
    if not os.path.exists(file_path):  
        print(f"{Fore.RED}文件 {file_path} 不存在，无法加载数据！{Style.RESET_ALL}")  
        return False  
    
    with open(file_path, "r") as f:  
        for line in f:  
            data = json.loads(line.strip())  
            state = data["state"]  
            action = data["action"]  
            reward = data["reward"]  
            next_state = data["next_state"]  
            done = data["done"]  
            buffer.add(state, action, reward, next_state, done)  
    
    print(f"{Fore.GREEN}成功从文件 {file_path} 加载数据！{Style.RESET_ALL}")  
    return True  


def collect_data(env, agent, buffer, num_episodes=100):  
    """收集数据（绿色日志）"""  
    for episode in tqdm(range(num_episodes), desc=f"{Fore.GREEN}数据收集中...{Style.RESET_ALL}"):  
        state = env.reset() 
        print(f"[DEBUG] collect_data: state 类型: {type(state)}")  # 调试信息 
        done = False  
        step_count = 0  # 跟踪单回合步数  
        while not done:  
            encoded_state = env._encode_state(state)  # state 是 ToTNode 对象  
            action = agent.get_action(encoded_state)  
            next_node, reward, done = env.step(action)  
            buffer.add(encoded_state, action, reward, env._encode_state(next_node), done)  
            state = next_node  # 更新 state   
            step_count += 1  
            # 强制终止（防止代码错误导致无限循环）  
            if step_count > 1000:  
                print(f"{Fore.RED}强制终止：Episode {episode+1} 超过1000步！{Style.RESET_ALL}")  
                done = True  
            # 实时打印关键信息（浅绿色）  
            print(f"{Fore.LIGHTGREEN_EX}[数据] Episode {episode+1}, Step Reward: {reward:.2f}, Done: {done}{Style.RESET_ALL}")  


def train_rl():  
    # 初始化组件（黄色标题）  
    print(f"{Fore.YELLOW}=== 初始化环境与智能体 ==={Style.RESET_ALL}")  
    env = MathSolver(equation="x² + 2x - 3 = 0", ground_truth="1,-3")  
    agent = PPOAgent(state_dim=env.state_dim)  
    buffer = ReplayBuffer(capacity=10000)  
    
    # 尝试从文件加载数据  
    data_loaded = load_data_from_file(buffer)  
    
    # 如果文件没有数据，进行数据收集  
    if not data_loaded or len(buffer) == 0:  
        # 数据收集阶段（绿色标题）  
        print(f"\n{Fore.GREEN}=== 开始数据收集 (共100回合) ==={Style.RESET_ALL}")  
        collect_data(env, agent, buffer, num_episodes=100)  
    
    # 训练阶段（黄色标题）  
    print(f"\n{Fore.YELLOW}=== 开始策略网络训练 ==={Style.RESET_ALL}")  
    batch_size = 64  
    num_epochs = 100  
    for epoch in range(num_epochs):  
        batch = buffer.sample(batch_size)  
        if batch is None:  
            continue  
        loss = agent.update(batch)  
        # 训练日志（黄色）  
        print(f"{Fore.YELLOW}[训练] Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}{Style.RESET_ALL}")  
    
    # 保存模型（青色标题）  
    print(f"{Fore.CYAN}\n=== 保存模型到 rl_policy.pth ==={Style.RESET_ALL}")  
    torch.save(agent.policy_net.state_dict(), "rl_policy.pth")  


if __name__ == "__main__":  
    train_rl()