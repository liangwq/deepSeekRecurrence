from tasks.math_solver import MathSolver
from agents.rl_agent import PPOAgent
import torch

def validate_rl():
    # 加载训练好的策略
    env = MathSolver(equation="x² + 2x - 3 = 0", ground_truth="1,-3")
    agent = PPOAgent(state_dim=env.state_dim)
    agent.policy_net.load_state_dict(torch.load("rl_policy.pth"))
    
    # 运行测试
    num_episodes = 20
    total_reward = 0.0
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = agent.get_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    print(f"平均测试奖励: {total_reward / num_episodes:.2f}")

if __name__ == "__main__":
    validate_rl()
