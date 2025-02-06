import torch  
from tasks.math_solver import MathSolver  
from agents.rl_agent import PolicyNetwork, PPOAgent  
from colorama import Fore, Style, init  

init(autoreset=True)  # 初始化颜色  


def load_trained_model(model_path, state_dim):  
    """加载训练好的策略网络"""  
    # 根据训练时的结构定义模型  
    policy_net = PolicyNetwork(input_dim=state_dim, hidden_dim=3, output_dim=3)  # 根据错误信息调整 hidden_dim  
    policy_net.load_state_dict(torch.load(model_path, weights_only=True))  # 使用 weights_only=True  
    policy_net.eval()  # 设置为评估模式  
    return policy_net  


def tree_of_thoughts(env, agent, max_depth=5):  
    """ToT的推理过程"""  
    root_node = env.reset()  # 初始化根节点  
    current_node = root_node  

    for depth in range(max_depth):  
        print(f"{Fore.YELLOW}=== 当前深度: {depth} ==={Style.RESET_ALL}")  

        # 编码当前节点的状态  
        try:  
            state = env._encode_state(current_node)  
        except AttributeError as e:  
            print(f"{Fore.RED}错误：current_node 类型错误。期望：节点对象，实际：{type(current_node)}。{Style.RESET_ALL}")  
            break  

        # 使用训练好的策略网络选择动作  
        action = agent.get_action(state)  
        print(f"{Fore.GREEN}选择的动作: {action}{Style.RESET_ALL}")  

        # 执行动作  
        if action == 0:  # 扩展节点  
            print(f"{Fore.CYAN}扩展当前节点: {current_node.name}{Style.RESET_ALL}")  
            env._expand_node(current_node)  
        elif action == 1:  # 选择最优子节点  
            print(f"{Fore.CYAN}选择最优子节点{Style.RESET_ALL}")  
            current_node = env._select_best_child(current_node)  
        elif action == 2:  # 回溯  
            print(f"{Fore.CYAN}回溯到父节点{Style.RESET_ALL}")  
            current_node = env._backtrack(current_node)  

        # 打印当前节点信息  
        print(f"{Fore.LIGHTGREEN_EX}当前节点: {current_node.name}{Style.RESET_ALL}")  

        # 如果达到终止条件，退出  
        if env._is_terminal(current_node):  
            print(f"{Fore.GREEN}找到答案：{current_node.name}{Style.RESET_ALL}")  
            break  


def main():  
    # 初始化环境  
    env = MathSolver(equation="x² + 2x - 3 = 0", ground_truth="1,-3")  

    # 加载训练好的策略网络  
    policy_net = load_trained_model("rl_policy.pth", env.state_dim)  

    # 创建 PPO 智能体  
    agent = PPOAgent(state_dim=env.state_dim)  
    agent.policy_net = policy_net  # 使用训练好的策略网络替换默认的策略网络  

    # 执行 ToT 推理  
    print(f"{Fore.YELLOW}=== 开始 ToT 推理 ==={Style.RESET_ALL}")  
    tree_of_thoughts(env, agent, max_depth=50)  


if __name__ == "__main__":  
    main()