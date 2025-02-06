import math  
import random  
from .tree import ToTNode, TreeManager  

class MCTS:  
    """蒙特卡洛树搜索（带日志打印）"""  
    def __init__(self, exploration_weight=1.414, max_depth=5):  
        self.exploration_weight = exploration_weight  
        self.max_depth = max_depth  

    def select(self, node: ToTNode) -> ToTNode:  
        """基于 UCB 选择子节点"""  
        while node.children:  
            # 处理没有访问过的情况  
            if any(c.visits == 0 for c in node.children):  
                # 选择未访问过的节点  
                unvisited = [c for c in node.children if c.visits == 0]  
                selected = random.choice(unvisited)  
                print(f"选择未访问子节点: {selected.name}")  
                return selected  

            # 计算总访问次数，避免 log(0)  
            total_visits = sum(c.visits for c in node.children)  
            log_n = math.log(total_visits) if total_visits > 0 else 0  

            # 计算 UCB 分数  
            ucb_scores = [  
                (c.value / (c.visits + 1e-6)) +  # 避免除以零  
                self.exploration_weight * math.sqrt(log_n / (c.visits + 1e-6))  
                for c in node.children  
            ]  

            # 选择最大 UCB 分数的节点  
            selected = node.children[ucb_scores.index(max(ucb_scores))]  
            print(f"选择子节点: {selected.name} (UCB={max(ucb_scores):.2f})")  
            node = selected  

        return node  

    def simulate(self, node: ToTNode, reward_fn) -> float:  
        """随机模拟并计算奖励"""  
        print(f"开始模拟，当前节点: {node.name}")  
        current = node  
        total_reward = 0.0  
        
        # 如果没有子节点，直接计算当前节点奖励  
        if not current.children:  
            print(f"直接计算叶子节点奖励")  
            return reward_fn(current.name)  

        for _ in range(self.max_depth):  
            if not current.children:  
                break  
            current = random.choice(current.children)  
            print(f"模拟步骤: {current.name}")  
            # 使用节点名称作为奖励计算参数  
            step_reward = reward_fn(current.name)  
            total_reward += step_reward  
            print(f"步骤奖励: {step_reward:.2f}")  

        print(f"模拟总奖励: {total_reward:.2f}")  
        return total_reward  

    def backpropagate(self, node: ToTNode, reward: float):  
        """反向传播更新节点值"""  
        print(f"反向传播奖励: {reward:.2f}")  
        while node is not None:  
            node.visits += 1  
            node.value += reward  
            print(f"更新节点: {node.name} (新价值={node.value:.2f}, 访问次数={node.visits})")  
            node = node.parent