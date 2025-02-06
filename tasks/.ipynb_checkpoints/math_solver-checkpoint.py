from core.tree import ToTNode, TreeManager  
from core.search import MCTS  
from core.model import DeepSeekModel  
from core.rewards import MathReward  
import yaml  
import copy  
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

class MathSolver:  
    """数学方程求解任务"""  
    def __init__(self, equation: str, ground_truth: str, max_steps=20):
         # 模型路径  
        self.MODEL_PATH = "/root/autodl-tmp/model_cache/DeepSeek-R1-Distill-Qwen-7B"  
        
        # 加载模型和分词器  
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)  
        self.model_enc = AutoModel.from_pretrained(self.MODEL_PATH)   
        self.embedding_dim = 256  # 嵌入维度（根据模型输出调整） 

        # 测试嵌入向量的输出维度  
        test_embedding = self._embed_text("test")  
        self.embedding_dim = test_embedding.shape[0]  # 动态调整嵌入维度  
        print(f"[DEBUG] 嵌入维度: {self.embedding_dim}")  
    
    
        # 添加最大步数限制
        self.max_steps = max_steps
        self.current_step = 0  # 跟踪当前步数
        # 初始化组件  
        self.model = DeepSeekModel()  
        self.state_dim = 2 + 2 * self.embedding_dim  # 手工特征（2） + 两个嵌入向量（2 * 768）  
        
        self.reward_calculator = MathReward(ground_truth)  
        print(f"Initializing MathSolver with equation: {equation}")  # Debug print  
        self.equation = equation 
        
        # 创建根节点，确保 children 是列表  
        root_node = ToTNode(name=f"解方程 {equation}")  
        
        # 如果 children 是元组，转换为列表  
        if isinstance(root_node.children, tuple):  
            root_node.children = list(root_node.children)  
        
        self.tree = TreeManager(root_node)  
        
        # 加载搜索配置  
        with open("configs/deepseek.yaml") as f:  
            config = yaml.safe_load(f)  
            self.search = MCTS(  
                exploration_weight=config.get("exploration_weight", 1.414),  
                max_depth=config.get("max_depth", 5)  
            )  

    def _is_terminal(self) -> bool:
        """终止条件：找到答案或超过最大步数"""
        is_solved = "答案正确" in self.current_node.name
        exceeded_max_steps = self.current_step >= self.max_steps
        return is_solved or exceeded_max_steps
        
    def _is_terminal(self, current_node) -> bool:  
        """终止条件：找到答案或超过最大步数"""  
        is_solved = "答案正确" in current_node.name  
        exceeded_max_steps = self.current_step >= self.max_steps  
        return is_solved or exceeded_max_steps  
        
    def _get_current_node(self):  
        """获取当前节点"""  
        # 如果 current_node 未设置，默认使用根节点  
        if not hasattr(self, 'current_node'):  
            self.current_node = self.tree.root  
        return self.current_node 
    def _select_best_child(self, node: ToTNode):  
        """选择最优子节点"""  
        # 如果节点没有子节点，不执行任何操作  
        if not node.children:  
            print("警告：当前节点没有子节点，无法选择最优子节点")  
            return node  
        
        # 根据节点价值选择最优子节点  
        try:  
            # 使用最大价值作为选择标准  
            best_child = max(node.children, key=lambda child: child.value)  
            
            # 更新当前节点为最优子节点  
            self.current_node = best_child  
            
            return best_child  
        except Exception as e:  
            print(f"选择最优子节点时出错：{e}")  
            # 如果出现异常，返回原节点  
            return node

    def _backtrack(self, node: ToTNode):  
        """回溯到父节点"""  
        # 如果节点没有父节点，保持在当前节点  
        if node.parent is None:  
            print("警告：当前节点是根节点，无法回溯")  
            return node  
        
        try:  
            # 更新当前节点为父节点  
            self.current_node = node.parent  
            
            print(f"回溯：从节点 '{node.name}' 到父节点 '{node.parent.name}'")  
            
            return node.parent  
        except Exception as e:  
            print(f"回溯时出错：{e}")  
            # 如果出现异常，返回原节点  
            return node

    def run(self, iterations=10):  
        """执行搜索过程"""  
        print(f"\n{'='*30}\n开始推理，最大迭代次数: {iterations}\n{'='*30}")  
        for i in range(iterations):  
            print(f"\n=== 迭代 {i+1}/{iterations} ===")  
            # 选择 -> 扩展 -> 模拟 -> 反向传播  
            leaf = self.search.select(self.tree.root)  
            
            # 确保 children 是列表  
            if isinstance(leaf.children, tuple):  
                leaf.children = list(leaf.children)  
            
            if not leaf.children:  
                self._expand_node(leaf)  
            reward = self.search.simulate(leaf, self.reward_calculator.calculate)  
            self.search.backpropagate(leaf, reward)  
            self.tree.print_tree()  
        return self.tree.root  

    def _expand_node(self, node: ToTNode):  
        """扩展节点"""  
        
        # 确保 children 是列表  
        if isinstance(node.children, tuple):  
            node.children = list(node.children)  
        # 调试信息  
        print(f"[DEBUG] node.children 类型: {type(node.children)}") 
        
        prompt = f"问题: {node.name}\n当前步骤: {self.tree.get_node_path(node)}\n请生成下一步推理步骤:"  
        print(f"\n生成扩展步骤，提示: {prompt[:50]}...")  
        node.context = {"last_prompt": prompt}  # 保存上下文信息  
    
        steps = self.model.generate_steps(prompt)  
        
        for step in steps:  
            # 创建新节点  
            new_node = ToTNode(name=step, parent=node)  
            
            # 如果新节点的 children 是元组，转换为列表  
            if isinstance(new_node.children, tuple):  
                new_node.children = list(new_node.children)  
            
            new_node.context = {"last_answer": step}  # 保存生成的答案  
            
            # 追加新节点  
            node.children.append(new_node)  
            print(f"添加子节点: {step}")
    def _embed_text(self, text: str) -> np.ndarray:  
        """使用 DeepSeek 模型生成文本嵌入"""  
        # 使用分词器将文本转换为输入 ID  
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)  
        
        # 生成嵌入  
        with torch.no_grad():  
            outputs = self.model_enc(**inputs)  
            embeddings = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的嵌入  
        
        # 转换为 numpy 数组  
        return embeddings.squeeze().detach().cpu().numpy()  
        
    def reset(self):
        """重置环境，返回初始状态"""
        self.tree = TreeManager(ToTNode(name=f"解方程 {self.equation}"))
        self.current_node = self.tree.root  # 设置当前节点为根节点  
        print(f"[DEBUG] reset: state 类型: {type(self.current_node)}")  # 调试信息  
        return self.current_node  # 返回根节点  
        #return self._encode_state(self.tree.root)
    
    def step(self, action: int):
        """执行动作并更新步数"""
        self.current_step += 1
        """
        执行动作，返回 (next_state, reward, done)
        - action: 0（扩展），1（选择最优子节点），2（回溯）
        """
        current_node = self._get_current_node()
        
        # 执行动作
        if action == 0:
            self._expand_node(current_node)
        elif action == 1:
            self._select_best_child(current_node)
        elif action == 2:
            self._backtrack(current_node)
        
        # 获取新状态和奖励
        next_node = self._get_current_node()  # 获取新的 ToTNode 对象 
        reward = self.reward_calculator.calculate(current_node.name)
        done = self._is_terminal(next_node)
        
        return next_node, reward, done
    
    def _encode_state(self, node: ToTNode) -> np.ndarray:  
        """将节点状态编码为向量，包含手工特征和语义信息"""  
        if isinstance(node, list):  # 如果 node 是 list，抛出错误  
            raise TypeError("node 必须是 ToTNode 对象，而不是 list")  
         
        # 手工特征  
        features = np.zeros(self.state_dim)  
        # 特征1：节点深度  
        features[0] = node.depth / 10.0  
        # 特征2：子节点平均价值  
        if node.children:  
            features[1] = sum(c.value for c in node.children) / len(node.children)  
    
        # 语义特征（上下文信息）  
        if hasattr(node, "context") and node.context:  # 检查是否有上下文信息  
            # 对 last_prompt 和 last_answer 进行嵌入  
            last_prompt = node.context.get("last_prompt", "")  
            last_answer = node.context.get("last_answer", "")  
    
            # 使用 DeepSeek 模型生成嵌入向量  
            prompt_embedding = self._embed_text(last_prompt)  
            answer_embedding = self._embed_text(last_answer)  
    
            # 将嵌入向量拼接到手工特征中  
            features[2:2 + self.embedding_dim] = prompt_embedding  
            features[2 + self.embedding_dim:2 + 2 * self.embedding_dim] = answer_embedding  
    
        return features.tolist() 
    '''
    def _is_terminal(self) -> bool:
        """判断是否终止（如找到解或达到最大深度）"""
        return "答案正确" in self.current_node.name or self.current_node.depth >= 5
    '''