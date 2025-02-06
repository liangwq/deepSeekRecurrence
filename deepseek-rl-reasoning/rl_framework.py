import torch  
import torch.nn.functional as F  
from torch.optim import Adam  
from base_model import BaseModel  
from config import Config  
from sentence_transformers import SentenceTransformer, util  
import os  

# 启用 PyTorch 显存管理优化  
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  

# 设置设备  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

class GRPO:  
    def __init__(self, model, learning_rate=Config.LEARNING_RATE):  
        print("Initializing GRPO...")  
        self.model = model.to(device)  # 将模型移动到 GPU  
        print(f"Base model initialized and moved to {device}")  
        
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)  
        print(f"Optimizer initialized with learning rate: {learning_rate}")  
        
        # 加载中文语义模型并移动到 GPU  
        print("Loading semantic model...")  
        self.semantic_model = SentenceTransformer("/root/autodl-tmp/sbert-base-chinese-nli").to(device)  
        print(f"Semantic model loaded successfully and moved to {device}")  

    def compute_reward(self, generated_text, target):  
        print("\nComputing reward...")  
        print(f"Generated text: {generated_text}")  
        print(f"Target text: {target}")  
        
        # 编码文本并确保结果在 GPU 上  
        generated_embedding = self.semantic_model.encode(  
            generated_text,   
            show_progress_bar=False,   
            convert_to_tensor=True  
        ).to(device)  
        target_embedding = self.semantic_model.encode(  
            target,   
            show_progress_bar=False,   
            convert_to_tensor=True  
        ).to(device)  
        
        # 计算余弦相似度  
        semantic_reward = util.pytorch_cos_sim(generated_embedding, target_embedding).item()  
        print(f"Semantic reward: {semantic_reward}")  

        # 格式奖励  
        format_reward = 1.0 if "<think>" in generated_text and "</think>" in generated_text else 0.0  
        print(f"Format reward: {format_reward}")  

        # 总奖励  
        total_reward = semantic_reward + format_reward  
        print(f"Total reward: {total_reward}")  
        return total_reward  

    def update(self, prompt, generated_texts, targets):  
        print("\nUpdating model...")  
        print(f"Prompt: {prompt}")  
        print(f"Generated texts: {generated_texts}")  
        print(f"Targets: {targets}")  
        
        # 计算奖励  
        print("Calculating rewards...")  
        rewards = [self.compute_reward(text, target) for text, target in zip(generated_texts, targets)]  
        print(f"Rewards: {rewards}")  
        
        # 计算优势  
        print("Calculating advantages...")  
        mean_reward = sum(rewards) / len(rewards)  
        advantages = [reward - mean_reward for reward in rewards]  
        print(f"Advantages: {advantages}")  

        # 更新策略  
        print("Updating model parameters...")  
        self.model.train()  # 设置模型为训练模式  
        for advantage, text in zip(advantages, generated_texts):  
            print(f"\nProcessing text: {text}")  
            print(f"Advantage: {advantage}")  
            
            # Tokenize the text 并移动到 GPU  
            inputs = self.model.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)  
            print(f"Inputs (on GPU): {inputs}")  
            
            # 计算 logits  
            outputs = self.model(**inputs)  
            logits = outputs.logits  
            print("Logits min:", logits.min().item())  
            print("Logits max:", logits.max().item())  
            
            # 限制 logits 的范围  
            logits = torch.clamp(logits, min=1e-8, max=1e8)  
            
            # 计算损失  
            loss = -torch.log(logits.float()) * advantage  
            loss = loss.mean()  # 对 loss 取均值，使其变为标量  
            print(f"Loss: {loss.item()}")  
            
            # 反向传播后手动释放显存  
            torch.cuda.empty_cache()  
            
            # 更新参数  
            self.optimizer.step()  
            self.optimizer.zero_grad()  
        
        print("Update complete") 