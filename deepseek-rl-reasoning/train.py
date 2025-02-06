import json  
from base_model import BaseModel  
from rl_framework import GRPO  
from config import Config  
import torch

def load_data(data_path):  
    with open(data_path, "r", encoding="utf-8") as f:  
        return json.load(f)  

def train_rl_model():  
    # 初始化基础模型和强化学习框架  
    base_model = BaseModel()  
    rl_framework = GRPO(base_model)  

    # 加载训练数据  
    train_data = load_data(Config.TRAIN_DATA_PATH)  

    # 训练循环  
    for epoch in range(Config.NUM_EPOCHS):  
        for sample in train_data:  
            prompt = sample["prompt"]  
            target = sample["target"]  

            # 生成一组输出  
            generated_texts = [base_model.generate(prompt) for _ in range(Config.GROUP_SIZE)]  
            targets = [target] * Config.GROUP_SIZE  

            # 更新策略  
            rl_framework.update(prompt, generated_texts, targets)  

        # 每轮训练后保存模型  
        torch.save(base_model.state_dict(), f"model_epoch_{epoch}.pth")  
        print(f"Model saved for epoch {epoch}")  
        
        # 每 1 步打印一次生成的文本  
        if epoch % 1 == 0:  
            print(f"Epoch {epoch}: Generated Text - {generated_texts[0]}")  

        # 如果需要从磁盘加载模型（例如在程序重启后）  
        # base_model.load_state_dict(torch.load(f"model_epoch_{epoch}.pth")) 

if __name__ == "__main__":  
    train_rl_model()