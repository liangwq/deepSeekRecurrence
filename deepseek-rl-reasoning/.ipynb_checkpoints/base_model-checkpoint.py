import torch.nn as nn  
from transformers import AutoModelForCausalLM, AutoTokenizer  
from config import Config  
import torch

# 定义设备  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

class BaseModel(nn.Module):  
    def __init__(self, model_path=Config.MODEL_PATH):  
        super(BaseModel, self).__init__()  
        # 加载模型并移动到 GPU  
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)  
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  
        print(f"Model loaded and moved to {device}")  

    def forward(self, **inputs):  
        # 将输入数据移动到 GPU  
        inputs = {key: value.to(device) for key, value in inputs.items()}  
        print("Inputs to forward (on GPU):", inputs)  
        return self.model(**inputs)  

    def generate(self, prompt, max_length=Config.MAX_LENGTH):  
        # Tokenize 输入并移动到 GPU  
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)  
        print("Inputs to generate (on GPU):", inputs)  
        outputs = self.model.generate(inputs["input_ids"], max_length=max_length)  
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)  

    def parameters(self):  
        return self.model.parameters()