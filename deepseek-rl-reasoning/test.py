import requests  
import json  

# API 配置  
url = "https://api.siliconflow.cn/v1/chat/completions"  
headers = {  
    "Authorization": "Bearer sk-wquqycgftjqtjiwbcwnvugfptqacnvtbojvdthsezqlyoajq",  
    "Content-Type": "application/json"  
}  

def generate_reasoning_data(prompt):  
    """生成推理任务数据（中文）"""  
    try:  
        # 引导模型生成推理步骤  
        reasoning_prompt = f"""  
        请逐步解释以下问题的解决方法，并给出最终答案：  
        问题：{prompt}  
        要求：  
        1. 详细列出每一步的推理过程。  
        2. 最后明确写出“最终答案：XXXX”。  
        """  
        payload = {  
            "model": "deepseek-ai/DeepSeek-V3",  
            "messages": [{"role": "user", "content": reasoning_prompt}],  
            "stream": False,  
            "max_tokens": 512,  
            "stop": ["null"],  
            "temperature": 0.7,  
            "top_p": 0.7,  
            "top_k": 50,  
            "frequency_penalty": 0.5,  
            "n": 1,  
            "response_format": {"type": "text"}  
        }  
        response = requests.post(url, json=payload, headers=headers)  
        response_data = response.json()  

        # 提取完整响应  
        full_response = response_data["choices"][0]["message"]["content"]  

        # 提取推理过程和最终答案  
        reasoning_content = full_response  # 这里假设完整响应就是推理过程  
        content = full_response.split("最终答案：")[-1].strip() if "最终答案：" in full_response else full_response  

        # 生成总结  
        summary_prompt = f"请总结以下推理过程和答案：\n推理过程：{reasoning_content}\n答案：{content}"  
        payload_summary = {  
            "model": "deepseek-ai/DeepSeek-V3",  
            "messages": [{"role": "user", "content": summary_prompt}],  
            "stream": False,  
            "max_tokens": 1024,  
            "stop": ["null"],  
            "temperature": 0.7,  
            "top_p": 0.7,  
            "top_k": 50,  
            "frequency_penalty": 0.5,  
            "n": 1,  
            "response_format": {"type": "text"}  
        }  
        response_summary = requests.post(url, json=payload_summary, headers=headers)  
        summary_data = response_summary.json()  
        summary = summary_data["choices"][0]["message"]["content"]  

        return reasoning_content, content, summary  

    except Exception as e:  
        print(f"Error in generating reasoning data: {e}")  
        return None, None, None  

# 加载差异化任务  
with open("data/differentiated_prompts_zh.json", "r", encoding="utf-8") as f:  
    prompts = json.load(f)  

# 构建数据集  
dataset = []  
for prompt in prompts:  
    reasoning_content, content, summary = generate_reasoning_data(prompt)  
    if reasoning_content and content and summary:  # 确保数据有效  
        dataset.append({  
            "prompt": prompt,  
            "reasoning": reasoning_content,  
            "target": content,  
            "summary": summary  
        })  

# 保存数据  
with open("data/reasoning_data_zh.json", "w", encoding="utf-8") as f:  
    json.dump(dataset, f, ensure_ascii=False, indent=4)  

print("数据集构建完成，已保存至 data/reasoning_data_zh.json")