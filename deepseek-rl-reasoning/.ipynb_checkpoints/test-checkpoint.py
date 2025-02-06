from openai import OpenAI  
import json  
import time  
import logging  

# 初始化DeepSeek R1客户端  
client = OpenAI(api_key="sk-5309e682a1f24b3ab0188b0caca7b9cb", base_url="https://api.deepseek.com")  

# 配置日志  
logging.basicConfig(filename="error.log", level=logging.ERROR, format="%(asctime)s - %(message)s")  

def generate_reasoning_data(prompt, max_retries=3):  
    retries = 0  
    while retries < max_retries:  
        try:  
            response = client.chat.completions.create(  
                model="deepseek-reasoner",  
                messages=[{"role": "user", "content": prompt}]  
            )  
            reasoning_content = response.choices[0].message.reasoning_content  
            content = response.choices[0].message.content  
            return reasoning_content, content  
        except Exception as e:  
            logging.error(f"Attempt {retries + 1} failed: {e}")  
            retries += 1  
            time.sleep(2)  # 等待 2 秒后重试  
    return None, None  

def build_dataset(prompts):  
    dataset = []  
    for prompt in prompts:  
        reasoning_content, content = generate_reasoning_data(prompt)  
        if reasoning_content is None or content is None:  
            continue  
        # 生成 summary  
        summary_prompt = f"请总结以下推理过程和答案：\n推理过程：{reasoning_content}\n答案：{content}"  
        summary_response = client.chat.completions.create(  
            model="deepseek-reasoner",  
            messages=[{"role": "user", "content": summary_prompt}]  
        )  
        summary = summary_response.choices[0].message.content  
        dataset.append({  
            "prompt": prompt,  
            "reasoning": reasoning_content,  
            "target": content,  
            "summary": summary  
        })  
    return dataset  

# 中文任务示例  
prompts = [  
    "9.11 和 9.8，哪个更大？",  
    "草莓这个单词中有多少个字母 R？",  
    "请解释光合作用的原理。"  
]   # 生成 1000 条数据  

# 构建数据集  
dataset = build_dataset(prompts)  

# 保存数据  
with open("data/reasoning_data_zh.json", "w", encoding="utf-8") as f:  
    json.dump(dataset, f, ensure_ascii=False, indent=4)
