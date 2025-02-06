import os  
import json  
import torch  # 导入 PyTorch  
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  

# 设置 PyTorch 的内存管理  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  

# 检查 GPU 是否可用  
device = "cuda" if torch.cuda.is_available() else "cpu"  
print(f"当前使用的设备: {device}")  

# 本地模型目录  
model_directory = "/root/autodl-tmp/model_cache/DeepSeek-R1-Distill-Qwen-7B"  # 替换为您本地模型的实际路径  

# 加载模型和分词器  
try:  
    # 从本地加载模型和分词器  
    tokenizer = AutoTokenizer.from_pretrained(model_directory)  

    # 配置量化加载  
    quant_config = BitsAndBytesConfig(  
        load_in_8bit=True  # 或者使用 load_in_4bit=True  
    )  

    model = AutoModelForCausalLM.from_pretrained(  
        model_directory,  
        quantization_config=quant_config,  # 启用量化  
        device_map="auto"  # 自动分配设备  
    )  
    print("模型和分词器加载成功！")  
except Exception as e:  
    print(f"模型加载错误: {e}")  
    exit(1)  # 退出程序  

def generate_reasoning_data(prompt):  
    """生成推理任务数据（中文）"""  
    try:  
        # 引导模型生成推理步骤  
        reasoning_prompt = f"""  
        请逐步解释以下问题的解决方法，并给出最终答案：  
        问题：{prompt}  
        要求：  
        1. 详细列出每一步的推理过程，并用序号标记。  
        2. 在最后明确写出“最终答案：XXXX”。  
        """  
        print("生成推理过程的提示：", reasoning_prompt)  

        # 对输入进行分词，并将数据移动到 GPU  
        inputs = tokenizer(reasoning_prompt, return_tensors="pt").to(device)  

        # 生成推理过程  
        outputs = model.generate(  
            inputs['input_ids'],  
            max_new_tokens=1024,  # 控制生成的新 token 数量  
            num_return_sequences=1,  
            attention_mask=inputs['attention_mask'],  
            pad_token_id=tokenizer.pad_token_id  
        )  
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)  
        print("生成的推理过程：", full_response)  

        # 提取推理过程和最终答案  
        reasoning_content = full_response  # 这里假设完整响应就是推理过程  
        if "最终答案：" in full_response:  
            content = full_response.split("最终答案：")[-1].strip()  
        else:  
            # 如果没有“最终答案：”，尝试通过其他方式提取答案  
            content = extract_answer(full_response)  

        # 生成总结  
        summary_prompt = f"""  
        请总结以下推理过程和答案：  
        推理过程：{reasoning_content}  
        答案：{content}  
        要求：  
        1. 总结推理过程的关键步骤。  
        2. 明确最终答案。  
        """  
        print("生成总结的提示：", summary_prompt)  

        # 对输入进行分词，并将数据移动到 GPU  
        inputs_summary = tokenizer(summary_prompt, return_tensors="pt").to(device)  

        # 生成总结  
        outputs_summary = model.generate(  
            inputs_summary['input_ids'],  
            max_new_tokens=768,  # 控制生成的新 token 数量  
            num_return_sequences=1,  
            attention_mask=inputs_summary['attention_mask'],  
            pad_token_id=tokenizer.pad_token_id  
        )  
        summary = tokenizer.decode(outputs_summary[0], skip_special_tokens=True)  
        print("生成的总结：", summary)  

        return reasoning_content, content, summary  

    except Exception as e:  
        print(f"Error in generating reasoning data: {e}")  
        return None, None, None  

# 其他代码保持不变...  

def extract_answer(text):  
    """从响应文本中提取最终答案（如果未明确标注）"""  
    # 尝试通过常见模式提取答案  
    if "答案是：" in text:  
        return text.split("答案是：")[-1].strip()  
    elif "最终结果：" in text:  
        return text.split("最终结果：")[-1].strip()  
    else:  
        # 如果未找到明确标注，返回最后一行作为答案  
        last_line = text.strip().split("\n")[-1]  
        return last_line  

# 加载差异化任务  
try:  
    with open("data/differentiated_prompts_zh.json", "r", encoding="utf-8") as f:  
        prompts = json.load(f)  
        print("加载的提示数据: 成功")  
except Exception as e:  
    print(f"加载提示错误: {e}")  
    prompts = []  

# 构建数据集  
dataset = []  
for prompt in prompts:  
    print("处理提示：", prompt)  
    reasoning_content, content, summary = generate_reasoning_data(prompt)  
    if reasoning_content and content and summary:  # 确保数据有效  
        dataset.append({  
            "prompt": prompt,  
            "reasoning": reasoning_content,  
            "target": content,  
            "summary": summary  
        })  
        # 保存单条数据  
        try:  
            with open("data/reasoning_data_zh0.json", "a", encoding="utf-8") as f:  
                json.dump([{  
                    "prompt": prompt,  
                    "reasoning": reasoning_content,  
                    "target": content,  
                    "summary": summary  
                }], f, ensure_ascii=False, indent=4)  
                f.write("\n")  # 换行以区分不同记录  
            print("单条数据已保存")  
        except Exception as e:  
            print(f"Error saving single record: {e}")  

print("数据集构建完成，数据已保存至 data/reasoning_data_zh0.json")