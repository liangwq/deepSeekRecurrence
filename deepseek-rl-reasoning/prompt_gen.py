import random  
import json  

# 任务模板和变量  
math_templates = [  
    "计算 {} + {} 的值是多少？",  
    "{} 的平方根是多少？",  
    "如果 x = {}，那么 2x + 5 等于多少？",  
    "{} 和 {} 的最大公约数是多少？",  
    "一个长方形的长是 {}，宽是 {}，它的面积是多少？"  
]  
logic_templates = [  
    "如果 A = {}，B = {}，那么 A 和 B 的逻辑与结果是什么？",  
    "如果今天是星期{}，那么 100 天后是星期几？",  
    "求解以下逻辑谜题：{}",  
    "如果 {} 是 {}，那么 {} 是什么？",  
    "判断以下陈述是否为真：{}"  
]  
language_templates = [  
    "单词 '{}' 中有多少个字母 {}？",  
    "将以下句子翻译成中文：{}",  
    "找出句子中的语法错误：{}",  
    "单词 '{}' 的复数形式是什么？",  
    "解释以下成语的含义：{}"  
]  
language_words = ["苹果", "香蕉", "橙子", "葡萄", "草莓"]  
language_letters = ["a", "b", "c", "d", "e"]  

science_templates = [  
    "解释 {} 的原理。",  
    "{} 的化学式是什么？",  
    "描述 {} 的实验步骤。",  
    "{} 在生物学中的作用是什么？",  
    "{} 的物理特性是什么？"  
]  
science_topics = ["光合作用", "重力", "酸碱性", "细胞分裂", "化学反应"]  

common_sense_templates = [  
    "如果 {}，你会怎么处理？",  
    "为什么 {} 会发生？",  
    "如何正确地 {}？",  
    "在日常生活中，{} 的重要性是什么？",  
    "如果 {}，最可能的结果是什么？"  
]  
common_sense_topics = ["煮饭", "洗衣服", "修理自行车", "清洁房间", "种植植物"]  

# 生成 1000 条差异化任务  
prompts = []  
for i in range(1000):  
    if i < 200:  # 数学推理  
        template = random.choice(math_templates)  
        num_placeholders = template.count("{}")  
        variables = [random.randint(1, 100) for _ in range(num_placeholders)]  
        prompt = template.format(*variables)  
    elif i < 400:  # 逻辑推理  
        template = random.choice(logic_templates)  
        num_placeholders = template.count("{}")  
        variables = [random.randint(1, 7) for _ in range(num_placeholders)]  
        prompt = template.format(*variables)  
    elif i < 600:  # 语言推理  
        template = random.choice(language_templates)  
        word = random.choice(language_words)  
        letter = random.choice(language_letters)  
        prompt = template.format(word, letter)  
    elif i < 800:  # 科学推理  
        template = random.choice(science_templates)  
        topic = random.choice(science_topics)  
        prompt = template.format(topic)  
    else:  # 常识推理  
        template = random.choice(common_sense_templates)  
        topic = random.choice(common_sense_topics)  
        prompt = template.format(topic)  
    prompts.append(prompt)  

# 保存任务  
with open("data/differentiated_prompts_zh.json", "w", encoding="utf-8") as f:  
    json.dump(prompts, f, ensure_ascii=False, indent=4)