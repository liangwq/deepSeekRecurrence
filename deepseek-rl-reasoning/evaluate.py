import json  
from base_model import BaseModel  
from config import Config  

def load_data(data_path):  
    with open(data_path, "r", encoding="utf-8") as f:  
        return json.load(f)  

def evaluate_model():  
    base_model = BaseModel()  
    test_data = load_data(Config.TEST_DATA_PATH)  

    correct_count = 0  
    for sample in test_data:  
        prompt = sample["prompt"]  
        target = sample["target"]  
        generated_text = base_model.generate(prompt)  
        if generated_text.strip() == target.strip():  
            correct_count += 1  

    accuracy = correct_count / len(test_data)  
    print(f"Model Accuracy: {accuracy * 100:.2f}%")  

if __name__ == "__main__":  
    evaluate_model()