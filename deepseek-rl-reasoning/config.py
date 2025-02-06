class Config:  
    MODEL_PATH = "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-1.5B"  
    LEARNING_RATE = 1e-5  
    MAX_LENGTH = 1024  
    NUM_EPOCHS = 3  
    TRAIN_DATA_PATH = "data/train.json"  
    TEST_DATA_PATH = "data/test.json"  
    GROUP_SIZE = 5  # GRPO 的组大小  