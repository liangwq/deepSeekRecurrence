import re

class MathReward:
    """数学问题奖励计算器"""
    def __init__(self, ground_truth: str):
        self.ground_truth = set(ground_truth.split(","))
        self.step_pattern = re.compile(r"x\s*=\s*([-+]?\d+)")
        self.invalid_keywords = ["除以零", "无解", "错误"]

    def calculate(self, text: str) -> float:
        """计算节点奖励"""
        print(f"\n计算奖励: {text[:30]}...")
        # 短期惩罚
        if any(kw in text for kw in self.invalid_keywords):
            print("检测到无效步骤，奖励 -1.0")
            return -1.0
        # 长期奖励
        answers = set(self.step_pattern.findall(text))
        if answers & self.ground_truth:
            print(f"答案正确 {answers}，奖励 +1.0")
            return 1.0
        print(f"答案未匹配，当前解 {answers}，正确答案 {self.ground_truth}，奖励 +0.1")
        return 0.1