from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import yaml

class DeepSeekModel:
    """DeepSeek API 接口封装"""
    def __init__(self, config_path="configs/deepseek.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        self.model_name = config["model"]
        self.system_prompt = config["system_prompt"]
        self.params = {
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "stream": False
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_steps(self, prompt: str) -> list:
        """调用 API 生成推理步骤"""
        print(f"\n[API 请求] 提示内容: {prompt[:50]}...")  # 打印前50字符
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                **self.params
            )
            raw_text = response.choices[0].message.content
            print(f"[API 响应] 原始内容:\n{raw_text}\n")
            return self._parse_response(raw_text)
        except Exception as e:
            print(f"[ERROR] API 调用失败: {str(e)}")
            return []

    def _parse_response(self, text: str) -> list:
        """解析模型响应为步骤列表"""
        steps = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith(("1.", "2.", "3.", "首先", "接着")):
                steps.append(line.split("。")[0])  # 取步骤的第一句话
        return steps[:5]  # 最多保留5个步骤