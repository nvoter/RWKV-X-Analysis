import requests
from needlehaystack.evaluators.evaluator import Evaluator
import random

class QwenEvaluator(Evaluator):
    def __init__(
        self,
        endpoint_url: str = "https://undatable-sympathetic-inez.ngrok-free.dev/evaluate",
        true_answer: str = None
    ):
        if (not true_answer):
            raise ValueError("true_answer must be supplied with init.")
        
        self.endpoint_url = endpoint_url
        self.true_answer = true_answer
    
    def evaluate_response(self, response: str) -> int:
        if response is None or response.strip() == "":
            return 0

        payload = {
            "text": response,
            "needle": self.true_answer,
        }

        try:
            r = requests.post(
                self.endpoint_url,
                json=payload,
                timeout=10.0,
            )
            r.raise_for_status()
            data = r.json()

            return int(data.get("score", 0))

        except Exception as e:
            print(f"[QwenEvaluator] Evaluation failed: {e}")
            return 0