from .model import ModelProvider

from rwkv_x.model import RWKV_X
from rwkv_x.utils import PIPELINE, PIPELINE_ARGS


class RWKVX(ModelProvider):
    def __init__(
        self,
        model_name: str,
    ):
        self.model_name = model_name

        self.model = RWKV_X(
            model_path=f"models\{model_name}.pth",
            strategy="cuda fp16"
        )

        self.pipeline = PIPELINE(self.model)

        self.args = PIPELINE_ARGS(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            alpha_frequency=0.0,
            alpha_presence=0.0,
            chunk_len=256,
        )

    def encode_text_to_tokens(self, text: str):
        return self.pipeline.encode(text)

    def decode_tokens(self, tokens, max_tokens: int | None = None):
        return self.pipeline.decode(tokens[:max_tokens])

    def generate_prompt(self, context: str, question: str) -> str:
        return f"{context}\n\n{question}\nAnswer:"

    async def evaluate_model(self, prompt: str) -> str:
        return self.generate(prompt, max_new_tokens=128)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        output = self.pipeline.generate(
            prompt,
            token_count=max_new_tokens,
            args=self.args
        )
        return output.strip()
