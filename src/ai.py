"""Module for AI generation using the Qwen model."""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any


class AiGenerator:
    """Handles text generation using a local Large Language Model.

    Attributes:
        device (str): Computation device ('cuda' or 'cpu').
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (AutoModelForCausalLM): The loaded causal language model.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B") -> None:
        """Initializes the LLM and tokenizer[cite: 18].

        Args:
            model_id (str, optional): The model repository ID.
                Defaults to "Qwen/Qwen3-0.6B"[cite: 18].
        """
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.device
        )
        self.model.eval()

    def generate(self, query: str, gross_res: List[Dict[str, Any]]) -> str:
        """Generates an answer based on the provided context[cite: 18].

        Args:
            query (str): The user's question[cite: 18].
            gross_res (List[Dict[str, Any]]): Retrieved context
            chunks[cite: 18].

        Returns:
            str: The generated text answer[cite: 18].
        """
        context = "\n\n".join(
            (
                f"[Source {i + 1}: {res['file_path']}]\n"
                f"{res['text']}"
            )
            for i, res in enumerate(gross_res)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical assistant answering questions about"
                    " the vLLM codebase. "
                    "Answer ONLY using the provided context."
                    "If the context does not contain "
                    "the answer, say so explicitly instead of guessing."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")

        try:
            inputs_gpu = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs_gpu,
                    max_new_tokens=200,
                    do_sample=False,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        except torch.cuda.OutOfMemoryError:
            print("WARNING : Insufficient VRAM. Switching to CPU.")
            torch.cuda.empty_cache()
            gc.collect()

            self.model.to("cpu")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            self.model.to(self.device)

        tmp_res = outputs[0][inputs.input_ids.shape[-1]:]
        raw_answer = str(self.tokenizer.decode(tmp_res,
                                               skip_special_tokens=True))

        last_period_idx = raw_answer.rfind(".")
        if last_period_idx != -1:
            return raw_answer[:last_period_idx + 1]
        return raw_answer.strip()
