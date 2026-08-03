"""
Wrapper unique autour de `transformers` pour charger/interroger les 3 LLMs du
projet (Mistral-7B, LLaMA-3-8B, Gemma-2-9B) de façon identique quel que soit
le pipeline appelant.

Pourquoi un wrapper et pas des appels transformers dispersés :
- Les 3 pipelines (self-consistency, calibration, RAG) ont besoin des mêmes
  briques (génération multi-échantillons, log-probabilités par token), donc
  centraliser évite la duplication et les incohérences de config
  (température, quantization, chat template...).
- Permet de swapper facilement un modèle par un autre dans les scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class GenerationResult:
    text: str
    # log-probabilité de chaque token généré (pour l'entropie, pipeline 2)
    token_logprobs: list[float]


class LLMWrapper:
    """Charge un modèle causal HF avec quantization optionnelle et expose
    des méthodes de génération utilisées par les 3 pipelines."""

    def __init__(self, hf_id: str, quantization: str = "4bit", device: Optional[str] = None):
        self.hf_id = hf_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        quant_config = None
        if quantization == "4bit" and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "4bit" and self.device != "cuda":
            # bitsandbytes nécessite un GPU : on retombe sur fp32/fp16 CPU
            print(f"[LLMWrapper] Pas de GPU détecté, quantization 4bit ignorée pour {hf_id}.")

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, user_message: str, system_message: Optional[str] = None) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        system_message: Optional[str] = None,
        return_logprobs: bool = False,
    ) -> list[GenerationResult]:
        """Génère `n` réponses indépendantes échantillonnées à `temperature`.

        Utilisé tel quel par :
          - self_consistency.py  (n=20, T=0.7)
          - confidence_calibration.py (n=1, return_logprobs=True, pour l'entropie)
        """
        full_prompt = self._build_prompt(prompt, system_message)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        do_sample = temperature > 0
        outputs = self.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_return_sequences=n,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=return_logprobs,
        )

        sequences = outputs.sequences[:, input_len:]
        texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        results = []
        for i, text in enumerate(texts):
            logprobs: list[float] = []
            if return_logprobs:
                # outputs.scores : liste (par step de génération) de logits [n, vocab]
                for step, step_logits in enumerate(outputs.scores):
                    token_id = sequences[i, step].item()
                    if token_id == self.tokenizer.pad_token_id:
                        break
                    log_probs_step = torch.log_softmax(step_logits[i], dim=-1)
                    logprobs.append(log_probs_step[token_id].item())
            results.append(GenerationResult(text=text.strip(), token_logprobs=logprobs))
        return results

    def verbalized_confidence(self, question: str, answer: str, confidence_prompt: str) -> int:
        """Demande au modèle d'auto-évaluer sa confiance (0-100), rapport 2 §2.2.2."""
        prompt = (
            f"Question: {question}\nYour previous answer: {answer}\n\n{confidence_prompt}"
        )
        result = self.generate(prompt, n=1, temperature=0.0, max_new_tokens=8)[0]
        digits = "".join(c for c in result.text if c.isdigit())
        return int(digits) if digits else -1  # -1 = parsing échoué, à filtrer en aval


def token_entropy(logprobs: list[float]) -> float:
    """Entropie de la distribution de sortie, proxy d'incertitude (rapport 2, éq. 2).

    Note : ceci est l'entropie moyenne *du token effectivement choisi* à chaque
    step (approximation standard en l'absence de la distribution complète pour
    chaque step gardée en mémoire) : H ≈ -mean(log p(w_t | w_<t, q)).
    """
    if not logprobs:
        return float("nan")
    return -sum(logprobs) / len(logprobs)


def perplexity_from_logprobs(logprobs: list[float]) -> float:
    if not logprobs:
        return float("nan")
    avg_neg_logprob = -sum(logprobs) / len(logprobs)
    return math.exp(avg_neg_logprob)
