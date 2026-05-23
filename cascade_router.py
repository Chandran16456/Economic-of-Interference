"""
cascade_router.py — A cost-aware LLM router most startups build wrong.
═════════════════════════════════════════════════════════════════════

THE IDEA
────────
Try the cheap model first. Escalate to the expensive model only when the
cheap model is uncertain. On typical production workloads this cuts spend
by 40–70% with no measurable quality loss — *if* you calibrate the
escalation threshold against a real eval set.

Most teams do not calibrate. They pick a number out of the air, ship it,
and either escalate too often (no savings) or too rarely (quality drops).
This module includes a one-shot calibration routine that picks the
cheapest threshold meeting your quality bar. It is the part everyone
skips, and it is the part that actually matters.

WHEN TO USE THIS
────────────────
- Production traffic where most requests are easy and some are hard
  (classification, extraction, summarization, simple Q&A).
- Workloads that already have, or can cheaply produce, a held-out
  eval set with a programmatic correctness check.

WHEN NOT TO USE THIS
────────────────────
- Open-ended creative generation where "quality" cannot be reduced to a
  boolean check. Use human eval and a static routing rule instead.
- Workloads where latency dominates cost; the cheap call you pay for
  before escalating adds 100–300 ms.

USAGE
─────
    from cascade_router import (
        CascadeRouter, ModelTier, EvalCase, calibrate, openai_caller,
    )

    cheap = ModelTier(
        name="gpt-4o-mini",
        price_in_per_1k=0.00015, price_out_per_1k=0.00060,
        call=openai_caller("gpt-4o-mini"),
    )
    expensive = ModelTier(
        name="gpt-4o",
        price_in_per_1k=0.0025,  price_out_per_1k=0.01,
        call=openai_caller("gpt-4o"),
    )

    router = CascadeRouter(cheap, expensive)

    # One-time calibration against your eval set:
    best_t = calibrate(router, eval_set, target_quality=0.95, verbose=True)
    router.threshold = best_t           # pin and ship

    # In production:
    out = router.route("What is the capital of France?")
    print(out.text, "→", out.model_used, f"${out.cost_usd:.5f}")

LICENSE
───────
Public domain. Take it, modify it, ship it. No attribution required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelTier:
    """One tier in a cascade.

    The `call` callable must accept (prompt, **kwargs) and return a dict:
        {
          "text":       str,           # generated text
          "logprobs":   list[float],   # log-probs of chosen tokens (or None)
          "tokens_in":  int,
          "tokens_out": int,
        }
    """
    name: str
    price_in_per_1k: float    # USD per 1k input tokens
    price_out_per_1k: float   # USD per 1k output tokens
    call: Callable[..., Dict[str, Any]]


@dataclass
class RouterResult:
    text: str
    model_used: str
    cost_usd: float
    escalated: bool
    confidence: float
    tokens_in: int
    tokens_out: int


@dataclass
class EvalCase:
    """One row of your held-out evaluation set."""
    prompt: str
    check: Callable[[str], bool]   # True iff the response is acceptable


# ─────────────────────────────────────────────────────────────────────────────
# The router
# ─────────────────────────────────────────────────────────────────────────────

class CascadeRouter:
    """
    Cascade: try `cheap` first, escalate to `expensive` only when the cheap
    model's confidence falls below `threshold`. Confidence is the geometric
    mean of top-token probabilities over the first `head_tokens` decoded
    tokens — a simple but well-calibrated proxy on most providers.

    If your provider does not expose log-probabilities (e.g. Anthropic at
    time of writing), supply a custom `confidence_fn(cheap_result) -> float`
    in [0, 1]. A reasonable alternative is self-consistency: re-ask the
    cheap model n times and use the agreement rate as the confidence.
    """

    def __init__(
        self,
        cheap: ModelTier,
        expensive: ModelTier,
        threshold: float = 0.80,
        head_tokens: int = 8,
        confidence_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ):
        self.cheap = cheap
        self.expensive = expensive
        self.threshold = threshold
        self.head_tokens = head_tokens
        self._confidence_fn = confidence_fn

    def route(self, prompt: str, **kwargs) -> RouterResult:
        cheap_out = self.cheap.call(prompt, logprobs=True, **kwargs)
        cheap_cost = self._cost(self.cheap, cheap_out["tokens_in"], cheap_out["tokens_out"])
        confidence = self._confidence(cheap_out)

        if confidence >= self.threshold:
            return RouterResult(
                text=cheap_out["text"],
                model_used=self.cheap.name,
                cost_usd=cheap_cost,
                escalated=False,
                confidence=confidence,
                tokens_in=cheap_out["tokens_in"],
                tokens_out=cheap_out["tokens_out"],
            )

        # Escalate. We still paid for the cheap call.
        exp_out = self.expensive.call(prompt, **kwargs)
        exp_cost = self._cost(self.expensive, exp_out["tokens_in"], exp_out["tokens_out"])
        return RouterResult(
            text=exp_out["text"],
            model_used=self.expensive.name,
            cost_usd=cheap_cost + exp_cost,
            escalated=True,
            confidence=confidence,
            tokens_in=cheap_out["tokens_in"] + exp_out["tokens_in"],
            tokens_out=cheap_out["tokens_out"] + exp_out["tokens_out"],
        )

    # ── internals ────────────────────────────────────────────────────────────

    def _confidence(self, cheap_result: Dict[str, Any]) -> float:
        if self._confidence_fn is not None:
            return self._confidence_fn(cheap_result)
        lp = cheap_result.get("logprobs") or []
        if not lp:
            return 0.0
        head = lp[: self.head_tokens]
        return math.exp(sum(head) / len(head))

    @staticmethod
    def _cost(tier: ModelTier, t_in: int, t_out: int) -> float:
        return (t_in / 1000) * tier.price_in_per_1k + (t_out / 1000) * tier.price_out_per_1k


# ─────────────────────────────────────────────────────────────────────────────
# Calibration — the part everyone skips
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(
    router: CascadeRouter,
    eval_set: List[EvalCase],
    target_quality: float = 0.95,
    thresholds: Optional[List[float]] = None,
    verbose: bool = False,
) -> Optional[float]:
    """
    Sweep escalation thresholds; return the cheapest one that maintains at
    least `target_quality` on the eval set.

    Returns the chosen threshold, or None if no threshold hits the quality
    bar — in which case your cheap model is not good enough and the cascade
    cannot help. Try a stronger cheap model or accept a lower quality bar.

    Cost of running this once:  ~ len(thresholds) × len(eval_set) requests
    through the router. Use a small but representative eval set (50–200
    cases is usually enough) and pin the resulting threshold in production.
    """
    thresholds = thresholds or [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    best_t: Optional[float] = None
    best_cost = float("inf")

    if verbose:
        print(f"{'threshold':>10}  {'quality':>9}  {'escalated':>10}  {'cost (USD)':>12}")
        print("─" * 50)

    for t in thresholds:
        router.threshold = t
        results = [router.route(c.prompt) for c in eval_set]

        quality = sum(c.check(r.text) for c, r in zip(eval_set, results)) / len(eval_set)
        cost = sum(r.cost_usd for r in results)
        escalation_rate = sum(r.escalated for r in results) / len(results)

        if verbose:
            print(f"{t:>10.2f}  {quality:>8.1%}  {escalation_rate:>9.1%}  ${cost:>11.4f}")

        if quality >= target_quality and cost < best_cost:
            best_t, best_cost = t, cost

    if verbose and best_t is not None:
        print(f"\n→ Pin threshold = {best_t}  (cheapest above {target_quality:.0%} quality)")

    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# Optional provider adapters
# ─────────────────────────────────────────────────────────────────────────────

def openai_caller(model: str):
    """Caller for OpenAI chat-completions, returning real token-level
    log-probabilities. `pip install openai`."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e

    client = OpenAI()

    def call(prompt: str, system: Optional[str] = None, logprobs: bool = False, **_):
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            logprobs=logprobs,
            top_logprobs=1 if logprobs else None,
            max_tokens=1024,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""

        lp: Optional[List[float]] = None
        if logprobs and choice.logprobs and choice.logprobs.content:
            lp = [tok.logprob for tok in choice.logprobs.content]

        return {
            "text": text,
            "logprobs": lp,
            "tokens_in": resp.usage.prompt_tokens,
            "tokens_out": resp.usage.completion_tokens,
        }
    return call


def anthropic_caller(model: str):
    """Caller for Anthropic Messages API. Anthropic does not expose token
    log-probabilities, so pair this with a custom `confidence_fn` —
    e.g. self-consistency (re-ask n times, use agreement rate) or a
    structural check (does the JSON parse?). `pip install anthropic`."""
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError("pip install anthropic") from e

    client = Anthropic()

    def call(prompt: str, system: Optional[str] = None, **_):
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        return {
            "text": text,
            "logprobs": None,
            "tokens_in": msg.usage.input_tokens,
            "tokens_out": msg.usage.output_tokens,
        }
    return call


# ─────────────────────────────────────────────────────────────────────────────
# Self-contained demo with mocked providers
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    # A fake cheap model: right 70% of the time, confident when right.
    def mock_cheap(prompt, logprobs=False, **_):
        right = random.random() < 0.7
        lp = ([-0.05] * 8 if right else [-2.5] * 8) if logprobs else None
        return {
            "text": "right" if right else "wrong",
            "logprobs": lp,
            "tokens_in": 10,
            "tokens_out": 5,
        }

    # A fake expensive model: right 99% of the time.
    def mock_expensive(prompt, **_):
        return {
            "text": "right" if random.random() < 0.99 else "wrong",
            "tokens_in": 10,
            "tokens_out": 5,
        }

    cheap = ModelTier("cheap-mock", 0.00025, 0.00125, mock_cheap)
    expensive = ModelTier("expensive-mock", 0.015, 0.075, mock_expensive)
    router = CascadeRouter(cheap, expensive)

    eval_set = [EvalCase(prompt=f"q{i}", check=lambda t: t == "right") for i in range(200)]

    print("Calibrating cascade router on a 200-case eval set:\n")
    best_t = calibrate(router, eval_set, target_quality=0.95, verbose=True)
