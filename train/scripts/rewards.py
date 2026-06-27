"""GRPO reward functions for agent harness alignment (Cursor, Zed, BFCL-style tools)."""

from __future__ import annotations

import json
import re
from typing import Any

from dataset_mix import TOOL_CALL_END, TOOL_CALL_START

# LFM2.5 bracket tool call: <|tool_call_start|>[fn(arg="x")]<|tool_call_end|>
LFM_TOOL_CALL_RE = re.compile(
    rf"{re.escape(TOOL_CALL_START)}\[(?P<body>[^\]]+)\]{re.escape(TOOL_CALL_END)}"
)
LFM_FUNC_RE = re.compile(r"^(?P<name>[a-zA-Z_][\w]*)\((?P<args>.*)\)$")


def _extract_completions(kwargs: dict[str, Any]) -> list[str]:
    completions = kwargs.get("completions")
    if completions is None:
        raise KeyError("reward functions expect `completions` from GRPOTrainer")
    if completions and isinstance(completions[0], list):
        return [c[-1]["content"] if isinstance(c[-1], dict) else str(c[-1]) for c in completions]
    return [str(c) for c in completions]


def lfm_tool_format_reward(completions, **kwargs) -> list[float]:  # noqa: ANN001
    """Reward valid LFM2.5 bracket tool-call syntax."""
    texts = _extract_completions({"completions": completions, **kwargs})
    scores: list[float] = []
    for text in texts:
        matches = LFM_TOOL_CALL_RE.findall(text)
        if not matches:
            scores.append(0.0)
            continue
        valid = 0
        for body in matches:
            if LFM_FUNC_RE.match(body.strip()):
                valid += 1
        scores.append(valid / len(matches))
    return scores


def tool_json_args_reward(completions, **kwargs) -> list[float]:  # noqa: ANN001
    """Reward parseable keyword arguments inside LFM tool calls."""
    texts = _extract_completions({"completions": completions, **kwargs})
    scores: list[float] = []
    for text in texts:
        matches = LFM_TOOL_CALL_RE.findall(text)
        if not matches:
            scores.append(0.0)
            continue
        ok = 0
        for body in matches:
            m = LFM_FUNC_RE.match(body.strip())
            if not m:
                continue
            args_str = m.group("args").strip()
            if not args_str:
                ok += 1
                continue
            # Basic sanity: balanced quotes and no dangling commas
            try:
                # Convert pythonic kwargs to JSON-ish dict for validation
                fake = "{" + ", ".join(
                    f'"{k.strip()}": {v.strip()}'
                    for k, v in (
                        part.split("=", 1) for part in _split_kwargs(args_str)
                    )
                ) + "}"
                json.loads(fake.replace("'", '"').replace("True", "true").replace("False", "false"))
                ok += 1
            except (ValueError, json.JSONDecodeError):
                if "=" in args_str and args_str.count('"') % 2 == 0:
                    ok += 0.5
        scores.append(ok / max(len(matches), 1))
    return scores


def _split_kwargs(args_str: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_str = False
    quote = ""
    for ch in args_str:
        if ch in "\"'" and (not in_str or ch == quote):
            in_str = not in_str
            quote = ch if in_str else ""
        if not in_str:
            if ch in "({[":
                depth += 1
            elif ch in ")}]":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


def concise_reward(completions, **kwargs) -> list[float]:  # noqa: ANN001
    """Prefer concise agent responses (helps Cursor/Zed UX). Target ~256-768 tokens."""
    texts = _extract_completions({"completions": completions, **kwargs})
    max_good = float(kwargs.get("max_good_chars", 1200))
    scores: list[float] = []
    for text in texts:
        n = len(text)
        if n <= max_good:
            scores.append(1.0)
        elif n <= max_good * 2:
            scores.append(0.5)
        else:
            scores.append(0.1)
    return scores


def solution_match_reward(completions, solution, **kwargs) -> list[float]:  # noqa: ANN001
    """Exact/normalized match against dataset solution (tool calls, boxed math, etc.)."""
    texts = _extract_completions({"completions": completions, **kwargs})
    solutions = solution if isinstance(solution, list) else [solution] * len(texts)
    scores: list[float] = []
    for text, ref in zip(texts, solutions):
        ref_s = str(ref).strip()
        if not ref_s:
            scores.append(0.0)
            continue
        norm_text = _normalize(text)
        norm_ref = _normalize(ref_s)
        if norm_ref in norm_text or norm_text == norm_ref:
            scores.append(1.0)
        elif _tool_names_match(text, ref_s):
            scores.append(0.8)
        else:
            scores.append(0.0)
    return scores


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _tool_names_match(completion: str, reference: str) -> bool:
    comp_names: set[str] = set()
    for body in LFM_TOOL_CALL_RE.findall(completion):
        m = LFM_FUNC_RE.match(body.strip())
        if m:
            comp_names.add(m.group("name"))
    ref_names = set(re.findall(r'"name"\s*:\s*"([^"]+)"', reference))
    for body in LFM_TOOL_CALL_RE.findall(reference):
        m = LFM_FUNC_RE.match(body.strip())
        if m:
            ref_names.add(m.group("name"))
    return bool(comp_names and ref_names and comp_names == ref_names)


def ifeval_keyword_reward(completions, prompts, **kwargs) -> list[float]:  # noqa: ANN001
    """Lightweight instruction checks when prompt embeds constraints."""
    texts = _extract_completions({"completions": completions, **kwargs})
    prompt_list = prompts if isinstance(prompts, list) else [prompts] * len(texts)
    scores: list[float] = []
    for text, prompt in zip(texts, prompt_list):
        prompt_s = _prompt_text(prompt)
        score = 1.0
        lower = prompt_s.lower()
        if "json" in lower and "only" in lower:
            score *= 1.0 if _looks_like_json(text) else 0.0
        if "one sentence" in lower or "single sentence" in lower:
            score *= 1.0 if text.count(".") <= 1 and len(text) < 300 else 0.3
        if "bullet" in lower or "list" in lower:
            score *= 1.0 if ("-" in text or "•" in text or re.search(r"\d+\.", text)) else 0.4
        scores.append(score)
    return scores


def _prompt_text(prompt) -> str:  # noqa: ANN001
    if isinstance(prompt, list):
        return " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else str(m.get("content", ""))
            for m in prompt
        )
    return str(prompt)


def _looks_like_json(text: str) -> bool:
    text = text.strip()
    if not (text.startswith("{") or text.startswith("[")):
        return False
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


REWARD_REGISTRY = {
    "lfm_tool_format": lfm_tool_format_reward,
    "tool_json_args": tool_json_args_reward,
    "concise": concise_reward,
    "solution_match": solution_match_reward,
    "ifeval_keyword": ifeval_keyword_reward,
}


def build_reward_funcs(names: list[str], weights: list[float] | None = None):
    funcs = [REWARD_REGISTRY[n] for n in names]
    if weights:
        w = weights

        def weighted(*args, **kwargs):  # noqa: ANN001
            totals = [0.0] * len(args[0])
            for fn, weight in zip(funcs, w):
                scores = fn(*args, **kwargs)
                totals = [t + weight * s for t, s in zip(totals, scores)]
            return totals

        return weighted
    return funcs
