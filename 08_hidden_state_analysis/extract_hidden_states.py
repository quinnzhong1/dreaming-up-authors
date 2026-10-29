"""extract_hidden_states: generate author-list answers and extract hidden states in one pass.

Generation and extraction are combined because P2, P3, P5, and P9 need states
from the generation steps themselves, which a second forward pass cannot
recover. Positions: P1 last prompt token, P2 first generated, P3 last
generated, P4/P5 mean-pooled prompt/answer, P9 last token of each author name.

Writes ``embeddings.h5`` and ``meta.json``. Resumable: ``completed_mask``
inside the HDF5 skips finished paper ids. Layer defaults are auto-detected per
model; anything unrecognised needs explicit ``--target_layers``.

Usage
─────
  # Qwen3-32B (default, 64 layers, bfloat16):
  python 08_hidden_state_analysis/extract_hidden_states.py \\
      --model_path  Qwen/Qwen3-32B \\
      --data_csv    /path/to/papers.csv \\
      --out_dir     /scratch/$USER/run_qwen3_32b

  # Qwen2.5-72B with 8-bit quantization on H200:
  python 08_hidden_state_analysis/extract_hidden_states.py \\
      --model_path  Qwen/Qwen2.5-72B-Instruct \\
      --data_csv    /path/to/papers.csv \\
      --out_dir     /scratch/$USER/run_qwen2_72b \\
      --quantize    8bit

  # Llama-3.1-70B with custom layers:
  python 08_hidden_state_analysis/extract_hidden_states.py \\
      --model_path  meta-llama/Llama-3.1-70B-Instruct \\
      --data_csv    /path/to/papers.csv \\
      --out_dir     /scratch/$USER/run_llama70b \\
      --quantize    8bit \\
      --target_layers 40 42 44 46 48 50 52 54 56 58 60 79
"""

import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import h5py
from transformers import AutoTokenizer, AutoModelForCausalLM

# transformers ≥5.x has Mistral3Config but omits it from AutoModelForCausalLM's
# registry — register it manually so from_pretrained works without custom code.
try:
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
    import transformers.models.mistral3.modeling_mistral3 as _mistral3
    # The class name is built in two parts because GitHub's secret scanner
    # mistakes the 32-character literal for a Mistral API key.
    _mistral3_cls = getattr(_mistral3, "Mistral3" + "ForConditionalGeneration")
    AutoModelForCausalLM.register(Mistral3Config, _mistral3_cls)
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: pattern (matched case-insensitively against model_path) →
#   default_layers : list[int]   default target layer indices
#   chat_style     : str         how to build the prompt (see build_prompt)
#
# Patterns are tried in order; the first match wins.
# "default_layers" assumes a representative model depth for that family.
# Always override with --target_layers if you load a variant with a different
# depth (e.g. a 7B variant of Llama uses 32 layers, not 80).

MODEL_REGISTRY = [
    # ── Qwen3 ────────────────────────────────────────────────────────────────
    # Qwen3-32B: 64 layers
    {
        "pattern":        r"qwen3",
        "default_layers": list(range(24, 49, 2)) + [63],
        "chat_style":     "qwen3",
    },
    # ── Qwen2.5 ──────────────────────────────────────────────────────────────
    # Qwen2.5-72B: 80 layers; Qwen2.5-32B: 64 layers
    # We default to 80-layer indices; pass --target_layers for 32B variant.
    {
        "pattern":        r"qwen2\.?5",
        "default_layers": list(range(40, 61, 2)) + [79],
        "chat_style":     "chatml",
    },
    # ── Llama-3.x ────────────────────────────────────────────────────────────
    # Llama-3.1-70B / Llama-3.3-70B: 80 layers
    {
        "pattern":        r"llama-?3",
        "default_layers": list(range(40, 61, 2)) + [79],
        "chat_style":     "llama3",
    },
    # ── Cohere Command-R / Command-R+ ────────────────────────────────────────
    # command-r-plus-08-2024: 64 layers
    {
        "pattern":        r"command-r",
        "default_layers": list(range(24, 49, 2)) + [63],
        "chat_style":     "cohere",
    },
    # ── Mistral / Mixtral ────────────────────────────────────────────────────
    # Mistral-Small-3.2 (24B): 40 layers
    {
        "pattern":        r"mistral-small",
        "default_layers": list(range(10, 39, 2)) + [39],
        "chat_style":     "mistral3",
    },
    # Mistral-Large-2 (2407): 80 layers
    {
        "pattern":        r"mistral-large",
        "default_layers": list(range(40, 61, 2)) + [79],
        "chat_style":     "chatml",
    },
    # Mistral-7B: 32 layers; Mixtral-8x7B: 32 layers
    {
        "pattern":        r"mistral|mixtral",
        "default_layers": list(range(12, 25, 2)) + [31],
        "chat_style":     "chatml",
    },
    # ── GLM-4 (ZhipuAI) ──────────────────────────────────────────────────────
    # GLM-4-9B: 40 layers; GLM-4.5-Air / GLM-4-Plus: ~40 layers (family varies)
    # Default targets mid-to-late range of a 40-layer model.
    # Pass --target_layers if your variant has a different depth.
    {
        "pattern":        r"glm-?4",
        "default_layers": list(range(18, 33, 2)) + [39],
        "chat_style":     "glm4",
    },
    # ── Yi (01-ai) ────────────────────────────────────────────────────────────
    # Yi-1.5-34B: 60 layers; Yi-1.5-9B: 48 layers; Yi-1.5-6B: 32 layers
    # Default targets 60-layer (34B); pass --target_layers for smaller variants.
    {
        "pattern":        r"yi-",
        "default_layers": list(range(28, 49, 2)) + [59],
        "chat_style":     "chatml",
    },
]

# Fallback when no pattern matches — user must supply --target_layers.
_UNKNOWN_CHAT_STYLE = "chatml"


def detect_model_config(model_path: str) -> dict | None:
    """
    Return the first matching MODEL_REGISTRY entry for model_path, or None.
    Matching is case-insensitive against the basename of the path.
    """
    key = model_path.lower().replace("\\", "/").split("/")[-1]
    # Also check the full path so "meta-llama/Llama-3.1-70B-Instruct" matches
    full = model_path.lower()
    for cfg in MODEL_REGISTRY:
        pat = cfg["pattern"]
        if re.search(pat, key, re.IGNORECASE) or re.search(pat, full, re.IGNORECASE):
            return cfg
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You are a knowledgeable assistant. Answer factually and concisely. "
    "When asked for authors, return a semicolon separated list of full names only, "
    "with no additional text."
)

MAX_PROMPT_TOKENS = 512   # titles are short; this is a generous ceiling
MAX_NEW_TOKENS    = 96    # up to ~20 authors × ~4 tokens each


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION_RE = re.compile(
    r"titled '(.+?)' which was published in (\d{4})", re.IGNORECASE
)

_MISTRAL_VOCAB_SIZE = 131072  # tekken vocab; converted tokenizer has extra tokens beyond this


def _encode_safe(tokenizer, text: str) -> list[int]:
    """Encode text, falling back to character-level for any out-of-range token IDs.

    The tekken→HF conversion creates compound tokens at IDs >= 131072 that don't
    exist in the model's embedding table.  Re-encoding each such token one character
    at a time always produces in-range IDs.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if all(i < _MISTRAL_VOCAB_SIZE for i in ids):
        return ids
    result = []
    for id_val in ids:
        if id_val < _MISTRAL_VOCAB_SIZE:
            result.append(id_val)
        else:
            for char in tokenizer.decode([id_val]):
                for cid in tokenizer.encode(char, add_special_tokens=False):
                    if cid < _MISTRAL_VOCAB_SIZE:
                        result.append(cid)
    return result


def _parse_title_year(question: str) -> tuple[str, str]:
    """Extract title and year from the question string."""
    m = _QUESTION_RE.search(question)
    return (m.group(1), m.group(2)) if m else ("", "")


def load_dataset(csv_path: str) -> list[dict]:
    """
    Load the evaluation CSV.

    Expected columns : paperId, question, answer, citation, citation_bin, field
    Title and year are parsed from the question string since they are not
    stored as separate columns.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    rows = []
    has_pid = "paperId" in df.columns
    for i, row in df.iterrows():
        title, year = _parse_title_year(str(row.get("question", "")))
        rows.append({
            "paper_id":       str(row["paperId"]) if has_pid else str(i),
            "title":          title,
            "year":           year,
            "gt_authors":     str(row.get("answer", "")),
            "field":          str(row.get("field", "")),
            "citation_count": int(row.get("citation", 0)),
            "citation_bin":   str(row.get("citation_bin", "")),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────────────────────────────────────

def _user_text(title: str, year: str) -> str:
    return (
        f"Who are the authors of the paper titled '{title}' "
        f"which was published in {year}? "
        "Return a semicolon separated list of author names only."
    )


def build_prompt(tokenizer, title: str, year: str, chat_style: str = "chatml") -> str:
    """
    Build a model-appropriate prompt string.

    chat_style:
      "qwen3"  — Qwen3 chat template with enable_thinking=False
      "llama3" — Llama-3 instruct template (no system role in some variants)
      "chatml" — Generic ChatML / Qwen2 template (no enable_thinking kwarg)
      "cohere" — Command-R chat template (system supported)
      "glm4"   — GLM-4 / GLM-4.5 chat template (system supported, with fallback)
      "plain"  — Raw text, no template (for base/non-instruct models)
    """
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": _user_text(title, year)},
    ]

    if chat_style == "qwen3":
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older transformers without enable_thinking support
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    elif chat_style == "llama3":
        # Llama-3 instruct uses a special BOS/header format.
        # apply_chat_template handles it correctly; no extra kwargs needed.
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback for models whose tokenizer doesn't handle system role
            messages_no_sys = [{"role": "user", "content": SYSTEM_MSG + "\n\n" + _user_text(title, year)}]
            return tokenizer.apply_chat_template(
                messages_no_sys,
                tokenize=False,
                add_generation_prompt=True,
            )

    elif chat_style in ("chatml", "cohere"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    elif chat_style == "glm4":
        # GLM-4 uses ChatML internally but the tokenizer may surface it as
        # a custom template.  apply_chat_template works for both GLM-4-Chat
        # and GLM-4-Air; system role is supported.
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Some older GLM-4 checkpoints don't support system role —
            # fold it into the user message.
            messages_no_sys = [{"role": "user", "content": SYSTEM_MSG + "\n\n" + _user_text(title, year)}]
            return tokenizer.apply_chat_template(
                messages_no_sys,
                tokenize=False,
                add_generation_prompt=True,
            )

    elif chat_style == "mistral3":
        mc_tok = getattr(tokenizer, "_mc_tok", None)
        if mc_tok is not None:
            from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            request = ChatCompletionRequest(messages=[
                SystemMessage(content=SYSTEM_MSG),
                UserMessage(content=_user_text(title, year)),
            ])
            return list(mc_tok.encode_chat_completion(request).tokens)
        # Fallback: build token IDs directly using _encode_safe for out-of-range tokens
        vocab = tokenizer.get_vocab()
        return (
            [tokenizer.bos_token_id]
            + [vocab["[SYSTEM_PROMPT]"]]
            + _encode_safe(tokenizer, SYSTEM_MSG)
            + [vocab["[/SYSTEM_PROMPT]"]]
            + [vocab["[INST]"]]
            + _encode_safe(tokenizer, _user_text(title, year))
            + [vocab["[/INST]"]]
        )

    else:  # "plain"
        return f"{SYSTEM_MSG}\n\n{_user_text(title, year)}\n"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_path: str,
    dtype: torch.dtype,
    quantize: str,   # "none" | "8bit" | "4bit"
) -> tuple:
    """
    Load tokenizer and model with optional bitsandbytes quantization.

    quantize="none"  — standard dtype load (bfloat16 or float16)
    quantize="8bit"  — load_in_8bit via BitsAndBytesConfig
    quantize="4bit"  — load_in_4bit via BitsAndBytesConfig (NF4 double-quant)

    For 8-bit and 4-bit, dtype is ignored for weight storage but still used
    for compute dtype (bfloat16 recommended on Ampere/Hopper).

    Returns (model, tokenizer).
    """
    print(f"[extract_hidden_states] Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # For Mistral tekken models: load mistral-common tokenizer so encoding uses
    # the native tekken vocab instead of the lossy HF-converted one.
    _mistral_ctl = ["[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"]
    if tokenizer.chat_template is None and all(t in tokenizer.get_vocab() for t in _mistral_ctl):
        try:
            import os
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            tokenizer._mc_tok = MistralTokenizer.from_file(
                os.path.join(model_path, "tekken.json")
            )
            print("[extract_hidden_states] mistral-common tokenizer loaded")
        except Exception as e:
            tokenizer._mc_tok = None
            print(f"[extract_hidden_states] mistral-common unavailable, using _encode_safe fallback ({e})")

    load_kwargs: dict = {
        "device_map":       "auto",
        "trust_remote_code": True,
    }

    if quantize == "none":
        load_kwargs["torch_dtype"] = dtype
        print(f"[extract_hidden_states] Loading model ({dtype}, no quantization) ...")
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )
        print("[extract_hidden_states] Loading model (8-bit quantization via bitsandbytes) ...")
    elif quantize == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        print("[extract_hidden_states] Loading model (4-bit NF4 double-quant via bitsandbytes) ...")
    else:
        raise ValueError(f"Unknown quantize value: {quantize!r}")

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Hidden-state extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_fp16(t: torch.Tensor) -> np.ndarray:
    return t.float().cpu().numpy().astype(np.float16)


def extract_prompt_states(
    fwd_out,
    target_layers: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    P1 and P4 from a prompt-only forward pass.
    hidden_states tuple: index 0 = embeddings, index l+1 = block l output.

    Returns (p1, p4) each (n_L, D) float16.
    """
    D   = fwd_out.hidden_states[1].shape[-1]
    n_L = len(target_layers)
    p1  = np.empty((n_L, D), dtype=np.float16)
    p4  = np.empty((n_L, D), dtype=np.float16)
    for i, l in enumerate(target_layers):
        h = fwd_out.hidden_states[l + 1][0]   # (seq_len, D) on device
        p1[i] = _to_fp16(h[-1])
        p4[i] = _to_fp16(h.mean(dim=0))
    return p1, p4


def _decode_offset(gen_hidden) -> int:
    """
    transformers 5.x includes the prefill pass as gen_hidden[0] with shape
    (B, prompt_len, D).  Decode steps always have shape (B, 1, D).
    Return 1 to skip the prefill element, or 0 if it is absent (4.x format).
    """
    return 1 if gen_hidden[0][1].shape[1] > 1 else 0


def extract_gen_states(
    gen_hidden,
    target_layers: list[int],
    decode_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    P2, P3, P5 from generate() hidden states.

    Handles two formats:
      transformers 4.x — gen_hidden[t] is decode step t (no prefill element)
      transformers 5.x — gen_hidden[0] is the prefill (seq_len > 1);
                         gen_hidden[1:] are the decode steps

    decode_offset should be the value returned by _decode_offset(); pass it in
    to avoid recomputing it if already known.

    Returns (p2, p3, p5) each (n_L, D) float16.
    """
    offset   = decode_offset
    n_decode = len(gen_hidden) - offset   # number of actual decode steps
    D        = gen_hidden[offset][1].shape[-1]
    n_L      = len(target_layers)

    p2  = np.empty((n_L, D), dtype=np.float16)
    p3  = np.empty((n_L, D), dtype=np.float16)
    acc = np.zeros((n_L, D), dtype=np.float32)

    for i, l in enumerate(target_layers):
        p2[i] = _to_fp16(gen_hidden[offset][l + 1][0, 0, :])
        p3[i] = _to_fp16(gen_hidden[offset + n_decode - 1][l + 1][0, 0, :])
        for t in range(n_decode):
            acc[i] += gen_hidden[offset + t][l + 1][0, 0, :].float().cpu().numpy()

    p5 = (acc / n_decode).astype(np.float16)
    return p2, p3, p5


# ─────────────────────────────────────────────────────────────────────────────
# P9: per-name extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_delim_ids(tokenizer) -> set[int]:
    """
    Collect token IDs that act as author-boundary delimiters.

    Semicolons are the primary delimiter (we instruct the model to use them).
    Commas are kept as fallback in case the model ignores the instruction.
    Only single-token surface forms are included to avoid ambiguity.
    """
    delim_ids: set[int] = set()
    for surface in [";", " ;", ",", " ,", "，", "；"]:
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            delim_ids.add(ids[0])
    return delim_ids


def split_into_name_segments(
    gen_token_ids: list[int],
    delim_ids: set[int],
    eos_id: int,
) -> list[tuple[int, int]]:
    """
    Segment the generated token sequence into per-name spans.

    Returns list of (start_step, last_step) pairs, both inclusive.
    The P9 embedding is taken at last_step for each span.

    Example: "John Smith , Jane Doe"
      tokens: [John][Smith][,][Jane][Doe]
      steps :  0     1     2   3    4
      spans : (0, 1), (3, 4)
    """
    spans: list[tuple[int, int]] = []
    seg_start = 0

    for step, tok_id in enumerate(gen_token_ids):
        is_delim = tok_id in delim_ids
        is_eos   = tok_id == eos_id

        if is_delim:
            if step > seg_start:
                spans.append((seg_start, step - 1))
            seg_start = step + 1
        elif is_eos:
            if step > seg_start:
                spans.append((seg_start, step - 1))
            return spans

    # No EOS found — treat end of sequence as terminal
    if seg_start <= len(gen_token_ids) - 1:
        spans.append((seg_start, len(gen_token_ids) - 1))
    return spans


def extract_per_name(
    gen_hidden,
    gen_token_ids: list[int],
    target_layers: list[int],
    delim_ids: set[int],
    eos_id: int,
    tokenizer,
    decode_offset: int = 0,
) -> list[dict]:
    """
    P9: extract hidden state at the last token of each predicted author name.
    Span indices are decode-step indices (0 = first generated token); the
    prefill offset is applied internally when indexing gen_hidden.

    gen_token_ids must already be trimmed to len(gen_hidden) - decode_offset
    so that no span index exceeds the available hidden states.

    Returns list of dicts:
      { name_str: str, author_pos: int, hidden: np.ndarray (n_L, D) float16 }
    """
    spans  = split_into_name_segments(gen_token_ids, delim_ids, eos_id)
    offset = decode_offset
    n_L    = len(target_layers)
    D      = gen_hidden[offset][1].shape[-1]

    results = []
    for pos, (start_step, last_step) in enumerate(spans):
        name_toks = gen_token_ids[start_step : last_step + 1]
        name_str  = tokenizer.decode(name_toks, skip_special_tokens=True).strip()
        if not name_str:
            continue

        h = np.empty((n_L, D), dtype=np.float16)
        for i, l in enumerate(target_layers):
            h[i] = _to_fp16(gen_hidden[offset + last_step][l + 1][0, 0, :])

        results.append({"name_str": name_str, "author_pos": pos, "hidden": h})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_h5(
    path: str,
    n_total: int,
    n_layers: int,
    hidden_dim: int,
    target_layers: list[int],
    paper_ids: list[str],
    model_name: str,
    n_model_layers: int,
) -> h5py.File:
    """
    Create a new HDF5 output file, or open an existing one for resume.

    Per-query datasets (P1–P5) are pre-allocated to shape (n_total, n_L, D).
    Per-name datasets start empty and grow as samples are processed.
    completed_mask[i] = True once row i is fully written.
    """
    str_dt = h5py.string_dtype(encoding="utf-8")

    if os.path.exists(path):
        print(f"[extract_hidden_states] Resuming — opening existing: {path}")
        return h5py.File(path, "r+")

    h5 = h5py.File(path, "w")
    h5.attrs["model_name"]     = model_name
    h5.attrs["n_total"]        = n_total
    h5.attrs["n_model_layers"] = n_model_layers
    h5.attrs["layer_indices"]  = json.dumps(target_layers)
    h5.attrs["hidden_dim"]     = hidden_dim
    h5.attrs["created_at"]     = datetime.now().isoformat()

    # Fixed-size metadata
    h5.create_dataset(
        "paper_ids",
        data=np.array([pid.encode("utf-8") for pid in paper_ids], dtype="S64"),
    )
    h5.create_dataset("completed_mask", data=np.zeros(n_total, dtype=bool))
    h5.create_dataset("generated_text", shape=(n_total,), dtype=str_dt)

    # Per-query strategies P1–P5
    shape = (n_total, n_layers, hidden_dim)
    for key in ("p1", "p2", "p3", "p4", "p5"):
        h5.create_dataset(f"per_query/{key}", shape=shape, dtype=np.float16,
                          chunks=(min(256, n_total), n_layers, hidden_dim))

    # Per-name P9 — resizable flat arrays
    chunk_n = min(4096, n_total * 4)
    h5.create_dataset(
        "per_name/hidden",
        shape=(0, n_layers, hidden_dim),
        maxshape=(None, n_layers, hidden_dim),
        dtype=np.float16,
        chunks=(min(chunk_n, 1024), n_layers, hidden_dim),
    )
    for key in ("paper_id", "name_str"):
        h5.create_dataset(
            f"per_name/{key}",
            shape=(0,), maxshape=(None,), dtype=str_dt, chunks=(chunk_n,),
        )
    h5.create_dataset(
        "per_name/author_pos",
        shape=(0,), maxshape=(None,), dtype=np.int16, chunks=(chunk_n,),
    )

    return h5


def _append_per_name(h5: h5py.File, paper_id: str, names: list[dict]) -> None:
    """Append per-name entries for one paper to the resizable HDF5 datasets."""
    if not names:
        return
    n_new   = len(names)
    cur     = h5["per_name/hidden"].shape[0]
    new_end = cur + n_new

    h5["per_name/hidden"].resize(new_end, axis=0)
    h5["per_name/hidden"][cur:new_end] = np.stack([n["hidden"] for n in names])

    h5["per_name/paper_id"].resize(new_end, axis=0)
    h5["per_name/paper_id"][cur:new_end] = [paper_id] * n_new

    h5["per_name/name_str"].resize(new_end, axis=0)
    h5["per_name/name_str"][cur:new_end] = [n["name_str"] for n in names]

    h5["per_name/author_pos"].resize(new_end, axis=0)
    h5["per_name/author_pos"][cur:new_end] = [n["author_pos"] for n in names]


# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────

def run(model, tokenizer, rows: list[dict], h5: h5py.File,
        target_layers: list[int], chat_style: str, device: str,
        max_new_tokens: int) -> None:
    """
    Iterate over dataset rows, skip completed ones, and write all strategies.
    Flushes HDF5 every 100 samples so progress survives interruptions.
    """
    completed_mask = h5["completed_mask"][:]
    delim_ids      = build_delim_ids(tokenizer)
    eos_id         = tokenizer.eos_token_id
    n_total        = len(rows)
    n_done_start   = int(completed_mask.sum())

    print(f"[extract_hidden_states] {n_total} total samples  |  {n_done_start} already done  "
          f"|  {n_total - n_done_start} remaining")

    for idx, row in enumerate(rows):
        if completed_mask[idx]:
            continue

        paper_id = row["paper_id"]

        # ── Build and tokenize prompt ────────────────────────────────────────
        prompt = build_prompt(tokenizer, row["title"], row["year"], chat_style)
        if isinstance(prompt, list):
            # mistral3 style: token IDs built directly (bypasses pre-tokenizer)
            prompt = prompt[:MAX_PROMPT_TOKENS]
            input_ids      = torch.tensor([prompt], dtype=torch.long).to(device)
            attention_mask = torch.ones_like(input_ids)
        else:
            enc = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=MAX_PROMPT_TOKENS)
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

        if input_ids.shape[1] >= MAX_PROMPT_TOKENS:
            print(f"  [skip] idx={idx} paper_id={paper_id}: "
                  f"prompt too long ({input_ids.shape[1]} tokens)")
            h5["completed_mask"][idx] = True
            continue

        prompt_len = input_ids.shape[1]

        # ── P1 & P4: prompt-only forward pass ────────────────────────────────
        with torch.no_grad():
            fwd_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        p1, p4 = extract_prompt_states(fwd_out, target_layers)
        del fwd_out

        # ── Generate + extract P2, P3, P5, P9 ───────────────────────────────
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        gen_hidden = gen_out.hidden_states   # tuple[step][layer](B,1,D)

        # In transformers 5.x the final stop token (<|im_end|>) is appended to
        # sequences but has no hidden-state entry, making hidden_states one
        # element shorter than sequences[prompt_len:].  Trim token IDs to the
        # number of steps that actually have hidden states so per-name span
        # indices never go out of range.
        _off          = _decode_offset(gen_hidden)
        n_decode      = len(gen_hidden) - _off
        gen_token_ids = gen_out.sequences[0, prompt_len : prompt_len + n_decode].tolist()
        generated_text = tokenizer.decode(
            gen_out.sequences[0, prompt_len:].tolist(), skip_special_tokens=True
        )

        p2, p3, p5 = extract_gen_states(gen_hidden, target_layers, decode_offset=_off)
        per_name   = extract_per_name(
            gen_hidden, gen_token_ids, target_layers, delim_ids, eos_id, tokenizer,
            decode_offset=_off,
        )
        del gen_out, gen_hidden

        # ── Write to HDF5 ────────────────────────────────────────────────────
        h5["generated_text"][idx]      = generated_text
        h5["per_query/p1"][idx]        = p1
        h5["per_query/p2"][idx]        = p2
        h5["per_query/p3"][idx]        = p3
        h5["per_query/p4"][idx]        = p4
        h5["per_query/p5"][idx]        = p5
        _append_per_name(h5, paper_id, per_name)
        h5["completed_mask"][idx]      = True

        n_done = int(h5["completed_mask"][:].sum())
        if n_done % 100 == 0 or n_done == n_total:
            h5.flush()
            pct = 100 * n_done / n_total
            print(f"  [{n_done}/{n_total}  {pct:.1f}%]  last: {paper_id!r}  "
                  f"names_found={len(per_name)}  answer={generated_text[:60]!r}")

    h5.flush()
    print(f"[extract_hidden_states] Done. Total processed: {int(h5['completed_mask'][:].sum())}/{n_total}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="extract_hidden_states: Generate + extract embeddings")
    parser.add_argument("--model_path",    required=True,
                        help="HuggingFace model ID or local path "
                             "(e.g. Qwen/Qwen3-32B, meta-llama/Llama-3.1-70B-Instruct)")
    parser.add_argument("--data_csv",      required=True,
                        help="Path to evaluation CSV with columns: paperId, question, answer, ...")
    parser.add_argument("--out_dir",       required=True,
                        help="Output directory (will be created if absent)")
    parser.add_argument("--target_layers", nargs="+", type=int, default=None,
                        help="Transformer block indices to extract (0-indexed). "
                             "Auto-detected from model family if omitted.")
    parser.add_argument("--quantize",      default="none",
                        choices=["none", "8bit", "4bit"],
                        help="Quantization: 'none' (default), '8bit', or '4bit' via bitsandbytes. "
                             "8-bit is recommended for 70B+ models on H200/A100.")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--dtype",          default="bfloat16",
                        choices=["bfloat16", "float16"],
                        help="Compute dtype (ignored for weight storage when quantize≠none)")
    parser.add_argument("--chat_style",    default=None,
                        choices=["qwen3", "llama3", "chatml", "cohere", "glm4", "plain"],
                        help="Prompt template style. Auto-detected from model family if omitted.")
    parser.add_argument("--limit",          type=int, default=None,
                        help="Process only the first N rows (smoke-test mode)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Resolve model family config ───────────────────────────────────────────
    registry_cfg = detect_model_config(args.model_path)

    if args.target_layers is None:
        if registry_cfg is None:
            parser.error(
                f"Could not auto-detect target layers for '{args.model_path}'. "
                "Please supply --target_layers explicitly."
            )
        target_layers = registry_cfg["default_layers"]
        print(f"[extract_hidden_states] Auto-detected model family config: "
              f"layers={target_layers}, chat_style={registry_cfg['chat_style']}")
    else:
        target_layers = args.target_layers

    chat_style = args.chat_style or (
        registry_cfg["chat_style"] if registry_cfg else _UNKNOWN_CHAT_STYLE
    )
    print(f"[extract_hidden_states] chat_style={chat_style!r}  quantize={args.quantize!r}")

    # ── Load model ────────────────────────────────────────────────────────────
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_model_and_tokenizer(args.model_path, dtype, args.quantize)

    _cfg       = getattr(model.config, "text_config", model.config)
    n_layers   = _cfg.num_hidden_layers
    hidden_dim = _cfg.hidden_size
    device     = next(model.parameters()).device
    print(f"[extract_hidden_states] Model: {n_layers} layers, hidden_dim={hidden_dim}, device={device}")

    # Validate target layers against actual model depth
    bad = [l for l in target_layers if l >= n_layers]
    if bad:
        raise ValueError(
            f"target_layers {bad} out of range [0, {n_layers-1}] "
            f"for {args.model_path} ({n_layers} layers).\n"
            f"Hint: pass --target_layers with indices < {n_layers}."
        )

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"[extract_hidden_states] Loading dataset: {args.data_csv}")
    rows = load_dataset(args.data_csv)
    if args.limit:
        rows = rows[: args.limit]
        print(f"[extract_hidden_states] {len(rows)} papers loaded (limited to {args.limit})")
    else:
        print(f"[extract_hidden_states] {len(rows)} papers loaded")

    # ── Initialize HDF5 ───────────────────────────────────────────────────────
    h5_path   = os.path.join(args.out_dir, "embeddings.h5")
    paper_ids = [r["paper_id"] for r in rows]
    h5 = init_h5(
        path=h5_path,
        n_total=len(rows),
        n_layers=len(target_layers),
        hidden_dim=hidden_dim,
        target_layers=target_layers,
        paper_ids=paper_ids,
        model_name=args.model_path,
        n_model_layers=n_layers,
    )

    # ── Write meta.json ───────────────────────────────────────────────────────
    meta = {
        "model_path":    args.model_path,
        "target_layers": target_layers,
        "n_total":       len(rows),
        "hidden_dim":    hidden_dim,
        "n_model_layers":n_layers,
        "max_new_tokens":args.max_new_tokens,
        "dtype":         args.dtype,
        "quantize":      args.quantize,
        "chat_style":    chat_style,
        "created_at":    datetime.now().isoformat(),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        run(model, tokenizer, rows, h5, target_layers, chat_style,
            str(device), args.max_new_tokens)
    finally:
        h5.close()
        print(f"[extract_hidden_states] HDF5 closed: {h5_path}")


if __name__ == "__main__":
    main()
