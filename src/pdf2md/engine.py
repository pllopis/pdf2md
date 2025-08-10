# engine.py
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info

LOG = logging.getLogger("pdf2md.engine")

# ---------------- flash_attn shim for non-CUDA ----------------
import sys as _sys, types as _types, importlib.machinery as _machinery
if not torch.cuda.is_available():
    if "flash_attn" not in _sys.modules:
        _m = _types.ModuleType("flash_attn")
        _m.__spec__ = _machinery.ModuleSpec(name="flash_attn", loader=None)
        _m.__path__ = []
        def flash_attn_varlen_func(*args, **kwargs):
            raise RuntimeError("flash_attn isn't available here. Use attn_implementation='sdpa'.")
        _m.flash_attn_varlen_func = flash_attn_varlen_func
        _sys.modules["flash_attn"] = _m
# ---------------------------------------------------------------

# =========================
# Prompt handling
# =========================

DEFAULT_JSON_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.
    - Any Markdown Text fields should be adequately escaped to allow them to be contained within JSON strings.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
""".strip()

def _load_prompt_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        txt = p.read_text(encoding="utf-8").strip()
        return txt or None
    return None

def resolve_prompt(prompt: Optional[str], prompt_mode: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt:
        return prompt.strip()
    pf = _load_prompt_file(prompt_file)
    if pf:
        return pf
    if prompt_mode:
        LOG.warning("`--prompt-mode` provided but repo prompts are not available. Using built-in default prompt.")
    return DEFAULT_JSON_PROMPT

# =========================
# JSON helpers
# =========================

def json_pre_repair(s: str) -> str:
    """Lightweight cleanup of common LLM JSON glitches (non-destructive)."""
    s = s.strip()

    # Strip code fences, if any
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

    # DO NOT convert curly quotes to straight quotes; this can break valid JSON strings.
    # Keep smart quotes as-is. JSON supports Unicode characters inside strings.
    # s = s.replace("“", '"').replace("”", '"').replace("’", "'")

    # Normalize non-breaking spaces (seen often in PDFs)
    s = s.replace("\u00a0", " ")

    # Remove trailing commas before ] or }
    s = re.sub(r",(\s*[}\]])", r"\1", s)

    return s

def parse_json_flex(s: str) -> Any:
    t = s.strip()
    if t.startswith("{"):
        return json.loads(t[: t.rfind("}") + 1])
    if t.startswith("["):
        return json.loads(t[: t.rfind("]") + 1])
    m = re.search(r"\{[\s\S]*\}", s) or re.search(r"$begin:math:display$[\\s\\S]*$end:math:display$", s)
    if not m:
        raise ValueError("No JSON object/array found in model output.")
    return json.loads(m.group(0))

def to_blocks(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, list):
        return {"blocks": obj}
    if isinstance(obj, dict) and "blocks" in obj:
        return obj
    if isinstance(obj, dict):
        for k in ("items", "elements", "segments"):
            if k in obj and isinstance(obj[k], list):
                return {"blocks": obj[k]}
    return {"blocks": [{"category": "Text", "text": json.dumps(obj, ensure_ascii=False)}]}

# =========================
# Markdown helpers
# =========================

def filter_blocks(blocks: List[dict], drop_categories: Iterable[str]) -> List[dict]:
    drop = {c.strip() for c in drop_categories}
    return [b for b in blocks if b.get("category") not in drop]

def sort_blocks_reading_order(blocks: List[dict]) -> List[dict]:
    def key(b):
        bbox = b.get("bbox") or [0, 0, 0, 0]
        x0, y0 = (bbox[0], bbox[1]) if len(bbox) >= 2 else (0, 0)
        return (y0, x0)
    return sorted(blocks, key=key)

def _sanitize_heading(text: str) -> str:
    return text.strip().lstrip("# ").strip()

def block_to_markdown(b: dict) -> str:
    cat = (b.get("category") or "").lower()
    txt = (b.get("text") or "").strip()
    if not txt:
        return ""
    if cat in {"title"}:
        return f"# {_sanitize_heading(txt)}"
    if cat in {"section-header", "section_header", "heading", "header"}:
        return f"## {_sanitize_heading(txt)}"
    if cat in {"subheading", "subtitle"}:
        return f"### {_sanitize_heading(txt)}"
    if cat in {"quote", "blockquote"}:
        return "> " + txt.replace("\n", "\n> ")
    if cat in {"table", "list", "bullet_list", "numbered_list"}:
        return txt
    return txt

def blocks_to_markdown(blocks: List[dict], sort_blocks: bool) -> str:
    if sort_blocks:
        blocks = sort_blocks_reading_order(blocks)
    md_lines: List[str] = []
    for b in blocks:
        line = block_to_markdown(b)
        if line:
            md_lines.append(line)
    return "\n\n".join(md_lines).strip() + ("\n" if md_lines else "")

# =========================
# Model acquisition
# =========================

def ensure_local_model(model_dir: Path, model_id: Optional[str]) -> None:
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        LOG.info("Using local model from %s", model_dir)
        return
    if not model_id:
        raise RuntimeError(f"Model directory {model_dir} is missing; provide --model-id to download.")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError("huggingface_hub is required to download models. pip install huggingface_hub") from e
    LOG.info("Downloading model '%s' into %s ...", model_id, model_dir)
    snapshot_download(repo_id=model_id, local_dir=str(model_dir), local_dir_use_symlinks=False)
    if not cfg_path.exists():
        raise RuntimeError(f"Download finished but config.json not found in {model_dir}")

# =========================
# Engine
# =========================

@dataclass
class DotsConfig:
    model_dir: str
    model_id: Optional[str]
    dpi: int
    max_new_tokens: int = 1536
    prompt: Optional[str] = None
    prompt_file: Optional[str] = None
    prompt_mode: Optional[str] = None
    retry_attempts: int = 2
    retry_temperatures: Tuple[float, ...] = (0.0, 0.2, 0.7)
    strict_json_suffix: str = (
        "\n\nReturn ONLY valid JSON. Do not include backticks, explanations, or any extra text."
    )
    json_repair: bool = True

    # vLLM options (optional)
    use_vllm: bool = False
    vllm_ip: str = "localhost"
    vllm_port: int = 8000
    vllm_model_name: str = "model"
    vllm_temperature: float = 0.1
    vllm_top_p: float = 1.0
    vllm_max_tokens: int = 16384

class DotsRunner:
    def __init__(self, cfg: DotsConfig):
        self.cfg = cfg

        # If using vLLM, we don't load a local model at all.
        self._use_vllm = bool(cfg.use_vllm)
        self._vllm_infer = None

        if self._use_vllm:
            try:
                from dots_ocr.model.inference import inference_with_vllm
                self._vllm_infer = inference_with_vllm
                LOG.info("Using vLLM backend at %s:%d", cfg.vllm_ip, cfg.vllm_port)
            except Exception as e:
                raise RuntimeError(
                    "use_vllm=True, but dots_ocr.model.inference.inference_with_vllm "
                    "is not importable. Install dots.ocr or set use_vllm=False."
                ) from e
            # We still need a processor for prompt construction if we ever add token-level logic,
            # but for now vLLM path builds a raw text prompt and sends the PIL image to the server.
            self.processor = None
            self.tokenizer = None
            self.model = None
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            self.dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else (
                torch.float16 if self.device in ("cuda", "mps") else torch.float32
            )
        else:
            # HF in-process path
            if torch.cuda.is_available():
                self.device = "cuda"
                self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.dtype = torch.float16
            else:
                self.device = "cpu"
                self.dtype = torch.float32

            model_dir = Path(cfg.model_dir)
            ensure_local_model(model_dir, cfg.model_id)

            LOG.info("Loading model from %s", cfg.model_dir)
            config = AutoConfig.from_pretrained(cfg.model_dir, trust_remote_code=True, local_files_only=True)

            attn_impl = "sdpa"
            if self.device == "cuda":
                try:
                    import flash_attn  # noqa: F401
                    attn_impl = "flash_attention_2"
                except Exception:
                    attn_impl = "sdpa"

            if getattr(config, "vision_config", None) is not None:
                config.vision_config.attn_implementation = attn_impl
            else:
                setattr(config, "attn_implementation", attn_impl)
            if hasattr(config, "use_sliding_window"):
                config.use_sliding_window = False
            if hasattr(config, "sliding_window"):
                config.sliding_window = None

            self.processor = AutoProcessor.from_pretrained(
                cfg.model_dir, local_files_only=True, trust_remote_code=True, use_fast=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_dir, trust_remote_code=True, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_dir,
                local_files_only=True,
                trust_remote_code=True,
                config=config,
                attn_implementation=attn_impl,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )

            LOG.info("device=%s dtype=%s attn=%s use_sliding_window=%s",
                     self.device, str(self.dtype).replace("torch.", ""), attn_impl,
                     getattr(self.model.config, "use_sliding_window", None))

        self.base_prompt = resolve_prompt(cfg.prompt, cfg.prompt_mode, cfg.prompt_file)
        LOG.debug("Using prompt base (len=%d chars)", len(self.base_prompt))

    @staticmethod
    def pil_from_page(page, dpi: int) -> Image.Image:
        pix = page.get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def build_messages(self, img: Image.Image, strict: bool) -> List[dict]:
        text = self.base_prompt + (self.cfg.strict_json_suffix if strict else "")
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": text},
            ],
        }]

    # Build the same raw prompt string the parser uses:
    @staticmethod
    def _render_chat_text(prompt_text: str) -> str:
        return f" <|img|><|imgpad|><|endofimg|>{prompt_text}\n<|assistant|>"

    def _generate_once_hf(self, messages: List[dict], temperature: float) -> str:
        # 1) Build raw template
        try:
            user_content = messages[0]["content"]
            prompt_text = next(c["text"] for c in user_content if c.get("type") == "text")
        except Exception:
            prompt_text = self.base_prompt
        chat_text = self._render_chat_text(prompt_text)

        # 2) Pack vision + text
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 3) Generate
        inputs = inputs.to(self.device)
        do_sample = (temperature > 0.0) and (self.device != "mps")
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=None if not do_sample else temperature,
                do_sample=do_sample,
            )

        trimmed = [o[len(iids):] for iids, o in zip(inputs["input_ids"], out_ids)]
        out_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return out_text

    def _generate_once_vllm(self, messages: List[dict]) -> str:
        # Extract the PIL image + prompt text
        try:
            user_content = messages[0]["content"]
            img: Image.Image = next(c["image"] for c in user_content if c.get("type") == "image")
            prompt_text = next(c["text"] for c in user_content if c.get("type") == "text")
        except StopIteration as e:
            raise RuntimeError("vLLM path requires an image and a text prompt in messages[0].content") from e

        # Call the same helper the upstream parser uses
        resp = self._vllm_infer(
            img,
            prompt_text,
            model_name=self.cfg.vllm_model_name,
            ip=self.cfg.vllm_ip,
            port=self.cfg.vllm_port,
            temperature=self.cfg.vllm_temperature,
            top_p=self.cfg.vllm_top_p,
            max_completion_tokens=self.cfg.vllm_max_tokens,
        )
        return resp

    def _generate_once(self, messages: List[dict], temperature: float) -> str:
        if self._use_vllm:
            return self._generate_once_vllm(messages)
        return self._generate_once_hf(messages, temperature)

    def infer_json_with_retries(self, img: Image.Image) -> Dict[str, Any]:
        attempts = max(1, self.cfg.retry_attempts + 1)
        temps = list(self.cfg.retry_temperatures)
        if len(temps) < attempts:
            temps += [temps[-1]] * (attempts - len(temps))

        last_err: Optional[Exception] = None
        for i in range(attempts):
            strict = (i > 0)
            temperature = temps[i]
            LOG.debug("Generate try %d (strict=%s, temp=%.2f)", i + 1, strict, temperature)
            try:
                out_text = self._generate_once(self.build_messages(img, strict), temperature)
                text_for_parse = json_pre_repair(out_text) if self.cfg.json_repair else out_text
                parsed = parse_json_flex(text_for_parse)
                return to_blocks(parsed)
            except json.JSONDecodeError as e:
                raw = locals().get("out_text", "")
                start = max(0, e.pos - 120); end = min(len(raw), e.pos + 120)
                LOG.debug("JSON decode error context (attempt %d):\n%s\n%s",
                          i + 1, raw[start:end], " " * (e.pos - start) + "^")
                last_err = e
                LOG.warning("JSON parse failed on attempt %d: %s", i + 1, e)
            except Exception as e:
                last_err = e
                LOG.warning("JSON parse/generate failed on attempt %d: %s", i + 1, e)
        assert last_err is not None
        raise last_err