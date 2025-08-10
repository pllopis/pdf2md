# cli.py
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import torch

from .engine import (
    DotsConfig,
    DotsRunner,
    blocks_to_markdown,
    filter_blocks,
)

LOG = logging.getLogger("pdf2md")


# =========================
# Logging
# =========================

def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("pdf2md").setLevel(level)
    logging.getLogger("pdf2md.engine").setLevel(level)

# =========================
# Resume cache helpers
# =========================

def default_cache_dir_for(pdf_path: Path) -> Path:
    base = Path(".pdf2md_cache")
    return base / pdf_path.stem

def page_basename(i: int) -> str:
    return f"page_{i:05}"

def cache_paths(cache_dir: Path, i: int) -> Tuple[Path, Path]:
    base = cache_dir / page_basename(i)
    return base.with_suffix(".json"), base.with_suffix(".md")

def compute_fingerprint(meta: Dict[str, Any]) -> str:
    j = json.dumps(meta, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()

def write_meta(cache_dir: Path, meta: Dict[str, Any]) -> None:
    meta["fingerprint"] = compute_fingerprint(meta)
    (cache_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def read_meta(cache_dir: Path) -> Optional[Dict[str, Any]]:
    p = cache_dir / "_meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def list_cached_pages(cache_dir: Path) -> List[int]:
    pages = []
    if not cache_dir.exists():
        return pages
    for p in cache_dir.glob("page_*.md"):
        try:
            pages.append(int(p.stem.split("_")[1]))
        except Exception:
            continue
    pages.sort()
    return pages

def assemble_from_cache(cache_dir: Path, total_pages: int) -> List[str]:
    md_pages: List[str] = []
    for i in range(1, total_pages + 1):
        _, md_path = cache_paths(cache_dir, i)
        if md_path.exists():
            md_pages.append(md_path.read_text(encoding="utf-8"))
        else:
            md_pages.append("")
    return md_pages

def copy_cached_pages_to_dir(cache_dir: Path, output_dir: Path, total_pages: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, total_pages + 1):
        _, md_path = cache_paths(cache_dir, i)
        out_path = output_dir / f"page_{i:03}.md"
        if md_path.exists():
            out_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            out_path.write_text("", encoding="utf-8")

def extract_picture_blocks(pil_page, blocks, page_index: int, cache_dir: Path, combined_output: bool) -> List[dict]:
    """
    For each Picture block, crop the bbox out of the PIL page, save it, and
    replace the block's text with a Markdown image tag so it renders.
    Returns modified blocks.
    """
    W, H = pil_page.size
    images_dir = cache_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    out_blocks = []
    pic_counter = 0

    for b in blocks:
        cat = (b.get("category") or "").lower()
        if cat != "picture":
            out_blocks.append(b)
            continue

        bbox = b.get("bbox") or [0, 0, 0, 0]
        if len(bbox) != 4:
            out_blocks.append(b)
            continue

        x0, y0, x1, y1 = map(int, bbox)
        # clamp to page bounds
        x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
        if x1 <= x0 or y1 <= y0:
            out_blocks.append(b)
            continue

        crop = pil_page.crop((x0, y0, x1, y1))
        pic_counter += 1
        fname = f"page_{page_index:03}_img_{pic_counter:03}.png"
        out_path = images_dir / fname
        crop.save(out_path, format="PNG")

        # Build a relative path that works for both combined and page-split modes.
        # Markdown files live either in OUTPUT dir (page-split) or combined file’s parent,
        # but we always cache images under cache_dir/images and reference relatively from there.
        rel_path = Path("images") / fname

        # If there’s an adjacent Caption block, you could attach it as alt text—simple default for now:
        alt = b.get("alt", f"Image {pic_counter}")

        # Inject markdown so existing block_to_markdown will output it
        new_b = dict(b)
        new_b["text"] = f"![{alt}]({rel_path.as_posix()})"
        out_blocks.append(new_b)

    return out_blocks

# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert a PDF to Markdown using dots.ocr JSON output (resumable).")
    p.add_argument("pdf", help="Path to the input PDF file")
    p.add_argument("-o", "--output", default=None,
                   help=("Output markdown file or directory (default: stdout). "
                         "If --page-split is given and OUTPUT is a directory, "
                         "one file per page is written there."))

    # Model location / acquisition (HF path)
    p.add_argument("--model-dir", default="./weights/DotsOCR",
                   help="Local directory containing (or to contain) the model files.")
    p.add_argument("--model-id", default="rednote-hilab/dots.ocr",
                   help="Hugging Face repo id to download if --model-dir is missing.")

    # vLLM (optional) — when enabled, HF model isn't loaded locally
    p.add_argument("--use-vllm", action="store_true",
                   help="Use a running vLLM server instead of loading the model in-process.")
    p.add_argument("--vllm-ip", default="localhost", help="vLLM server host.")
    p.add_argument("--vllm-port", type=int, default=8000, help="vLLM server port.")
    p.add_argument("--vllm-model-name", default="model", help="Model name as exposed by the vLLM server.")
    p.add_argument("--vllm-temperature", type=float, default=0.1, help="Sampling temperature for vLLM requests.")
    p.add_argument("--vllm-top-p", type=float, default=1.0, help="Top-p for vLLM requests.")
    p.add_argument("--vllm-max-tokens", type=int, default=16384, help="Max completion tokens for vLLM requests.")

    # Dots.ocr knobs
    p.add_argument("--dpi", type=int, default=144)
    p.add_argument("--max-new-tokens", type=int, default=1792)
    p.add_argument("--prompt", default=None, help="Custom user prompt requesting JSON.")
    p.add_argument("--prompt-file", default=None, help="Path to a file with custom JSON extraction instructions.")
    p.add_argument("--prompt-mode", default=None,
                   help="Legacy: repo prompt key (ignored; we use a built-in prompt and log a warning).")

    # Markdown shaping
    p.add_argument("--drop-categories", default="Page-header,Page-footer",
                   help="Comma-separated categories to drop (default: Page-header,Page-footer)")
    p.add_argument("--no-sort", action="store_true",
                   help="Do not sort blocks by reading order (y,x).")
    p.add_argument("--page-split", action="store_true",
                   help="Write one markdown file per page. OUTPUT must be a directory.")
    p.add_argument("--skip-images", action="store_true",
                   help="Skip cropping 'Picture' blocks and including them in Markdown with image links.")

    # Robustness + diagnostics
    p.add_argument("--retries", type=int, default=3,
                   help="Number of retries after the first attempt if JSON parsing fails.")
    p.add_argument("--retry-temps", default="0.0,0.0,0.0",
                   help="Comma-separated temperatures per attempt (first is initial).")
    p.add_argument("--no-json-repair", action="store_true",
                   help="Disable pre-repair of model output before JSON parsing.")
    p.add_argument("--on-parse-fail", choices=["skip", "abort", "raw"], default="abort",
                   help="If all retries fail: skip page, abort the run, or include raw MuPDF text.")
    p.add_argument("--debug", action="store_true", help="Enable debug logging.")

    # Resume controls
    p.add_argument("--cache-dir", default=None,
                   help="Directory for per-page cache (default: ./.dots2md_cache/<pdf_stem>/).")
    p.add_argument("--clear-cache", action="store_true",
                   help="Delete the cache dir before processing.")
    p.add_argument("--reprocess-cached", action="store_true",
                   help="Reprocess pages even if cached files exist.")
    p.add_argument("--start-page", type=int, default=1,
                   help="Start page (1-based, inclusive).")
    p.add_argument("--end-page", type=int, default=None,
                   help="End page (1-based, inclusive).")
    p.add_argument("--ignore-cache-mismatch", action="store_true",
                   help="Ignore cache meta fingerprint mismatch and resume anyway.")

    return p


# =========================
# Main
# =========================

def main():
    args = build_parser().parse_args()
    setup_logging(args.debug)

    retry_temps = tuple(float(x.strip()) for x in args.retry_temps.split(",") if x.strip())

    cfg = DotsConfig(
        model_dir=args.model_dir,
        model_id=args.model_id,
        dpi=args.dpi,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        prompt_mode=args.prompt_mode,
        retry_attempts=max(0, int(args.retries)),
        retry_temperatures=retry_temps,
        json_repair=not args.no_json_repair,

        # vLLM options
        use_vllm=bool(args.use_vllm),
        vllm_ip=args.vllm_ip,
        vllm_port=int(args.vllm_port),
        vllm_model_name=args.vllm_model_name,
        vllm_temperature=float(args.vllm_temperature),
        vllm_top_p=float(args.vllm_top_p),
        vllm_max_tokens=int(args.vllm_max_tokens),
    )

    runner = DotsRunner(cfg)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        LOG.error("PDF not found: %s", pdf_path)
        sys.exit(2)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        LOG.error("Failed to open PDF: %s", e)
        sys.exit(2)

    total_pages = len(doc)
    start_page = max(1, int(args.start_page))
    end_page = total_pages if args.end_page is None else min(total_pages, int(args.end_page))
    if start_page > end_page:
        LOG.error("Invalid page range: start_page (%d) > end_page (%d)", start_page, end_page)
        sys.exit(2)

    # Cache setup
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir_for(pdf_path)
    if args.clear_cache and cache_dir.exists():
        LOG.info("Clearing cache dir: %s", cache_dir)
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache meta + fingerprint (warn on mismatch)
    drop_categories = [c.strip() for c in args.drop_categories.split(",") if c.strip()]
    sort_blocks = not args.no_sort
    meta_current = {
        "pdf_name": pdf_path.name,
        "pdf_size": pdf_path.stat().st_size,
        "pdf_mtime": int(pdf_path.stat().st_mtime),
        "num_pages": total_pages,
        "model_dir": args.model_dir,
        "model_id": args.model_id,
        "use_vllm": bool(args.use_vllm),
        "vllm_ip": args.vllm_ip,
        "vllm_port": int(args.vllm_port),
        "vllm_model_name": args.vllm_model_name,
        "vllm_temperature": float(args.vllm_temperature),
        "vllm_top_p": float(args.vllm_top_p),
        "vllm_max_tokens": int(args.vllm_max_tokens),
        "dpi": args.dpi,
        "max_new_tokens": args.max_new_tokens,
        "prompt_present": bool(args.prompt or args.prompt_file),
        "drop_categories": drop_categories,
        "sort_blocks": sort_blocks,
        "json_repair": not args.no_json_repair,
    }
    meta_existing = read_meta(cache_dir)
    if meta_existing:
        fp_old = meta_existing.get("fingerprint")
        fp_new = compute_fingerprint(meta_current)
        if fp_old and fp_old != fp_new:
            msg = "Cache meta fingerprint differs from current settings."
            if args.ignore_cache_mismatch:
                LOG.warning("%s Proceeding due to --ignore-cache-mismatch.", msg)
            else:
                LOG.warning("%s Use --ignore-cache-mismatch to proceed, or --clear-cache.", msg)

    write_meta(cache_dir, meta_current)

    LOG.info("Converting pages %d–%d of %d from %s", start_page, end_page, total_pages, pdf_path.name)

    for i in range(start_page, end_page + 1):
        json_path, md_path = cache_paths(cache_dir, i)

        # Skip if cached and not forcing reprocess
        if not args.reprocess_cached and md_path.exists() and json_path.exists():
            LOG.info("Page %d/%d: cached — skipping", i, total_pages)
            continue

        try:
            page = doc[i - 1]
            pil = runner.pil_from_page(page, dpi=cfg.dpi)

            page_json = runner.infer_json_with_retries(pil)
            blocks = page_json.get("blocks", [])
            before = len(blocks)
            blocks = filter_blocks(blocks, drop_categories)
            after = len(blocks)
            if before != after:
                LOG.debug("Page %d: filtered %d → %d blocks (drop=%s)", i, before, after, drop_categories)

            if not args.skip_images:
                # NOTE: use the same PIL page we rendered for the model input
                blocks = extract_picture_blocks(pil, blocks, i, cache_dir, combined_output=not args.page_split)

            md = blocks_to_markdown(blocks, sort_blocks=sort_blocks)

            # Persist per-page artifacts immediately (resume point)
            json_path.write_text(json.dumps({"blocks": blocks}, ensure_ascii=False, indent=2), encoding="utf-8")
            md_path.write_text(md, encoding="utf-8")

            LOG.info("Page %d/%d converted and cached (%d blocks)", i, total_pages, after)

            # per-page cleanup
            del pil, page_json, blocks, md
            if runner.device == "mps":
                torch.mps.empty_cache()

        except Exception as e:
            LOG.error("Page %d failed: %s", i, e)
            if args.on_parse_fail == "abort":
                LOG.error("Aborting due to --on-parse-fail=abort. Cached pages remain for resume.")
                sys.exit(1)
            if args.on_parse_fail == "raw":
                try:
                    page = doc[i - 1]
                    fallback = page.get_text().strip()
                except Exception:
                    fallback = ""
                md_path.write_text((fallback + "\n") if fallback else "", encoding="utf-8")
                if not json_path.exists():
                    json_path.write_text(json.dumps({"blocks": []}, ensure_ascii=False), encoding="utf-8")
                LOG.warning("Page %d: included non-OCR raw text fallback and cached", i)
            else:
                # skip: create empty markers so assembly keeps page slots aligned
                if not md_path.exists():
                    md_path.write_text("", encoding="utf-8")
                if not json_path.exists():
                    json_path.write_text(json.dumps({"blocks": []}, ensure_ascii=False), encoding="utf-8")
                LOG.warning("Page %d: skipped due to parse failure; empty page cached", i)

    # Assemble final outputs from cache
    if args.output:
        out_path = Path(args.output)
        if args.page_split:
            if not out_path.exists():
                out_path.mkdir(parents=True, exist_ok=True)
            copy_cached_pages_to_dir(cache_dir, out_path, total_pages)
            LOG.info("Wrote per-page Markdown to %s", out_path)
        else:
            pages_md = assemble_from_cache(cache_dir, total_pages)
            out_path.write_text("\n\n".join(pages_md), encoding="utf-8")
            LOG.info("Wrote combined Markdown to %s", out_path)
    else:
        pages_md = assemble_from_cache(cache_dir, total_pages)
        for i, page_md in enumerate(pages_md, 1):
            print(f"\n--- Page {i} ---\n{page_md}")

    LOG.info("Done.")


if __name__ == "__main__":
    main()