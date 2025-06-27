# OCR-to-Markdown with VLLMs

**pdf2md** + **clean_md** is a lightweight, two-stage pipeline that turns **image-scanned PDF books** into clean, re-flowable Markdown using vision-capable large-language models.

---

## Why two stages?

| Stage | Script | Purpose |
|-------|--------|---------|
| **1** | **`pdf2md`** | Uses a multimodal Vision-LLM (Gemini by default) to *see* each PDF page and emit raw Markdown (one file or one page per file). |
| **2** | **`clean`** | Feeds that Markdown to a chat-LLM to strip running headers, footers, and page numbers, leaving continuous text ready for Pandoc / Calibre. |

Keeping the passes separate lets you re-clean existing `.md` dumps without re-OCRing the whole book, or swap in a different model for either step.

---

## Installation

```bash
git clone https://github.com/pllopis/pdf2md.git
cd pdf2md
python -m venv env && source env/bin/activate
pip install -r requirements.txt
```

Add your Gemini API key to a file named apikey in the repo root
—or set the environment variable GEMINI_API_KEY.

## Quickstart

```bash
# 1️⃣  Vision-OCR pass
./pdf2md -o my_scan.md --concurrency my_scan.pdf 

# 2️⃣  Cleanup pass
./clean -o book.md \
       --clean-model gemini-2.0-flash \
       my_scan.md
```

book.md now contains clean Markdown you can convert to EPUB (or anything else) using Calibre or pandoc, e.g.:

```bash
pandoc book.md -o book.epub
```

Happy scanning!