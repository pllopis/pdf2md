# pdf2md

pdf2md converts PDF files into clean, structured Markdown using a Vision LLM.
It leverages dots.ocr to perform state-of-the-art OCR and layout analysis, preserving headings, paragraphs, tables, and images with high fidelity.

Features:

  *	Checkpoint/restart — resume from partially processed PDFs without losing progress.
  *	Page range selection — convert only the pages you need.
  *	Preserves document structure and (optionally) filters out headers/footers.
  *	Experimental MPS (Apple Silicon) and CPU backends for non-CUDA systems.

## Quickstart

```
# 1. Install PyTorch (CUDA 12.8 build)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 2. Install pdf2md in editable mode
pip install -e .
```

Example

```
pdf2md mydoc.pdf -o mydoc.md
```

## Notes

	*	Requires a local or auto-downloaded dots.ocr model (see dots.ocr repo for details).
	*	On first run, the model will be downloaded to ./weights/DotsOCR unless overridden with --model-dir.

