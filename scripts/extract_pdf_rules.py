#!/usr/bin/env python3
"""
advanced_manual.pdf のテキストを rules/advanced_rule.md に書き出す。
使い方: conda activate deckAi && python scripts/extract_pdf_rules.py
"""
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "rules" / "advanced_manual.pdf"
OUT_PATH = ROOT / "rules" / "advanced_rule.md"


def main() -> None:
    if not PDF_PATH.is_file():
        raise SystemExit(f"PDF が見つかりません: {PDF_PATH}")
    reader = PdfReader(PDF_PATH)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        chunks.append(text if text else "")
    out = "\n\n".join(chunks)
    OUT_PATH.write_text(out, encoding="utf-8")
    print(f"抽出完了: {len(reader.pages)} ページ, {len(out)} 文字 → {OUT_PATH}")


if __name__ == "__main__":
    main()
