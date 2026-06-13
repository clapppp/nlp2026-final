"""
PART1 노트북을 실행하고 HTML로 변환하는 스크립트.

Windows + Python 3.8 환경의 ssl 인증서 스토어 버그
([ASN1: NOT_ENOUGH_DATA] not enough data)를 우회하기 위해
ssl.SSLContext.load_default_certs를 패치한 뒤에 nbconvert를 import한다.

사용법 (PowerShell, nlp_final 환경에서):
    conda activate nlp_final
    cd $HOME\nlp2026-final
    pip install nbconvert pandas matplotlib
    python notebooks\export_part1.py

결과: notebooks\PART1_sentiment_analysis.html 생성
      → Edge로 열고 Ctrl+P → "PDF로 저장"
"""
import os
import ssl

# ── 1. Windows 인증서 스토어 버그 우회 ────────────────────────────
#  create_default_context() 가 내부에서 load_default_certs()를 호출하며
#  깨진 인증서를 만나 SSLError를 던지는데, 이를 조용히 무시하게 한다.
#  (HuggingFace 다운로드는 requests+certifi를 쓰므로 영향받지 않는다.)
_orig_load_default_certs = ssl.SSLContext.load_default_certs


def _safe_load_default_certs(self, *args, **kwargs):
    try:
        return _orig_load_default_certs(self, *args, **kwargs)
    except ssl.SSLError:
        return None


ssl.SSLContext.load_default_certs = _safe_load_default_certs

# ── 2. HF 다운로드 진행바 끄기 (IProgress 에러 방지) ──────────────
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── 3. 이제 안전하게 import ──────────────────────────────────────
from pathlib import Path

import nbformat
from ipykernel.kernelspec import install as install_kernel
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "PART1_sentiment_analysis.ipynb"
HTML_PATH = ROOT / "notebooks" / "PART1_sentiment_analysis.html"

# ── 4. 현재 파이썬을 가리키는 커널 등록 (커널 이름 불일치 방지) ────
KERNEL_NAME = "nlp_final_export"
install_kernel(user=True, kernel_name=KERNEL_NAME, display_name="nlp_final (export)")
print(f"[1/4] kernel '{KERNEL_NAME}' registered")

# ── 5. 노트북 읽기 ───────────────────────────────────────────────
print(f"[2/4] reading {NB_PATH}")
nb = nbformat.read(str(NB_PATH), as_version=4)

# ── 6. 실행 (학습 4개 포함, GPU에서 약 30~60분) ──────────────────
print("[3/4] executing notebook... (trains 4 models, may take 30-60 min)")
ep = ExecutePreprocessor(timeout=-1, kernel_name=KERNEL_NAME, allow_errors=True)
ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})

# 실행 결과를 노트북에도 저장 (출력 포함)
nbformat.write(nb, str(NB_PATH))

# ── 7. HTML로 변환 ───────────────────────────────────────────────
print("[4/4] exporting to HTML")
exporter = HTMLExporter()
body, _ = exporter.from_notebook_node(nb)
HTML_PATH.write_text(body, encoding="utf-8")

print(f"\nDONE -> {HTML_PATH}")
print("Edge/Chrome 으로 열고 Ctrl+P -> 'PDF로 저장' 하세요.")
