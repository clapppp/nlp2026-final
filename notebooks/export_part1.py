"""
PART1 노트북을 (실행하지 않고) 현재 상태 그대로 HTML로 변환하는 스크립트.

Windows + Python 3.8 환경의 ssl 인증서 스토어 버그
([ASN1: NOT_ENOUGH_DATA] not enough data)를 우회하기 위해
ssl.SSLContext.load_default_certs를 패치한 뒤에 nbconvert를 import한다.

사용법 (PowerShell, nlp_final 환경에서):
    conda activate nlp_final
    cd $HOME\nlp2026-final
    pip install nbconvert
    python notebooks\export_part1.py

결과: notebooks\PART1_sentiment_analysis.html 생성
      → Edge로 열고 Ctrl+P → "PDF로 저장"
"""
import ssl

# ── Windows 인증서 스토어 버그 우회 ──────────────────────────────
#  nbconvert가 tornado를 import할 때 create_default_certs()에서 터지는 것을 막는다.
_orig_load_default_certs = ssl.SSLContext.load_default_certs


def _safe_load_default_certs(self, *args, **kwargs):
    try:
        return _orig_load_default_certs(self, *args, **kwargs)
    except ssl.SSLError:
        return None


ssl.SSLContext.load_default_certs = _safe_load_default_certs

# ── 이제 안전하게 import ─────────────────────────────────────────
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "PART1_sentiment_analysis.ipynb"
HTML_PATH = ROOT / "notebooks" / "PART1_sentiment_analysis.html"

print(f"reading {NB_PATH}")
nb = nbformat.read(str(NB_PATH), as_version=4)

print("converting to HTML (실행하지 않고 현재 출력 그대로)")
body, _ = HTMLExporter().from_notebook_node(nb)
HTML_PATH.write_text(body, encoding="utf-8")

print(f"\nDONE -> {HTML_PATH}")
print("Edge/Chrome 으로 열고 Ctrl+P -> 'PDF로 저장' 하세요.")
