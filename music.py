# ── モノトーン調カスタムCSS（改訂版） ──
st.markdown("""
<style>
  /* ページ全体の背景と文字色 */
  html, body, [data-testid="stAppViewContainer"], .css-k1vhr4, .css-1outpf7 {
    background-color: #FFFFFF !important;
    color: #000000 !important;
  }
  /* 見出しやマークダウン本文 */
  h1, h2, h3, h4, .markdown-text-container, .stMarkdown, .stWrite {
    color: #000000 !important;
  }
  /* サイドバー */
  [data-testid="stSidebar"], .css-1d391kg {
    background-color: #F8F8F8 !important;
    color: #000000 !important;
  }
  /* ボタン */
  .stButton>button, button[kind] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 4px !important;
  }
  .stButton>button:hover, button[kind]:hover {
    background-color: #EEEEEE !important;
  }
  /* スライダーのバー */
  .stSlider div[data-baseweb] > div {
    background: #C0C0C0 !important;
  }
</style>
""", unsafe_allow_html=True)
