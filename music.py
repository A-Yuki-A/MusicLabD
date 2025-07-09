import streamlit as st

# ── ページ設定 ──
st.set_page_config(
    page_title="WaveForge",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── おしゃれなダークテーマ & カラーデザイン ┼─
st.markdown("""
<style>
  /* アプリ全体背景 */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #181818 !important;
    color: #E0E0E0 !important;
  }
  /* サイドバー */
  [data-testid="stSidebar"] {
    background-color: #232323 !important;
    color: #E0E0E0 !important;
    border-right: 1px solid #333333;
  }
  /* 見出し */
  h1, h2, h3, h4 {
    font-family: 'Helvetica Neue', sans-serif;
    color: #BB86FC !important;
  }
  /* マークダウン本文 */
  .stMarkdown, .stWrite {
    color: #E0E0E0 !important;
  }
  /* ボタン */
  .stButton>button {
    background-color: #BB86FC !important;
    color: #181818 !important;
    border: none !important;
    border-radius: 4px !important;
  }
  .stButton>button:hover {
    background-color: #CF9CFF !important;
  }
  /* スライダー */
  .stSlider div[data-baseweb] > div {
    background: #BB86FC !important;
  }
  /* 入力コンポーネントのテキスト */
  .css-1tcy0l5, .css-1bxjl7a {
    color: #E0E0E0 !important;
  }
</style>
""", unsafe_allow_html=True)

import numpy as np
import matplotlib.pyplot as plt
import librosa
import tempfile
import soundfile as sf

# pydub読み込み
try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    st.error("pydub モジュールがインストールされていません。requirements.txt に 'pydub' を追加してください。")
    st.stop()

# ffmpeg/ffprobe のパス指定
AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe   = "/usr/bin/ffprobe"

# 音声ロード関数
def load_mp3(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    sr = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        data = data.reshape((-1, 2)).mean(axis=1)
    data /= np.abs(data).max()
    return data, sr

# アプリ本体\st.title("WaveForge")

# ファイルアップロード
uploaded_file = st.file_uploader("MP3ファイルをアップロード", type="mp3")
if not uploaded_file:
    st.info("MP3ファイルをアップロードしてください。")
    st.stop()

# 音声読み込み
data, orig_sr = load_mp3(uploaded_file)
duration = len(data) / orig_sr

# 設定変更セクション
st.markdown("## 設定変更")
st.markdown("**標本化周波数と量子化ビット数を変えて、音の違いを聴き比べしなさい。**")

st.markdown("**標本化周波数 (Hz)：** 1秒間に何回標本を取るかを示します。 対応例: 44100Hz")
target_sr = st.slider("", 4000, 48000, orig_sr if orig_sr >= 4000 else 44100, step=1000)

st.markdown("**量子化ビット数：** 1サンプルを何段階に分けるかを示します。 対応例: 16bit")
bit_depth = st.slider("", 3, 24, 16, step=1)

# 再サンプリングと量子化
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# 波形比較セクション
st.markdown("## 波形比較")
st.markdown(
    "- **振幅 (Amplitude)**: 時間ごとに変わる音の強さ\n"
    "- **量子化ビット数**: 振幅を何段階に分けるか\n"
    "- **関係**: ビット数が多いほど振幅の変化を細かく表現できる",
    unsafe_allow_html=True
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
# 元の波形
t_orig = np.linspace(0, duration, num=len(data))
ax1.plot(t_orig, data)
ax1.set(title="Original Waveform", xlabel="Time (s)", ylabel="Amplitude")
ax1.set(xlim=(0, duration), ylim=(-1, 1))
# 処理後の波形
proc_len = min(len(quantized), int(duration * target_sr))
t_proc = np.linspace(0, duration, num=proc_len)
ax2.plot(t_proc, quantized[:proc_len])
ax2.set(title=f"Processed Waveform ({target_sr:,} Hz, {bit_depth:,}bit)", xlabel="Time (s)", ylabel="Amplitude")
ax2.set(xlim=(0, duration), ylim=(-1, 1))
st.pyplot(fig)

# 標本点表示 & ズーム
st.markdown("## 標本化周波数の違いを強調")
st.markdown(
    f"上: 全体の波形を●で示し、下: 最初の0.001秒をズームして標本化周波数 {target_sr:,}Hz の点の間隔を確認"
)

fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
# 全体
ax3.plot(t_full, quantized, linewidth=1)
ax3.scatter(t_full, quantized, s=5)
ax3.set(title="Processed with Sample Points", xlabel="Time (s)", ylabel="Amplitude")
# ズーム
zoom_end = 0.001
zoom_count = int(target_sr * zoom_end)
t_zoom = t_full[:zoom_count]
ax4.plot(t_zoom, quantized[:zoom_count], linewidth=1)
ax4.scatter(t_zoom, quantized[:zoom_count], s=20)
ax4.set(title=f"Zoom ({target_sr:,}Hz, 0–{zoom_end}s)", xlabel="Time (s)", ylabel="Amplitude")
ax4.set(xlim=(0, zoom_end), ylim=(-1, 1))
st.pyplot(fig2)

# 再生セクション
st.markdown("## 再生")
subtype_map = {8: 'PCM_U8', 16: 'PCM_16', 24: 'PCM_24'}
selected_subtype = subtype_map.get(bit_depth, 'PCM_16')
if np.all(quantized == 0):
    st.warning(f"{target_sr:,}Hz にリサンプリング結果、無音になりました。周波数を上げてください。")
else:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
        sf.write(out.name, quantized, target_sr, subtype=selected_subtype)
        st.audio(out.name, format="audio/wav")

# データ量計算
st.markdown("## データ量計算")
bytes_size = target_sr * bit_depth * 1 * duration / 8
kb_size = bytes_size / 1024
mb_size = kb_size / 1024
st.markdown(f"**{target_sr:,}Hz × {bit_depth:,}bit × 1ch × {duration:.2f}s = {mb_size:.2f}MB**")
