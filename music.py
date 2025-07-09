import streamlit as st
try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    st.error("pydub モジュールがインストールされていません。requirements.txt に 'pydub' を追加してください。")
    st.stop()
import numpy as np
import matplotlib.pyplot as plt
import librosa
import tempfile
import soundfile as sf

# ── ffmpeg/ffprobe のパス指定 ──
AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe   = "/usr/bin/ffprobe"

# ── ページ設定 ──
st.set_page_config(
    page_title="WaveForgeTEST",
    layout="centered"
)

# ── 音声ロード関数 ──
def load_mp3(uploaded_file):
    """
    MP3を一時ファイル経由で読み込み、正規化した NumPy 配列とサンプリングレートを返す
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    sr = audio.frame_rate
    data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        # ステレオをモノラルに平均化
        data = data.reshape((-1, 2)).mean(axis=1)
    data /= np.abs(data).max()
    return data, sr

# ── アプリ本体 ──
st.title("WaveForge")

# ファイルアップロード
uploaded_file = st.file_uploader("MP3ファイルをアップロード", type="mp3")
if not uploaded_file:
    st.info("MP3ファイルをアップロードしてください。")
    st.stop()

# 音声読み込み
data, orig_sr = load_mp3(uploaded_file)
duration = len(data) / orig_sr

# ── 設定変更 ──
st.write("### 設定変更")
st.markdown("**標本化周波数と量子化ビット数を変えて、音の違いを聴き比べしなさい。**", unsafe_allow_html=True)

st.markdown(
    "<span style='font-weight:bold; color:orange;'>標本化周波数 (Hz)：</span>1秒間に何回標本を取るかを示します。",
    unsafe_allow_html=True
)
target_sr = st.slider("", 4000, 48000, orig_sr if orig_sr >= 4000 else 44100, step=1000)

st.markdown(
    "<span style='font-weight:bold; color:orange;'>量子化ビット数：</span>1サンプルを何段階に分けるかを示します。",
    unsafe_allow_html=True
)
bit_depth = st.slider("", 3, 24, 16, step=1)

# ── 再サンプリングと量子化 ──
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2**(bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# ── 波形比較（従来） ──
st.write("### 波形比較")
st.markdown("""
- **振幅（Amplitude）**……時間ごとに連続的に変わる音の強さ  
- **量子化ビット数**……「振幅」を何段階に分けるか  
- **関係**……ビット数が多いほど振幅の変化を細かく表現でき、ノイズ（量子化誤差）を減らせる
""")

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
ax2.set(title=f"Processed Waveform ({target_sr:,} Hz, {bit_depth:,} bit)",
         xlabel="Time (s)", ylabel="Amplitude")
ax2.set(xlim=(0, duration), ylim=(-1, 1))

st.pyplot(fig)

# ── 波形比較（標本点表示 & ズーム） ──
st.write("### 標本化周波数の違いを強調（標本点表示 & ズーム）")
st.markdown(f"""
このグラフでは、標本点（サンプリングした点）を●で示し、標本化周波数の違いによって標本点の間隔がどう変わるかを比較しています。  
- **上のグラフ**：全体の波形を標本点付きで表示  
- **下のグラフ**：最初の0.001秒をズーム表示し、標本化周波数 **{target_sr:,} Hz** と標本点の粗さを強調  
標本化周波数が低いほど、標本点の間隔が広がり、波形が粗くなることがわかります。
""")

fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

# 1. 全体波形＋標本点
t_full = np.linspace(0, duration, num=len(quantized))
ax3.plot(t_full, quantized, linewidth=1)
ax3.scatter(t_full, quantized, s=5)
ax3.set(title="Processed Waveform with Sample Points",
        xlabel="Time (s)", ylabel="Amplitude")
ax3.set(xlim=(0, duration), ylim=(-1, 1))

# 2. ズーム表示（最初の0.001秒を拡大）
zoom_end = 0.001  # 1ミリ秒
zoom_count = int(target_sr * zoom_end)
t_zoom = t_full[:zoom_count]
ax4.plot(t_zoom, quantized[:zoom_count], linewidth=1)
ax4.scatter(t_zoom, quantized[:zoom_count], s=20)
ax4.set(title=f"Zoomed Waveform ({target_sr:,} Hz, 0–{zoom_end}s)",
        xlabel="Time (s)", ylabel="Amplitude")
ax4.set(xlim=(0, zoom_end), ylim=(-1, 1))

st.pyplot(fig2)

# ── 以下、再生・データ量計算など ──
