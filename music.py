import streamlit as st

# ── ページ設定 ──
st.set_page_config(
    page_title="WaveForge",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── ライトテーマ カスタムCSS ──
st.markdown("""
<style>
  /* 全体背景と文字色 */
  html, body, [data-testid=\"stAppViewContainer\"] {
    background-color: #FFFFFF !important;
    color: #333333 !important;
  }
  /* サイドバー */
  [data-testid=\"stSidebar\"] {
    background-color: #F0F0F0 !important;
    color: #333333 !important;
  }
  /* 見出し */
  h1, h2, h3, h4 {
    color: #1E90FF !important;
  }
  /* マークダウンテキスト */
  .stMarkdown, .stWrite {
    color: #333333 !important;
  }
  /* ボタン */
  .stButton>button {
    background-color: #1E90FF !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
  }
  .stButton>button:hover {
    background-color: #63B8FF !important;
  }
  /* スライダーのトラックを白に */
  .stSlider div[data-baseweb] > div {
    background: #FFFFFF !important;
  }
</style>
""", unsafe_allow_html=True)

import numpy as np
import matplotlib.pyplot as plt
import librosa
import tempfile
import soundfile as sf

# pydub 読み込み
try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    st.error("pydub モジュールがインストールされていません。requirements.txt に 'pydub' を追加してください。")
    st.stop()

# ffmpeg/ffprobe のパス指定
AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe    = "/usr/bin/ffprobe"

# 音声ロード関数
def load_mp3(uploaded_file):
    """
    MP3ファイルを一時保存し、サンプリングレートと正規化済みデータを取得
    ・戻り値: data (np.ndarray), sr (int)
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    sr = audio.frame_rate  # サンプリング周波数 (Hz)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    # ステレオなら左右平均でモノラル化
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    # 正規化: 振幅を -1.0～1.0 の範囲に
    data = samples / np.abs(samples).max()
    return data, sr

# アプリ本体
st.title("WaveForge")

# ファイルアップロード
downloaded_file = st.file_uploader("MP3ファイルをアップロード", type="mp3")
if not downloaded_file:
    st.info("MP3ファイルをアップロードしてください。")
    st.stop()

# 音声読み込み
data, orig_sr = load_mp3(downloaded_file)
duration = len(data) / orig_sr  # 再生時間 (秒)

# 設定変更セクション
st.markdown("## ◼️ 設定変更")
st.markdown("音質に影響を与える2つの要素を調整できます。")

# 標本化周波数設定
st.markdown(
    "**標本化周波数 (Sample Rate)**: 1秒間に何回音を記録するか。数値が大きいほど高い音域まで再現可能。例: CDは44,100 Hz"
)
target_sr = st.slider(
    "標本化周波数 (Hz)", min_value=4000, max_value=48000,
    value=orig_sr if orig_sr >= 4000 else 44100,
    step=1000
)

# 量子化ビット数設定（最小3bit）
st.markdown(
    "**量子化ビット数 (Bit Depth)**: 振幅 (Amplitude) を何段階で記録するか。ビット数が大きいほど音の強弱を滑らかに。例: CDは16 bit"
)
bit_depth = st.slider(
    "量子化ビット数 (bit)",
    min_value=3, max_value=24,
    value=16, step=1
)

# 再サンプリングと量子化
rs_data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
max_int = 2 ** (bit_depth - 1) - 1
quantized = np.round(rs_data * max_int) / max_int

# 波形比較セクション
st.markdown("## ◼️ 波形比較")
st.markdown(
    "- 振幅 (Amplitude): 時間ごとの音の大きさ\n"
    "- 元波形: アップロードした元の音\n"
    "- 処理後: 標本化周波数とビット数を反映した波形",
    unsafe_allow_html=True
)

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)
# 元の波形
t_orig = np.linspace(0, duration, len(data))
ax1.plot(t_orig, data)
ax1.set(title="Original Waveform", xlabel="Time [s]", ylabel="Amplitude")
ax1.set(xlim=(0, duration), ylim=(-1, 1))
# 処理後の波形
t_proc = np.linspace(0, duration, len(quantized))
ax2.plot(t_proc, quantized)
ax2.set(title=f"Processed ({target_sr} Hz, {bit_depth} bit)", xlabel="Time [s]", ylabel="Amplitude")
ax2.set(xlim=(0, duration), ylim=(-1, 1))
st.pyplot(fig1)

# 標本点表示 & ズームセクション
st.markdown("## ◼️ 標本化周波数の違いを強調")
st.markdown("上: 全体の標本点, 下: 最初の1 ms をズーム表示")

fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)
# 全体標本点表示
t_full = np.linspace(0, duration, len(quantized))
ax3.scatter(t_full, quantized, s=6)
ax3.set(title="標本点表示", xlabel="Time [s]", ylabel="Amplitude")
# ズーム (0–0.001s)
zoom_dur = 0.001
idx = int(target_sr * zoom_dur)
t_zoom = t_full[:idx]
ax4.scatter(t_zoom, quantized[:idx], s=20)
ax4.set(title=f"ズーム (0–{zoom_dur*1000:.1f} ms)", xlabel="Time [s]", ylabel="Amplitude")
ax4.set(xlim=(0, zoom_dur), ylim=(-1, 1))
st.pyplot(fig2)

# 再生セクション
st.markdown("## ◼️ 再生")
subtypes = {8: 'PCM_U8', 16: 'PCM_16', 24: 'PCM_24'}
stype = subtypes.get(bit_depth, 'PCM_16')
if np.all(quantized == 0):
    st.warning(f"{target_sr} Hz では無音です。標本化周波数を上げてください。")
else:
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, quantized, target_sr, subtype=stype)
        st.audio(tmp.name)

# データ量計算セクション
st.markdown("## ◼️ データ量計算")
# データ量を求める式の表示
st.markdown(
    "音のデータ量 = 標本化周波数 (Hz) × 量子化ビット数 (bit) × 時間 (s) × チャンネル数 (ch)"
)
st.markdown("設定を変更したファイルのデータ量＝")

# 計算過程表示
count = target_sr * duration  # 標本化点の総数
bytes_ = count * bit_depth * 2 / 8  # ステレオ2chの場合
kb = bytes_ / 1024
mb = kb / 1024
# 見やすいようにコードブロックで改行表示
st.markdown(f"""
```text
{target_sr:,} Hz × {bit_depth} bit × 2 ch × {duration:.2f} 秒 ÷ 8 = {int(bytes_):,} バイト
{int(bytes_):,} バイト ÷ 1024 = {kb:,.2f} KB
{kb:,.2f} KB ÷ 1024 = {mb:,.2f} MB
```""", unsafe_allow_html=True)

# チャンネルの違い
st.markdown("""
- **ステレオ (2ch)**: 左右2つの音声信号で再生。音の広がりがあります。
- **モノラル (1ch)**: 1つの音声信号で再生。音の定位は中央になります。
""", unsafe_allow_html=True)

# 音声ファイル形式の比較テーブル
st.markdown("""
| 拡張子 | 圧縮方式     | 特徴                                      |
|--------|-------------|-------------------------------------------|
| MP3    | 非可逆圧縮  | ファイルサイズが小さく、多くの機器で再生可能  |
| FLAC   | 可逆圧縮    | オリジナルと同等の音質を保持、容量は中程度  |
| WAV    | 非圧縮      | 最大音質だがファイルサイズが非常に大きい    |
""", unsafe_allow_html=True)
