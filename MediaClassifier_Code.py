import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import seaborn as sns
import cv2
import librosa
import librosa.display
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter

# Paths (built from the location of this .py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# NLTK resources (downloaded only the first time)
for resource, path in [("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

sns.set_style("whitegrid")
pd.set_option("display.width", 200)       # show all the columns in the console
pd.set_option("display.max_columns", None)


def save_figure(number):
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"RESULTADO{number}.png"), dpi=120)
    plt.show()


# ---------------------------------------------------------------------------
# 1) IMAGES: color and pattern
# ---------------------------------------------------------------------------
# Hue ranges of OpenCV (H goes from 0 to 179)
HUE_RANGES = {"Red": [(0, 9), (170, 179)], "Orange": [(10, 21)], "Yellow": [(22, 34)],
              "Green": [(35, 84)], "Cyan": [(85, 99)], "Blue": [(100, 129)], "Purple": [(130, 169)]}


def analyze_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (512, int(512 * img.shape[0] / img.shape[1])))  # same width for all
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color: hue histogram of the pixels with a clear color
    colorful_mask = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 50)).astype(np.uint8)
    hue_hist = cv2.calcHist([hsv], [0], colorful_mask, [180], [0, 180]).flatten()
    color_pixels = {name: sum(hue_hist[a:b + 1].sum() for a, b in ranges) for name, ranges in HUE_RANGES.items()}

    # Pattern: percentage of edge pixels found with Canny
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 100, 200)

    return {"File": os.path.basename(path),
            "Brightness": gray.mean(),
            "Saturation": hsv[:, :, 1].mean(),
            "Colorful_%": colorful_mask.mean() * 100,
            "Dominant_hue": max(color_pixels, key=color_pixels.get),
            "Edge_density_%": (edges > 0).mean() * 100}


def classify_image(row):
    color = "Grayscale" if row["Colorful_%"] < 15 else row["Dominant_hue"]
    if row["Edge_density_%"] < 3:
        pattern = "Smooth"
    elif row["Edge_density_%"] < 10:
        pattern = "Medium texture"
    else:
        pattern = "Highly textured"
    return pd.Series({"Color_class": color, "Pattern_class": pattern})


# ---------------------------------------------------------------------------
# 2) AUDIO: noise level and dominant pitch
# ---------------------------------------------------------------------------
def butter_highpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype="high", analog=False)
    return lfilter(b, a, data)


def analyze_audio(path):
    y, sr = librosa.load(path, sr=None)
    y = butter_highpass_filter(y, 50, sr)

    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=y)[0], ref=1.0).astype(float)
    noise_floor = np.percentile(rms_db, 10)
    peak = np.percentile(rms_db, 95)

    N = len(y)
    yf = np.abs(fft(y))[:N // 2]
    xf = fftfreq(N, 1 / sr)[:N // 2]
    valid = (xf >= 50) & (xf <= 10000)
    dominant = xf[valid][np.argmax(yf[valid])]
    band = valid & (np.abs(xf - dominant) <= 20)
    pitch_energy = (yf[band] ** 2).sum() / (yf[valid] ** 2).sum() * 100

    return {"File": os.path.basename(path),
            "Duration_s": librosa.get_duration(y=y, sr=sr),
            "Noise_floor_dB": noise_floor,
            "SNR_dB": peak - noise_floor,
            "Dominant_freq_Hz": dominant,
            "Note": librosa.hz_to_note(dominant),
            "Pitch_energy_%": pitch_energy}


def classify_audio(row):
    if row["SNR_dB"] < 8:
        noise = "High noise"
    elif row["SNR_dB"] < 20:
        noise = "Medium noise"
    else:
        noise = "Low noise"
    if row["Pitch_energy_%"] < 5:
        pitch = "No clear pitch"
    elif row["Dominant_freq_Hz"] < 500:
        pitch = "Low pitch"
    elif row["Dominant_freq_Hz"] < 2000:
        pitch = "Medium pitch"
    else:
        pitch = "High pitch"
    return pd.Series({"Noise_class": noise, "Pitch_class": pitch})


# ---------------------------------------------------------------------------
# 3) TEXT: word frequencies and keywords
# ---------------------------------------------------------------------------
stop_words = set(stopwords.words("english"))
TOPIC_KEYWORDS = {
    "Politics/News": ["president", "election", "state", "government", "law", "court", "vote", "party",
                      "police", "city", "county", "committee", "bill", "senate", "jury", "mayor"],
    "Religion": ["god", "church", "christ", "faith", "spirit", "christian", "prayer", "religious",
                 "bible", "lord", "soul", "religion", "theology", "jesus", "holy", "sin"],
    "Romance/Fiction": ["love", "heart", "eyes", "smiled", "night", "looked", "kiss", "felt",
                        "room", "girl", "door", "thought", "face", "hand", "knew", "back"],
}


def analyze_text(path):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    words = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    content_words = [w for w in words if w not in stop_words and len(w) > 2]
    freq = FreqDist(content_words)

    scores = {topic: sum(freq[w] for w in kws) / len(content_words) * 100 for topic, kws in TOPIC_KEYWORDS.items()}
    return {"File": os.path.basename(path),
            "Words": len(words),
            "Unique_words": len(set(words)),
            "Lexical_diversity": len(set(words)) / len(words),
            "Avg_sentence_length": len(words) / len(sent_tokenize(text)),
            "Avg_word_length": np.mean([len(w) for w in words]),
            "Top_words": ", ".join(w for w, c in freq.most_common(5)),
            **{f"Score_{t}": s for t, s in scores.items()},
            "Topic": max(scores, key=scores.get),
            "Freq": freq}


# ---------------------------------------------------------------------------
# 4) AUTOMATED WORKFLOW: every file goes to its analysis by its extension
# ---------------------------------------------------------------------------
images, audios, texts = [], [], []
for root, dirs, files in os.walk(DATA_DIR):
    for name in sorted(files):
        path = os.path.join(root, name)
        ext = os.path.splitext(name)[1].lower()
        if ext in (".jpg", ".jpeg", ".png"):
            images.append(analyze_image(path))
        elif ext in (".wav", ".mp3", ".ogg"):
            audios.append(analyze_audio(path))
        elif ext == ".txt":
            texts.append(analyze_text(path))

images_df = pd.DataFrame(images)
audio_df = pd.DataFrame(audios)
text_df = pd.DataFrame(texts)
images_df = pd.concat([images_df, images_df.apply(classify_image, axis=1)], axis=1)
audio_df = pd.concat([audio_df, audio_df.apply(classify_audio, axis=1)], axis=1)
audio_df["Sound"] = audio_df["File"].str.replace(".wav", "")
print(f"Files processed -> Images: {len(images_df)}, Audio: {len(audio_df)}, Text: {len(text_df)}")

# ---------------------------------------------------------------------------
# 5) STATISTICAL SUMMARIES AND COMPARISONS
# ---------------------------------------------------------------------------
print("\n------------------ IMAGES ------------------")
print(images_df[["Brightness", "Saturation", "Colorful_%", "Edge_density_%"]].describe().round(2))
print(images_df[["File", "Color_class", "Pattern_class", "Edge_density_%"]].round(2).to_string(index=False))
print("\nMean features per pattern class:")
print(images_df.groupby("Pattern_class")[["Edge_density_%", "Saturation"]].mean().round(2))

print("\n------------------ AUDIO ------------------")
print(audio_df[["Noise_floor_dB", "SNR_dB", "Dominant_freq_Hz", "Pitch_energy_%"]].describe().round(2))
print(audio_df[["File", "SNR_dB", "Dominant_freq_Hz", "Note", "Noise_class", "Pitch_class"]].round(1).to_string(index=False))
print("\nMean features per noise class:")
print(audio_df.groupby("Noise_class")[["SNR_dB", "Noise_floor_dB", "Dominant_freq_Hz"]].mean().round(1))

print("\n------------------ TEXT ------------------")
print(text_df[["Words", "Unique_words", "Lexical_diversity", "Avg_sentence_length", "Avg_word_length"]].describe().round(3))
score_cols = [c for c in text_df.columns if c.startswith("Score_")]
print(text_df[["File", "Top_words"] + score_cols + ["Topic"]].round(2).to_string(index=False))
print("\nMean statistics per topic:")
print(text_df.groupby("Topic")[["Words", "Lexical_diversity", "Avg_sentence_length", "Avg_word_length"]].mean().round(3))
all_words = FreqDist()
for freq in text_df["Freq"]:
    all_words.update(freq)
print("\nMost common words in all the texts:", all_words.most_common(10))


summary = pd.concat([
    pd.DataFrame({"File": images_df["File"], "Media": "Image",
                  "Class": images_df["Color_class"] + " - " + images_df["Pattern_class"]}),
    pd.DataFrame({"File": audio_df["File"], "Media": "Audio",
                  "Class": audio_df["Noise_class"] + " - " + audio_df["Pitch_class"]}),
    pd.DataFrame({"File": text_df["File"], "Media": "Text", "Class": text_df["Topic"]}),
], ignore_index=True)
summary.to_csv(os.path.join(RESULTS_DIR, "classification_results.csv"), index=False)
print("\nClassification saved in Results/classification_results.csv")

# ---------------------------------------------------------------------------
# 6) VISUALIZATIONS
# ---------------------------------------------------------------------------
# 1. Images with their class
fig, axes = plt.subplots(2, 5, figsize=(15, 7.5), gridspec_kw={"hspace": 0.3})
for ax, (_, row) in zip(axes.flatten(), images_df.iterrows()):
    img = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, "Images", row["File"])), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(f"{row['File']}\n{row['Color_class']} | {row['Pattern_class']}", fontsize=10)
    ax.axis("off")
plt.suptitle("Image classification (color | pattern)", fontsize=15)
save_figure(1)

# 2. Step by step: how the color and the edges of an image are obtained
COLOR_PLOT = {"Red": "#D62728", "Orange": "#FF7F0E", "Yellow": "#E8C547", "Green": "#2CA02C",
              "Cyan": "#17BECF", "Blue": "#1F77B4", "Purple": "#9467BD"}
fig, axes = plt.subplots(2, 5, figsize=(26, 11))

# Row 1: color identification (starry_night.jpg)
img = cv2.imread(os.path.join(DATA_DIR, "Images", "starry_night.jpg"))
img = cv2.resize(img, (512, int(512 * img.shape[0] / img.shape[1])))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 50)).astype(np.uint8)
hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180]).flatten()
pixels = {name: sum(hist[a:b + 1].sum() for a, b in ranges) for name, ranges in HUE_RANGES.items()}
percent = {name: value / sum(pixels.values()) * 100 for name, value in pixels.items()}
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hue_image = hsv_to_rgb(np.dstack([hsv[:, :, 0] / 179, np.ones(mask.shape), np.ones(mask.shape)]))
colorful = rgb.copy()
colorful[mask == 0] = 0
axes[0, 0].imshow(rgb)
axes[0, 0].set_title("1. Original image")
axes[0, 1].imshow(hue_image)
axes[0, 1].set_title("2. Hue channel (H of HSV)")
axes[0, 2].imshow(colorful)
axes[0, 2].set_title(f"3. Colorful pixels (S > 60 and V > 50): {mask.mean() * 100:.0f}%")
for ax in axes[0, :3]:
    ax.axis("off")
axes[0, 3].bar(range(180), hist, width=1, color=[hsv_to_rgb([h / 179, 1, 1]) for h in range(180)])
axes[0, 3].set_title("4. Hue histogram (cv2.calcHist)")
axes[0, 3].set_xlabel("Hue (0-179)")
axes[0, 3].set_ylabel("Number of pixels")
bars = axes[0, 4].bar(percent.keys(), percent.values(), color=[COLOR_PLOT[c] for c in percent], edgecolor="black")
axes[0, 4].bar_label(bars, fmt="%.0f%%")
axes[0, 4].set_title(f"5. Pixels per color -> dominant color: {max(percent, key=percent.get)}")
axes[0, 4].set_ylabel("% of the colorful pixels")
axes[0, 4].tick_params(axis="x", rotation=45)

# Row 2: edge detection (coins.png)
img = cv2.imread(os.path.join(DATA_DIR, "Images", "coins.png"))
img = cv2.resize(img, (512, int(512 * img.shape[0] / img.shape[1])))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 100, 200)
overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
overlay[edges > 0] = [255, 0, 0]
steps = [(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "1. Original image"),
         (gray, "2. Grayscale (cv2.cvtColor)"),
         (blurred, "3. Gaussian blur 5x5 (removes small noise)"),
         (edges, f"4. Canny edges (100, 200): {(edges > 0).mean() * 100:.1f}% of pixels"),
         (overlay, "5. Edges over the original image")]
for ax, (picture, title) in zip(axes[1], steps):
    ax.imshow(picture, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.suptitle("Step by step: color identification (top) and edge detection (bottom)", fontsize=17)
save_figure(2)

# 3. Color identification: color histogram (B, G, R) of each image
fig, axes = plt.subplots(4, 5, figsize=(20, 14))
for i, (_, row) in enumerate(images_df.iterrows()):
    img = cv2.imread(os.path.join(DATA_DIR, "Images", row["File"]))
    img = cv2.resize(img, (512, int(512 * img.shape[0] / img.shape[1])))  # same size as in the analysis
    ax_img = axes[(i // 5) * 2, i % 5]
    ax_hist = axes[(i // 5) * 2 + 1, i % 5]
    ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_img.set_title(row["File"])
    ax_img.axis("off")
    for channel, color in enumerate(["b", "g", "r"]):
        hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
        ax_hist.plot(hist, color=color)
    ax_hist.set_title(f"Color: {row['Color_class']} ({row['Colorful_%']:.0f}% colorful pixels)", fontsize=10)
    ax_hist.set_xlabel("Pixel value")
    ax_hist.set_ylabel("Number of pixels")
plt.suptitle("Color identification: color histogram (B, G, R) of each image", fontsize=15)
save_figure(3)

# 4. Pattern identification: Canny edges of each image
fig, axes = plt.subplots(4, 5, figsize=(20, 14))
for i, (_, row) in enumerate(images_df.iterrows()):
    gray = cv2.imread(os.path.join(DATA_DIR, "Images", row["File"]), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (512, int(512 * gray.shape[0] / gray.shape[1])))  # same size as in the analysis
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 100, 200)
    ax_img = axes[(i // 5) * 2, i % 5]
    ax_edges = axes[(i // 5) * 2 + 1, i % 5]
    ax_img.imshow(gray, cmap="gray")
    ax_img.set_title(row["File"])
    ax_img.axis("off")
    ax_edges.imshow(edges, cmap="gray")
    ax_edges.set_title(f"Canny edges: {row['Edge_density_%']:.1f}% ({row['Pattern_class']})", fontsize=10)
    ax_edges.axis("off")
plt.suptitle("Pattern identification: Canny edges of each image", fontsize=15)
save_figure(4)

# 5. Pattern: edge density of each image
plt.figure(figsize=(11, 5))
data = images_df.sort_values("Edge_density_%", ascending=False)
sns.barplot(data=data, x="File", y="Edge_density_%", hue="Pattern_class", dodge=False, palette="Blues_r")
plt.axhline(3, color="red", linestyle="--", label="3% (smooth limit)")
plt.axhline(10, color="darkred", linestyle="--", label="10% (textured limit)")
plt.title("Pattern classification: edge density (Canny)")
plt.ylabel("Edge pixels (%)")
plt.xticks(rotation=45)
plt.legend()
save_figure(5)

# 6. Step by step: how the noise level and the dominant pitch of a sound are obtained
example = audio_df[audio_df["Sound"] == "cow"].iloc[0]
y_raw, sr = librosa.load(os.path.join(DATA_DIR, "Audio", example["File"]), sr=None)
y = butter_highpass_filter(y_raw, 50, sr)
rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=y)[0], ref=1.0)
frame_times = librosa.times_like(rms_db, sr=sr)
N = len(y)
xf = fftfreq(N, 1 / sr)[:N // 2]
yf_raw = np.abs(fft(y_raw))[:N // 2]
yf = np.abs(fft(y))[:N // 2]
dominant = example["Dominant_freq_Hz"]

fig, axes = plt.subplots(2, 3, figsize=(22, 10))
librosa.display.waveshow(y_raw, sr=sr, ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("1. Original sound (librosa.load)")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Amplitude")

low = xf <= 300
axes[0, 1].plot(xf[low], yf_raw[low], color="grey", label="Before the filter")
axes[0, 1].plot(xf[low], yf[low], color="#4C72B0", label="After the filter")
axes[0, 1].axvline(50, color="red", linestyle="--", label="Cutoff 50 Hz")
axes[0, 1].set_title("2. High-pass Butterworth filter (removes the rumble < 50 Hz)")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("Magnitude")
axes[0, 1].legend()

axes[0, 2].plot(frame_times, rms_db, color="#4C72B0")
axes[0, 2].axhline(example["Noise_floor_dB"] + example["SNR_dB"], color="green", linestyle="--",
                   label=f"Sound level (95%): {example['Noise_floor_dB'] + example['SNR_dB']:.1f} dB")
axes[0, 2].axhline(example["Noise_floor_dB"], color="red", linestyle="--",
                   label=f"Background noise (10%): {example['Noise_floor_dB']:.1f} dB")
axes[0, 2].set_title(f"3. RMS energy in dB -> SNR = {example['SNR_dB']:.1f} dB ({example['Noise_class']})")
axes[0, 2].set_xlabel("Time (s)")
axes[0, 2].set_ylabel("Level (dB)")
axes[0, 2].legend()

valid = (xf >= 50) & (xf <= 10000)
axes[1, 0].plot(xf[valid], yf[valid], color="#55A868")
axes[1, 0].axvline(dominant, color="red", linestyle="--", label=f"Maximum: {dominant:.0f} Hz")
axes[1, 0].set_title("4. FFT spectrum (50 Hz - 10 kHz)")
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("Magnitude")
axes[1, 0].legend()

zoom = (xf >= dominant - 200) & (xf <= dominant + 200)
axes[1, 1].plot(xf[zoom], yf[zoom], color="#55A868")
axes[1, 1].axvspan(dominant - 20, dominant + 20, color="red", alpha=0.2, label="Dominant pitch +/- 20 Hz")
axes[1, 1].set_title(f"5. Energy of the pitch = {example['Pitch_energy_%']:.1f}% -> {example['Pitch_class']}")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("Magnitude")
axes[1, 1].legend()

D = librosa.amplitude_to_db(np.abs(librosa.stft(y_raw)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=axes[1, 2])
axes[1, 2].axhline(dominant, color="white", linestyle="--")
axes[1, 2].set_title(f"6. Spectrogram: dominant pitch {dominant:.0f} Hz ({example['Note']})")
plt.suptitle(f"Step by step: noise level (top) and dominant pitch (bottom) - {example['File']}", fontsize=17)
save_figure(6)

audio_colors = dict(zip(audio_df["Sound"], sns.color_palette("Set2", len(audio_df))))

# 7. Amplitude (waveform) of the 8 recordings
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
for ax, (_, row) in zip(axes.flatten(), audio_df.iterrows()):
    y, sr = librosa.load(os.path.join(DATA_DIR, "Audio", row["File"]), sr=None)
    librosa.display.waveshow(y, sr=sr, ax=ax, color=audio_colors[row["Sound"]])
    ax.set_title(f"{row['Sound']} - {row['Noise_class']} (SNR {row['SNR_dB']:.1f} dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
plt.suptitle("Amplitude of the 8 recordings", fontsize=16)
save_figure(7)

# 8. FFT spectrum of the 8 recordings (dominant pitch)
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
for ax, (_, row) in zip(axes.flatten(), audio_df.iterrows()):
    y, sr = librosa.load(os.path.join(DATA_DIR, "Audio", row["File"]), sr=None)
    N = len(y)
    yf = np.abs(fft(y))[:N // 2]
    xf = fftfreq(N, 1 / sr)[:N // 2]
    ax.plot(xf, yf, color=audio_colors[row["Sound"]])
    ax.axvline(row["Dominant_freq_Hz"], color="red", linestyle="--",
               label=f"{row['Dominant_freq_Hz']:.0f} Hz ({row['Note']}) - {row['Pitch_class']}")
    ax.set_xlim(0, 10000)
    ax.set_title(f"FFT spectrum - {row['Sound']}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend(loc="upper right")
plt.suptitle("FFT spectrum of the 8 recordings: dominant pitch", fontsize=16)
save_figure(8)

# 9. Spectrograms of the 8 recordings with their dominant pitch
fig, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)
for ax, (_, row) in zip(axes.T.flatten(), audio_df.iterrows()):  # 4 sounds in each column
    y, sr = librosa.load(os.path.join(DATA_DIR, "Audio", row["File"]), sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax)
    ax.axhline(row["Dominant_freq_Hz"], color="white", linestyle="--")
    ax.set_title(f"{row['Sound']}: dominant pitch {row['Dominant_freq_Hz']:.0f} Hz ({row['Note']})")
fig.colorbar(img, ax=axes, format="%+2.0f dB")
plt.savefig(os.path.join(RESULTS_DIR, "RESULTADO9.png"), dpi=120)
plt.show()

# 10. Noise level vs dominant pitch
plt.figure(figsize=(10, 6))
sns.scatterplot(data=audio_df, x="Dominant_freq_Hz", y="SNR_dB", hue="Sound", style="Pitch_class", s=150)
plt.xscale("log")
plt.axhline(8, color="red", linestyle="--", label="8 dB (high noise limit)")
plt.axhline(20, color="orange", linestyle="--", label="20 dB (low noise limit)")
plt.title("Audio: noise level (SNR) vs dominant pitch")
plt.xlabel("Dominant frequency (Hz, log scale)")
plt.ylabel("Signal-to-noise ratio (dB)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
save_figure(10)

# 11. Most frequent words
plt.figure(figsize=(10, 5))
top = pd.DataFrame(all_words.most_common(15), columns=["Word", "Frequency"])
sns.barplot(data=top, x="Frequency", y="Word", color="#4C72B0")
plt.title("15 most frequent words in the texts (without stopwords)")
save_figure(11)

# 12. Top 10 keywords of each topic: words much more frequent in that topic than in the others
topics = list(TOPIC_KEYWORDS.keys())
topic_colors = dict(zip(topics, ["#E41A1C", "#377EB8", "#984EA3"]))
fig, axes = plt.subplots(1, len(topics), figsize=(20, 6))
for ax, topic in zip(axes, topics):
    in_topic, others = FreqDist(), FreqDist()
    for freq, t in zip(text_df["Freq"], text_df["Topic"]):
        if t == topic:
            in_topic.update(freq)
        else:
            others.update(freq)
    scores = {w: in_topic.freq(w) - others.freq(w) for w in in_topic if in_topic[w] >= 3}
    keywords = sorted(scores, key=scores.get, reverse=True)[:10]
    print(f"Top keywords of {topic}: {', '.join(keywords)}")
    ax.barh(keywords[::-1], [in_topic[w] for w in keywords[::-1]], color=topic_colors[topic])
    n_texts = (text_df["Topic"] == topic).sum()
    ax.set_title(f"Top 10 keywords - {topic} ({n_texts} texts)")
    ax.set_xlabel("Frequency in the topic")
save_figure(12)

# 13. Writing style of each topic (boxplots)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, feature in zip(axes, ["Avg_sentence_length", "Lexical_diversity", "Avg_word_length"]):
    sns.boxplot(data=text_df, x="Topic", y=feature, order=topics, hue="Topic", palette=topic_colors, ax=ax)
    sns.stripplot(data=text_df, x="Topic", y=feature, order=topics, color="black", size=6, ax=ax)
    ax.set_title(feature.replace("_", " "))
    ax.set_xlabel("Topic")
save_figure(13)

# 14. Keyword score of each text
plt.figure(figsize=(8, 6))
scores = text_df.set_index("File")[score_cols]
scores.columns = [c.replace("Score_", "") for c in score_cols]
sns.heatmap(scores, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Keyword score of each text per topic (%)")
plt.xlabel("Topic")
plt.ylabel("")
save_figure(14)

# 15. Summary of the three media types
image_colors = {"Red": "#D62728", "Orange": "#FF7F0E", "Yellow": "#E8C547", "Green": "#2CA02C",
                "Cyan": "#17BECF", "Blue": "#1F77B4", "Purple": "#9467BD", "Grayscale": "#7F7F7F"}
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

color_counts = images_df["Color_class"].value_counts()
color_counts.plot(kind="bar", ax=axes[0], color=[image_colors[c] for c in color_counts.index], edgecolor="black")
axes[0].set_title("Images per color class")
axes[0].set_ylabel("Number of files")
axes[0].tick_params(axis="x", rotation=45)

audio_counts = (audio_df["Noise_class"] + " - " + audio_df["Pitch_class"]).value_counts().sort_values()
audio_counts.plot(kind="barh", ax=axes[1], color="#55A868", edgecolor="black")
axes[1].set_title("Audio files per class (noise - pitch)")
axes[1].set_xlabel("Number of files")

topic_counts = text_df["Topic"].value_counts().reindex(topics, fill_value=0)
topic_counts.plot(kind="bar", ax=axes[2], color=[topic_colors[t] for t in topics], edgecolor="black")
axes[2].set_title("Texts per predicted topic")
axes[2].set_ylabel("Number of files")
axes[2].tick_params(axis="x", rotation=0)

axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # whole numbers only
axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[2].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
for ax in axes:
    for container in ax.containers:
        ax.bar_label(container)
plt.suptitle(f"Automatic classification of the {len(summary)} media files", fontsize=16)
save_figure(15)

print("\nAll the figures were saved in:", RESULTS_DIR)