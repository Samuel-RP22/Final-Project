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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

for resource, path in [("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

sns.set_style("whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def save_figure(number):
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"RESULTADO{number}.png"), dpi=120)
    plt.show()


# ---------------------------------------------------------------------------
# 1) IMAGES: color and pattern
# ---------------------------------------------------------------------------
# HUE = Tone of the color (0-179 in OpenCV)
HUE_RANGES = {"Red": [(0, 9), (170, 179)], "Orange": [(10, 21)], "Yellow": [(22, 34)],
              "Green": [(35, 84)], "Cyan": [(85, 99)], "Blue": [(100, 129)], "Purple": [(130, 169)]}


def load_image(path):
    img = cv2.imread(path)
    return cv2.resize(img, (512, int(512 * img.shape[0] / img.shape[1])))


# Color = hue histogram of the pixels with a clear color and % of each color range
def color_steps(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 50)).astype(np.uint8)
    hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180]).flatten()
    pixels = {name: sum(hist[a:b + 1].sum() for a, b in ranges) for name, ranges in HUE_RANGES.items()}
    percent = {name: value / max(sum(pixels.values()), 1) * 100 for name, value in pixels.items()}
    return hsv, mask, hist, percent


# Pattern = grayscale, blur and Canny edges
def edge_steps(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred, cv2.Canny(blurred, 100, 200)


def analyze_image(img):
    hsv, mask, hist, percent = color_steps(img)
    gray, blurred, edges = edge_steps(img)
    return {"Brightness": gray.mean(),
            "Saturation": hsv[:, :, 1].mean(),
            "Colorful_%": mask.mean() * 100,
            "Dominant_hue": max(percent, key=percent.get),
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


def load_audio(path):
    y_raw, sr = librosa.load(path, sr=None)
    return y_raw, butter_highpass_filter(y_raw, 50, sr), sr


def rms_db(y):
    return librosa.amplitude_to_db(librosa.feature.rms(y=y)[0], ref=1.0).astype(float)


def fft_spectrum(y, sr):
    N = len(y)
    return fftfreq(N, 1 / sr)[:N // 2], np.abs(fft(y))[:N // 2]


def spectrogram_db(y):
    return librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)


def analyze_audio(y, sr):
    levels = rms_db(y)
    noise_floor, peak = np.percentile(levels, 10), np.percentile(levels, 95)
    xf, yf = fft_spectrum(y, sr)
    valid = (xf >= 50) & (xf <= 10000)
    dominant = xf[valid][np.argmax(yf[valid])]
    band = valid & (np.abs(xf - dominant) <= 20)
    return {"Duration_s": librosa.get_duration(y=y, sr=sr),
            "Noise_floor_dB": noise_floor,
            "SNR_dB": peak - noise_floor,
            "Dominant_freq_Hz": dominant,
            "Note": librosa.hz_to_note(dominant),
            "Pitch_energy_%": (yf[band] ** 2).sum() / (yf[valid] ** 2).sum() * 100}


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


def analyze_text(text):
    words = [w for w in word_tokenize(text.lower()) if w.isalpha()]
    content_words = [w for w in words if w not in stop_words and len(w) > 2]
    freq = FreqDist(content_words)
    scores = {topic: sum(freq[w] for w in kws) / len(content_words) * 100 for topic, kws in TOPIC_KEYWORDS.items()}
    return {"Words": len(words),
            "Unique_words": len(set(words)),
            "Lexical_diversity": len(set(words)) / len(words),
            "Avg_sentence_length": len(words) / len(sent_tokenize(text)),
            "Avg_word_length": np.mean([len(w) for w in words]),
            "Top_words": ", ".join(w for w, c in freq.most_common(5)),
            **{f"Score_{t}": s for t, s in scores.items()},
            "Topic": max(scores, key=scores.get),
            "Freq": freq}


# Keywords of a topic: words much more frequent in that topic than in the others
def topic_keywords(text_df, topic, n=10):
    in_topic, others = FreqDist(), FreqDist()
    for freq, t in zip(text_df["Freq"], text_df["Topic"]):
        (in_topic if t == topic else others).update(freq)
    scores = {w: in_topic.freq(w) - others.freq(w) for w in in_topic if in_topic[w] >= 3}
    return sorted(scores, key=scores.get, reverse=True)[:n], in_topic


# ---------------------------------------------------------------------------
# 4) AUTOMATED WORKFLOW: every file goes to its analysis by its extension
# ---------------------------------------------------------------------------
images, audios, texts = [], [], []
image_data, audio_data = {}, {}
for root, dirs, files in os.walk(DATA_DIR):
    for name in sorted(files):
        path = os.path.join(root, name)
        ext = os.path.splitext(name)[1].lower()
        if ext in (".jpg", ".jpeg", ".png"):
            image_data[name] = load_image(path)
            images.append({"File": name, **analyze_image(image_data[name])})
        elif ext in (".wav", ".mp3", ".ogg"):
            audio_data[name] = load_audio(path)
            y_raw, y, sr = audio_data[name]
            audios.append({"File": name, **analyze_audio(y, sr)})
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8") as file:
                texts.append({"File": name, **analyze_text(file.read())})

images_df = pd.DataFrame(images)
audio_df = pd.DataFrame(audios)
text_df = pd.DataFrame(texts)
images_df = pd.concat([images_df, images_df.apply(classify_image, axis=1)], axis=1)
audio_df = pd.concat([audio_df, audio_df.apply(classify_audio, axis=1)], axis=1)
audio_df["Sound"] = audio_df["File"].str.rsplit(".", n=1).str[0]
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

topics = list(TOPIC_KEYWORDS)
keywords = {topic: topic_keywords(text_df, topic) for topic in topics}
for topic, (words, _) in keywords.items():
    print(f"Top keywords of {topic}: {', '.join(words)}")

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
COLORS = {"Red": "#D62728", "Orange": "#FF7F0E", "Yellow": "#E8C547", "Green": "#2CA02C",
          "Cyan": "#17BECF", "Blue": "#1F77B4", "Purple": "#9467BD", "Grayscale": "#7F7F7F"}
audio_colors = dict(zip(audio_df["Sound"], sns.color_palette("Set2", len(audio_df))))
topic_colors = dict(zip(topics, ["#E41A1C", "#377EB8", "#984EA3"]))


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# 1. Images with their class
fig, axes = plt.subplots(2, 5, figsize=(15, 7.5), gridspec_kw={"hspace": 0.3})
for ax, (_, row) in zip(axes.flatten(), images_df.iterrows()):
    ax.imshow(rgb(image_data[row["File"]]))
    ax.set_title(f"{row['File']}\n{row['Color_class']} | {row['Pattern_class']}", fontsize=10)
    ax.axis("off")
plt.suptitle("Image classification (color | pattern)", fontsize=15)
save_figure(1)

# 2. Step by step: color identification (starry_night.jpg) and edge detection (coins.png)
img = image_data["starry_night.jpg"]
hsv, mask, hist, percent = color_steps(img)
colorful = rgb(img).copy()
colorful[mask == 0] = 0
hue_image = hsv_to_rgb(np.dstack([hsv[:, :, 0] / 179, np.ones(mask.shape), np.ones(mask.shape)]))

fig, axes = plt.subplots(2, 5, figsize=(26, 11))
for ax, picture, title in zip(axes[0], [rgb(img), hue_image, colorful],
                              ["1. Original image", "2. Hue channel (H of HSV)",
                               f"3. Colorful pixels (S > 60 and V > 50): {mask.mean() * 100:.0f}%"]):
    ax.imshow(picture)
    ax.set_title(title)
    ax.axis("off")
axes[0, 3].bar(range(180), hist, width=1, color=[hsv_to_rgb([h / 179, 1, 1]) for h in range(180)])
axes[0, 3].set(title="4. Hue histogram (cv2.calcHist)", xlabel="Hue (0-179)", ylabel="Number of pixels")
bars = axes[0, 4].bar(percent.keys(), percent.values(), color=[COLORS[c] for c in percent], edgecolor="black")
axes[0, 4].bar_label(bars, fmt="%.0f%%")
axes[0, 4].set(title=f"5. Pixels per color -> dominant color: {max(percent, key=percent.get)}",
               ylabel="% of the colorful pixels")
axes[0, 4].tick_params(axis="x", rotation=45)

img = image_data["coins.png"]
gray, blurred, edges = edge_steps(img)
overlay = rgb(img).copy()
overlay[edges > 0] = [255, 0, 0]
for ax, picture, title in zip(axes[1], [rgb(img), gray, blurred, edges, overlay],
                              ["1. Original image", "2. Grayscale (cv2.cvtColor)",
                               "3. Gaussian blur 5x5 (removes small noise)",
                               f"4. Canny edges (100, 200): {(edges > 0).mean() * 100:.1f}% of pixels",
                               "5. Edges over the original image"]):
    ax.imshow(picture, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.suptitle("Step by step: color identification (top) and edge detection (bottom)", fontsize=17)
save_figure(2)

# 3. Color identification: color histogram (B, G, R) of each image
fig, axes = plt.subplots(4, 5, figsize=(20, 14))
for i, (_, row) in enumerate(images_df.iterrows()):
    img = image_data[row["File"]]
    ax_img, ax_hist = axes[(i // 5) * 2, i % 5], axes[(i // 5) * 2 + 1, i % 5]
    ax_img.imshow(rgb(img))
    ax_img.set_title(row["File"])
    ax_img.axis("off")
    for channel, color in enumerate(["b", "g", "r"]):
        ax_hist.plot(cv2.calcHist([img], [channel], None, [256], [0, 256]), color=color)
    ax_hist.set_title(f"Color: {row['Color_class']} ({row['Colorful_%']:.0f}% colorful pixels)", fontsize=10)
    ax_hist.set(xlabel="Pixel value", ylabel="Number of pixels")
plt.suptitle("Color identification: color histogram (B, G, R) of each image", fontsize=15)
save_figure(3)

# 4. Pattern identification: Canny edges of each image
fig, axes = plt.subplots(4, 5, figsize=(20, 14))
for i, (_, row) in enumerate(images_df.iterrows()):
    gray, blurred, edges = edge_steps(image_data[row["File"]])
    ax_img, ax_edges = axes[(i // 5) * 2, i % 5], axes[(i // 5) * 2 + 1, i % 5]
    ax_img.imshow(gray, cmap="gray")
    ax_img.set_title(row["File"])
    ax_edges.imshow(edges, cmap="gray")
    ax_edges.set_title(f"Canny edges: {row['Edge_density_%']:.1f}% ({row['Pattern_class']})", fontsize=10)
    ax_img.axis("off")
    ax_edges.axis("off")
plt.suptitle("Pattern identification: Canny edges of each image", fontsize=15)
save_figure(4)

# 5. Pattern classification: edge density of each image
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

# 6. Step by step: noise level and dominant pitch of cow.wav
example = audio_df[audio_df["Sound"] == "cow"].iloc[0]
y_raw, y, sr = audio_data[example["File"]]
levels = rms_db(y)
xf, yf = fft_spectrum(y, sr)
_, yf_raw = fft_spectrum(y_raw, sr)
dominant = example["Dominant_freq_Hz"]
sound_level = example["Noise_floor_dB"] + example["SNR_dB"]
low, valid = xf <= 300, (xf >= 50) & (xf <= 10000)
zoom = (xf >= dominant - 200) & (xf <= dominant + 200)

fig, axes = plt.subplots(2, 3, figsize=(22, 10))
librosa.display.waveshow(y_raw, sr=sr, ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set(title="1. Original sound (librosa.load)", xlabel="Time (s)", ylabel="Amplitude")

axes[0, 1].plot(xf[low], yf_raw[low], color="grey", label="Before the filter")
axes[0, 1].plot(xf[low], yf[low], color="#4C72B0", label="After the filter")
axes[0, 1].axvline(50, color="red", linestyle="--", label="Cutoff 50 Hz")
axes[0, 1].set(title="2. High-pass Butterworth filter (removes the rumble < 50 Hz)",
               xlabel="Frequency (Hz)", ylabel="Magnitude")

axes[0, 2].plot(librosa.times_like(levels, sr=sr), levels, color="#4C72B0")
axes[0, 2].axhline(sound_level, color="green", linestyle="--", label=f"Sound level (95%): {sound_level:.1f} dB")
axes[0, 2].axhline(example["Noise_floor_dB"], color="red", linestyle="--",
                   label=f"Background noise (10%): {example['Noise_floor_dB']:.1f} dB")
axes[0, 2].set(title=f"3. RMS energy in dB -> SNR = {example['SNR_dB']:.1f} dB ({example['Noise_class']})",
               xlabel="Time (s)", ylabel="Level (dB)")

axes[1, 0].plot(xf[valid], yf[valid], color="#55A868")
axes[1, 0].axvline(dominant, color="red", linestyle="--", label=f"Maximum: {dominant:.0f} Hz")
axes[1, 0].set(title="4. FFT spectrum (50 Hz - 10 kHz)", xlabel="Frequency (Hz)", ylabel="Magnitude")

axes[1, 1].plot(xf[zoom], yf[zoom], color="#55A868")
axes[1, 1].axvspan(dominant - 20, dominant + 20, color="red", alpha=0.2, label="Dominant pitch +/- 20 Hz")
axes[1, 1].set(title=f"5. Energy of the pitch = {example['Pitch_energy_%']:.1f}% -> {example['Pitch_class']}",
               xlabel="Frequency (Hz)", ylabel="Magnitude")

librosa.display.specshow(spectrogram_db(y_raw), sr=sr, x_axis="time", y_axis="log", ax=axes[1, 2])
axes[1, 2].axhline(dominant, color="white", linestyle="--")
axes[1, 2].set_title(f"6. Spectrogram: dominant pitch {dominant:.0f} Hz ({example['Note']})")
for ax in [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]:
    ax.legend()
plt.suptitle(f"Step by step: noise level (top) and dominant pitch (bottom) - {example['File']}", fontsize=17)
save_figure(6)

# 7. Amplitude (waveform) of the 8 recordings
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
for ax, (_, row) in zip(axes.flatten(), audio_df.iterrows()):
    y_raw, y, sr = audio_data[row["File"]]
    librosa.display.waveshow(y_raw, sr=sr, ax=ax, color=audio_colors[row["Sound"]])
    ax.set(title=f"{row['Sound']} - {row['Noise_class']} (SNR {row['SNR_dB']:.1f} dB)",
           xlabel="Time (s)", ylabel="Amplitude")
plt.suptitle("Amplitude of the 8 recordings", fontsize=16)
save_figure(7)

# 8. FFT spectrum of the 8 recordings (dominant pitch)
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
for ax, (_, row) in zip(axes.flatten(), audio_df.iterrows()):
    y_raw, y, sr = audio_data[row["File"]]
    xf, yf = fft_spectrum(y, sr)
    ax.plot(xf, yf, color=audio_colors[row["Sound"]])
    ax.axvline(row["Dominant_freq_Hz"], color="red", linestyle="--",
               label=f"{row['Dominant_freq_Hz']:.0f} Hz ({row['Note']}) - {row['Pitch_class']}")
    ax.set(xlim=(0, 10000), title=f"FFT spectrum - {row['Sound']}", xlabel="Frequency (Hz)", ylabel="Magnitude")
    ax.legend(loc="upper right")
plt.suptitle("FFT spectrum of the 8 recordings: dominant pitch", fontsize=16)
save_figure(8)

# 9. Spectrograms of the 8 recordings with their dominant pitch (4 sounds in each column)
fig, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)
for ax, (_, row) in zip(axes.T.flatten(), audio_df.iterrows()):
    y_raw, y, sr = audio_data[row["File"]]
    img = librosa.display.specshow(spectrogram_db(y_raw), sr=sr, x_axis="time", y_axis="log", ax=ax)
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

# 12. Top 10 keywords of each topic
fig, axes = plt.subplots(1, len(topics), figsize=(20, 6))
for ax, topic in zip(axes, topics):
    words, in_topic = keywords[topic]
    ax.barh(words[::-1], [in_topic[w] for w in words[::-1]], color=topic_colors[topic])
    ax.set(title=f"Top 10 keywords - {topic} ({(text_df['Topic'] == topic).sum()} texts)",
           xlabel="Frequency in the topic")
save_figure(12)

# 13. Writing style of each topic (boxplots)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, feature in zip(axes, ["Avg_sentence_length", "Lexical_diversity", "Avg_word_length"]):
    sns.boxplot(data=text_df, x="Topic", y=feature, order=topics, hue="Topic", palette=topic_colors, ax=ax)
    sns.stripplot(data=text_df, x="Topic", y=feature, order=topics, color="black", size=6, ax=ax)
    ax.set(title=feature.replace("_", " "), xlabel="Topic")
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
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
color_counts = images_df["Color_class"].value_counts()
color_counts.plot(kind="bar", ax=axes[0], color=[COLORS[c] for c in color_counts.index], edgecolor="black")
axes[0].set(title="Images per color class", ylabel="Number of files")
axes[0].tick_params(axis="x", rotation=45)

audio_counts = (audio_df["Noise_class"] + " - " + audio_df["Pitch_class"]).value_counts().sort_values()
audio_counts.plot(kind="barh", ax=axes[1], color="#55A868", edgecolor="black")
axes[1].set(title="Audio files per class (noise - pitch)", xlabel="Number of files")

topic_counts = text_df["Topic"].value_counts().reindex(topics, fill_value=0)
topic_counts.plot(kind="bar", ax=axes[2], color=[topic_colors[t] for t in topics], edgecolor="black")
axes[2].set(title="Texts per predicted topic", ylabel="Number of files")
axes[2].tick_params(axis="x", rotation=0)

axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[2].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
for ax in axes:
    for container in ax.containers:
        ax.bar_label(container)
plt.suptitle(f"Automatic classification of the {len(summary)} media files", fontsize=16)
save_figure(15)

print("\nAll the figures were saved in:", RESULTS_DIR)