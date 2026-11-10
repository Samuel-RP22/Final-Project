# -*- coding: utf-8 -*-
"""
Final Project - Python for Research
Option 1: Multimodal Media Classifier

@authors: Alejandro Garcia Castro and Samuel Riaño Patiño

The program reads the Datasets folder, detects the type of every file and
classifies it:
    - Images: by color (dominant color) and pattern (texture with Canny edges).
    - Audio: by noise level (signal-to-noise ratio) and dominant pitch (FFT).
    - Text: by word frequencies and keywords (topic of the text).
"""

# Importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import librosa
import librosa.display
import nltk
from nltk.tokenize import word_tokenize
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
    y = butter_highpass_filter(y, 50, sr)  # removes the very low rumble

    # Noise: loudness of every frame in dB
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=y)[0], ref=1.0).astype(float)
    noise_floor = np.percentile(rms_db, 10)  # quietest frames = background
    peak = np.percentile(rms_db, 95)         # loudest frames = the sound

    # Dominant pitch with the Fourier Transform
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
    # A sound that never stands out from its background (low SNR) is noisy
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
    # Score of each topic = % of the content words that are keywords of the topic
    scores = {topic: sum(freq[w] for w in kws) / len(content_words) * 100 for topic, kws in TOPIC_KEYWORDS.items()}
    return {"File": os.path.basename(path),
            "Words": len(words),
            "Unique_words": len(set(words)),
            "Lexical_diversity": len(set(words)) / len(words),
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
audio_df["Sound"] = audio_df["File"].str.rsplit("_", n=1).str[0]  # name of the sound (from the file name)
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
print("\nMean features per type of sound:")
print(audio_df.groupby("Sound")[["SNR_dB", "Dominant_freq_Hz"]].mean().round(1))

print("\n------------------ TEXT ------------------")
print(text_df[["Words", "Unique_words", "Lexical_diversity"]].describe().round(3))
score_cols = [c for c in text_df.columns if c.startswith("Score_")]
print(text_df[["File", "Top_words"] + score_cols + ["Topic"]].round(2).to_string(index=False))
print("\nMean statistics per topic:")
print(text_df.groupby("Topic")[["Words", "Lexical_diversity"]].mean().round(3))
all_words = FreqDist()
for freq in text_df["Freq"]:
    all_words.update(freq)
print("\nMost common words in all the texts:", all_words.most_common(10))

# Final table with the class of every file
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

# 2. Pattern: edge density of each image
plt.figure(figsize=(11, 5))
data = images_df.sort_values("Edge_density_%", ascending=False)
sns.barplot(data=data, x="File", y="Edge_density_%", hue="Pattern_class", dodge=False, palette="Blues_r")
plt.axhline(3, color="red", linestyle="--", label="3% (smooth limit)")
plt.axhline(10, color="darkred", linestyle="--", label="10% (textured limit)")
plt.title("Pattern classification: edge density (Canny)")
plt.ylabel("Edge pixels (%)")
plt.xticks(rotation=45)
plt.legend()
save_figure(2)

# 3. Spectrograms of one sound of each type
examples = audio_df.groupby("Sound").first().reset_index()
fig, axes = plt.subplots(1, len(examples), figsize=(20, 4.5))
for ax, (_, row) in zip(axes, examples.iterrows()):
    y, sr = librosa.load(os.path.join(DATA_DIR, "Audio", row["File"]), sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax)
    ax.axhline(row["Dominant_freq_Hz"], color="white", linestyle="--")
    ax.set_title(f"{row['Sound']}\n{row['Dominant_freq_Hz']:.0f} Hz | {row['Noise_class']}", fontsize=10)
save_figure(3)

# 4. Noise level vs dominant pitch
plt.figure(figsize=(10, 6))
sns.scatterplot(data=audio_df, x="Dominant_freq_Hz", y="SNR_dB", hue="Sound", style="Pitch_class", s=150)
plt.xscale("log")
plt.axhline(8, color="red", linestyle="--", label="8 dB (high noise limit)")
plt.axhline(20, color="orange", linestyle="--", label="20 dB (low noise limit)")
plt.title("Audio: noise level (SNR) vs dominant pitch")
plt.xlabel("Dominant frequency (Hz, log scale)")
plt.ylabel("Signal-to-noise ratio (dB)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
save_figure(4)

# 5. Most frequent words
plt.figure(figsize=(10, 5))
top = pd.DataFrame(all_words.most_common(15), columns=["Word", "Frequency"])
sns.barplot(data=top, x="Frequency", y="Word", color="#4C72B0")
plt.title("15 most frequent words in the texts (without stopwords)")
save_figure(5)

# 6. Keyword score of each text
plt.figure(figsize=(8, 6))
scores = text_df.set_index("File")[score_cols]
scores.columns = [c.replace("Score_", "") for c in score_cols]
sns.heatmap(scores, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Keyword score of each text per topic (%)")
plt.xlabel("Topic")
plt.ylabel("")
save_figure(6)

# 7. Summary of the three media types
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (media, color) in zip(axes, [("Image", "#4C72B0"), ("Audio", "#55A868"), ("Text", "#C44E52")]):
    counts = summary[summary["Media"] == media]["Class"].value_counts().sort_values()
    counts.plot(kind="barh", ax=ax, color=color, edgecolor="black")
    ax.set_title(f"{media}: files per class")
    ax.set_xlabel("Number of files")
    ax.set_ylabel("")
save_figure(7)

print("\nAll the figures were saved in:", RESULTS_DIR)
