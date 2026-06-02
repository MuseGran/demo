"""
Pre-render waveform and mel spectrogram PNGs for the interactive segment players.
Replicates the exact rendering logic from the JS in index.html.
"""

import json
import numpy as np
import librosa
from PIL import Image, ImageDraw

# Same magma colormap stops as in the JS code
MAGMA_STOPS = [
    (0,0,4),(1,0,11),(3,1,22),(7,2,36),(14,4,52),(23,6,68),(34,8,82),(47,10,93),
    (62,12,101),(77,14,106),(92,16,110),(107,18,112),(122,19,113),(137,21,112),
    (152,24,110),(166,28,106),(180,34,101),(192,42,95),(203,51,89),(213,62,82),
    (222,73,76),(229,86,70),(235,99,64),(240,113,59),(244,127,55),(247,142,51),
    (249,157,48),(251,172,46),(252,187,46),(252,202,48),(251,218,53),(251,233,63),
    (252,249,79)
]

def build_magma_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    n_stops = len(MAGMA_STOPS)
    for i in range(256):
        t = i / 255.0 * (n_stops - 1)
        idx = int(t)
        frac = t - idx
        c0 = MAGMA_STOPS[min(idx, n_stops - 1)]
        c1 = MAGMA_STOPS[min(idx + 1, n_stops - 1)]
        lut[i] = [
            int(c0[0] + (c1[0] - c0[0]) * frac),
            int(c0[1] + (c1[1] - c0[1]) * frac),
            int(c0[2] + (c1[2] - c0[2]) * frac),
        ]
    return lut

MAGMA_LUT = build_magma_lut()

# Segment data extracted from index.html data-segments attributes
SEGMENTS = {
    1: [
        {"start":0,"end":13.0},{"start":13.0,"end":26.2},{"start":26.2,"end":56.0},
        {"start":56.0,"end":69.5},{"start":69.5,"end":82.2},{"start":82.2,"end":107.0},
        {"start":107.0,"end":124.3},{"start":124.3,"end":137.9}
    ],
    2: [
        {"start":0,"end":3.5},{"start":3.5,"end":42.6},{"start":42.6,"end":60.0},
        {"start":60.0,"end":84.0},{"start":84.0,"end":120.0}
    ],
    3: [
        {"start":0,"end":1.9},{"start":1.9,"end":33.2},{"start":33.2,"end":56.9},
        {"start":56.9,"end":88.0},{"start":88.0,"end":118.7},{"start":118.7,"end":120.0}
    ],
}

AUDIO_FILES = {1: "1.mp3", 2: "2.mp3", 3: "3.mp3"}


def alpha_blend(fg, alpha, bg):
    """Blend fg color over bg color with given alpha (0..1), like canvas does."""
    return tuple(int(fg[i] * alpha + bg[i] * (1 - alpha)) for i in range(3))


def render_waveform(audio_path, segments, output_path, width=1200, height=80):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr

    bg_color = (248, 250, 252)  # #f8fafc
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Segment background bands (pre-compute alpha-blended colors like canvas would)
    band_fg = (90, 125, 149)
    band_even_color = alpha_blend(band_fg, 0.1, bg_color)  # rgba(90,125,149,0.1)
    band_odd_color = alpha_blend(band_fg, 0.03, bg_color)  # rgba(90,125,149,0.03)
    for i, seg in enumerate(segments):
        x0 = int((seg["start"] / duration) * width)
        x1 = int((seg["end"] / duration) * width)
        color = band_even_color if i % 2 == 0 else band_odd_color
        draw.rectangle([x0, 0, x1, height], fill=color)

    # Compute waveform envelope
    samples_per_px = len(y) // width
    wave_height = 30  # half-height in pixels
    center_y = height // 2

    # Find peak for normalization
    mins = np.zeros(width)
    maxs = np.zeros(width)
    for px in range(width):
        start = px * samples_per_px
        end = min(start + samples_per_px, len(y))
        chunk = y[start:end]
        if len(chunk) > 0:
            mins[px] = chunk.min()
            maxs[px] = chunk.max()

    peak_amp = max(abs(mins.min()), abs(maxs.max()))
    scale = wave_height / peak_amp if peak_amp > 0 else 1.0

    # Draw waveform lines
    waveform_color = (74, 124, 155)  # #4a7c9b
    for px in range(width):
        y_min = int(center_y - mins[px] * scale)
        y_max = int(center_y - maxs[px] * scale)
        if y_min > y_max:
            draw.line([(px, y_max), (px, y_min)], fill=waveform_color, width=1)
        else:
            draw.line([(px, y_min), (px, y_max)], fill=waveform_color, width=1)

    # Segment boundary dashed lines
    dash_on, dash_off = 4, 3
    boundary_color = (255, 255, 255)
    for seg in segments:
        if seg["start"] > 0:
            x = int((seg["start"] / duration) * width)
            y_pos = 0
            while y_pos < height:
                y_end = min(y_pos + dash_on, height)
                draw.line([(x, y_pos), (x, y_end)], fill=boundary_color, width=2)
                y_pos += dash_on + dash_off

    img.save(output_path)
    print(f"  Waveform saved: {output_path}")


def render_melspec(audio_path, segments, output_path, width=1200, height=120, top_db=80, freq_gamma=1.0):
    """freq_gamma > 1 stretches low frequencies, compresses high frequencies."""
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmax = 16000

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr

    # Compute mel spectrogram (matches JS: slaney norm, power spectrogram)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmax=fmax, fmin=0, norm="slaney", power=2.0
    )

    # power_to_db: ref=max, top_db=80
    S_db = 10 * np.log10(S + 1e-10)
    global_max = S_db.max()
    db_floor = global_max - top_db
    S_db = np.clip(S_db, db_floor, global_max)
    S_norm = (S_db - db_floor) / top_db  # 0..1

    # Resample to width
    n_frames = S_norm.shape[1]
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    for px in range(width):
        frame_idx = min(int(px / width * n_frames), n_frames - 1)
        for row in range(height):
            # ratio: 0=top(high freq), 1=bottom(low freq)
            ratio = row / height
            # gamma > 1: low freq gets more vertical space
            mel_ratio = 1 - ratio ** freq_gamma
            mel_idx = min(int(mel_ratio * n_mels), n_mels - 1)
            val = S_norm[mel_idx, frame_idx]
            c_idx = int(round(val * 255))
            img_array[row, px] = MAGMA_LUT[c_idx]

    img = Image.fromarray(img_array, "RGB")
    draw = ImageDraw.Draw(img)

    # Segment boundary dashed lines
    dash_on, dash_off = 4, 3
    boundary_color = (255, 255, 255)
    for seg in segments:
        if seg["start"] > 0:
            x = int((seg["start"] / duration) * width)
            y_pos = 0
            while y_pos < height:
                y_end = min(y_pos + dash_on, height)
                draw.line([(x, y_pos), (x, y_end)], fill=boundary_color, width=2)
                y_pos += dash_on + dash_off

    img.save(output_path)
    print(f"  Mel spectrogram saved: {output_path}")


# Planner segment boundaries (from HTML po-segs, converted to seconds)
PLANNER_SEGMENTS = {
    1: [
        {"start": 0, "end": 19}, {"start": 19, "end": 38},
        {"start": 38, "end": 56}, {"start": 56, "end": 73},
    ],
    2: [
        {"start": 0, "end": 19}, {"start": 19, "end": 38},
        {"start": 38, "end": 58}, {"start": 58, "end": 77},
        {"start": 77, "end": 98},
    ],
    3: [
        {"start": 0, "end": 24}, {"start": 24, "end": 40},
        {"start": 40, "end": 56}, {"start": 56, "end": 74},
        {"start": 74, "end": 88}, {"start": 88, "end": 104},
        {"start": 104, "end": 120}, {"start": 120, "end": 142},
    ],
}

# A/B comparison "with segment" boundaries
SEG_WITHSEG_SEGMENTS = {
    1: [
        {"start": 0, "end": 26.3}, {"start": 26.3, "end": 41.8},
        {"start": 41.8, "end": 60.0}, {"start": 60.0, "end": 83.1},
        {"start": 83.1, "end": 120.0},
    ],
    2: [
        {"start": 0, "end": 17.5}, {"start": 17.5, "end": 60.0},
        {"start": 60.0, "end": 80.3}, {"start": 80.3, "end": 117.8},
        {"start": 117.8, "end": 120.0},
    ],
}


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "figures", "spectrograms")
    os.makedirs(out_dir, exist_ok=True)

    # Interactive segment players
    for n, audio_file in AUDIO_FILES.items():
        audio_path = os.path.join(base_dir, audio_file)
        segs = SEGMENTS[n]
        print(f"Processing {audio_file}...")
        render_waveform(audio_path, segs, os.path.join(out_dir, f"interactive_{n}_waveform.png"))
        render_melspec(audio_path, segs, os.path.join(out_dir, f"interactive_{n}_melspec.png"))

    # StructPlanner spectrograms
    for n in [1, 2, 3]:
        audio_path = os.path.join(base_dir, "audio", f"planner_demo{n}.wav")
        segs = PLANNER_SEGMENTS[n]
        print(f"Processing planner_demo{n}.wav...")
        render_melspec(audio_path, segs, os.path.join(out_dir, f"planner_{n}_spec.png"))

    # A/B comparison spectrograms
    for n in [1, 2]:
        suffix = "" if n == 1 else f"_{n}"
        # Without segment control (no boundary lines)
        audio_path = os.path.join(base_dir, "audio", f"seg_noseg{suffix}.mp3")
        print(f"Processing seg_noseg{suffix}.mp3...")
        render_melspec(audio_path, [], os.path.join(out_dir, f"seg_noseg{suffix}_spec.png"))
        # With segment control (with boundary lines)
        audio_path = os.path.join(base_dir, "audio", f"seg_withseg{suffix}.mp3")
        segs = SEG_WITHSEG_SEGMENTS[n]
        print(f"Processing seg_withseg{suffix}.mp3...")
        render_melspec(audio_path, segs, os.path.join(out_dir, f"seg_withseg{suffix}_spec.png"))

    print("\nDone! All images generated.")
