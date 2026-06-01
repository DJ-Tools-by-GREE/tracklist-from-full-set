#!/usr/bin/env python3
"""
Tracklist Generator — Map an Engine DJ playlist against a recorded set
to produce a tracklist with timestamps.

Reads a playlist exported from Engine DJ as CSV, aligns each source track
against a long set recording (WAV) using chroma cross-correlation with
multiple tempo ratios, then writes:
  • tracklist.txt           — playlist with detected timestamps
  • confidence_heatmap.png  — per-track score curves over the set timeline

Handles DJ pitch-bend / tempo changes (e.g. 121 → 130 BPM) by searching across
a configurable range of playback speeds.  Tracks flagged as MASHUPs are
skipped — their vocals layer over a backing track without advancing the
timeline, so they don't appear in the tracklist.

Dependencies:
    pip install librosa soundfile numpy scipy matplotlib mutagen
    ffmpeg (for .mp4 conversion)
"""

import csv
import os
import sys
import subprocess
import sqlite3
import struct
import zlib
import tempfile
from dataclasses import dataclass, field
from typing import Optional

try:
    import numpy as np
    import librosa
    import soundfile as sf
    import matplotlib.pyplot as plt
    from scipy.signal import fftconvolve
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc}\n"
        "Install with: pip install librosa soundfile numpy scipy matplotlib mutagen"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these variables before running
# ═══════════════════════════════════════════════════════════════════════════════

# ── Inputs ────────────────────────────────────────────────────────────────────
PLAYLIST_CSV = os.path.expanduser(
    "~/Git_Repos/tracklist-from-full-set/Party Set V4.csv"
)
SET_WAV = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal/GREE/My Sets/Abiball Party Set V4/fullSet.wav"
)
ENGINE_DB_PATH = os.path.expanduser("~/Music/Engine Library/Database2/m.db")

# ── Outputs ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal/GREE/My Sets/Abiball Party Set V4"
)
OUTPUT_TRACKLIST = os.path.join(OUTPUT_DIR, "tracklist.txt")
OUTPUT_HEATMAP   = os.path.join(OUTPUT_DIR, "confidence_heatmap.png")

# ── Mashup overrides ──────────────────────────────────────────────────────────
# Track numbers (from the CSV "#" column) whose VOCALS only are layered over
# another track's backing.  These tracks are skipped — they don't appear in
# the tracklist and don't advance the timeline.
MASHUP_TRACK_NUMBERS: list = []     # e.g. [12, 18]

# ── Tempo search ──────────────────────────────────────────────────────────────
# DJ pitch-bend can shift speed/pitch by a few percent.  We search across
# DEFAULT_TEMPO_PCT first; if confidence is too low we automatically retry
# with WIDE_TEMPO_PCT (covers e.g. 121 → 130 BPM ≈ +7.4%).
DEFAULT_TEMPO_PCT = 3.0     # ±%
WIDE_TEMPO_PCT    = 10.0    # ±%  (used when default fails confidence threshold)
TEMPO_STEP_PCT    = 0.5     # search granularity (smaller = slower but finer)

# Per-track explicit override (CSV track number → max ±% to search).
# Use this to force a wide search for a transition you know is dramatic.
TEMPO_OVERRIDES: dict = {}  # e.g. {15: 12.0}

# ── Detection settings ────────────────────────────────────────────────────────
ANALYSIS_SR        = 22050     # downsample target for analysis
HOP_LENGTH         = 1024      # ~46 ms hop (chroma frame rate ≈ 21.5 Hz)
MIN_CONFIDENCE     = 0.40      # below this → retry with wide tempo range
PLAYED_THRESHOLD   = 0.55      # per-frame similarity threshold for "song is audible"
SEARCH_LOOKBACK    = 20        # search this many seconds before previous track's start
SEARCH_LOOKAHEAD   = 360       # search up to this many seconds after previous start

# ── Path remap (Engine DJ paths → current Mac locations) ──────────────────────
PATH_REMAPS = [
    ("C:/Users/Jan/OneDrive", "~/Library/CloudStorage/OneDrive-Personal"),
    ("~/Music/OneDrive",      "~/Library/CloudStorage/OneDrive-Personal"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Track:
    number:         int
    title:          str
    artist:         str
    length_secs:    int
    bpm:            float
    file_path:      str
    is_mashup:      bool  = False
    is_missing:     bool  = False

    # Detection results
    detected_start_secs:  Optional[float] = None
    detected_end_secs:    Optional[float] = None
    confidence:           float = 0.0
    tempo_ratio:          float = 1.0
    score_curve:          Optional[np.ndarray] = None   # full score curve over set timeline
    score_offsets_secs:   Optional[np.ndarray] = None   # x-axis (set time) for score_curve


# ═══════════════════════════════════════════════════════════════════════════════
# Path remapping
# ═══════════════════════════════════════════════════════════════════════════════

def remap_path(path: str) -> str:
    """Translate a stale (Windows) Engine DJ path to its current Mac location."""
    if not path:
        return path
    norm = path.replace("\\", "/")
    for legacy, current in PATH_REMAPS:
        legacy_norm  = os.path.expanduser(legacy).replace("\\", "/")
        current_norm = os.path.expanduser(current)
        if norm.lower().startswith(legacy_norm.lower()):
            rel = norm[len(legacy_norm):].lstrip("/")
            return os.path.normpath(os.path.join(current_norm, rel))
    return os.path.normpath(os.path.expanduser(norm))


# ═══════════════════════════════════════════════════════════════════════════════
# CSV reading
# ═══════════════════════════════════════════════════════════════════════════════

def read_playlist(csv_path: str) -> list:
    """Parse the Engine DJ playlist CSV.  Returns a list of Track objects in
    playlist order.  Mashup flag is applied from MASHUP_TRACK_NUMBERS."""
    tracks = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                num = int(row["#"])
            except (KeyError, ValueError):
                continue
            tracks.append(Track(
                number      = num,
                title       = (row.get("Title")  or "").strip(),
                artist      = (row.get("Artist") or "").strip(),
                length_secs = int(float(row["Length"])) if row.get("Length") else 0,
                bpm         = float(row["BPM"]) if row.get("BPM") else 0.0,
                file_path   = remap_path(row.get("File name", "")),
                is_mashup   = num in MASHUP_TRACK_NUMBERS,
            ))
    return tracks


# ═══════════════════════════════════════════════════════════════════════════════
# Audio loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio_mono(path: str, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Load any supported audio file as mono float32 at sr.
    Converts .mp4 to .mp3 first (ffmpeg) per project convention."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".mp4":
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", path,
                 "-vn", "-acodec", "libmp3lame", "-q:a", "2", tmp_path],
                capture_output=True,
            )
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg failed:\n{r.stderr.decode(errors='ignore')}")
            y, _ = librosa.load(tmp_path, sr=sr, mono=True)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return y

    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


# ═══════════════════════════════════════════════════════════════════════════════
# Chroma analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_chroma(y: np.ndarray, sr: int = ANALYSIS_SR, hop: int = HOP_LENGTH) -> np.ndarray:
    """Compute L2-normalized chroma_cqt feature sequence (12 × n_frames)."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    norms  = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
    return (chroma / norms).astype(np.float32)


def warp_chroma(chroma: np.ndarray, ratio: float) -> np.ndarray:
    """Simulate DJ vinyl-style playback (pitch + speed coupled) at given ratio.
    ratio > 1 → faster + higher-pitched.

    Time:   compress sequence length by 1/ratio (faster track → shorter chroma).
    Pitch:  roll rows by 12·log2(ratio) semitones (rounded to nearest bin).

    For small ratios (<3 %) the pitch shift is sub-bin so only the time warp
    matters; for larger shifts (e.g. +7 %) the row roll matters too.
    """
    n_orig = chroma.shape[1]
    n_new  = max(1, int(round(n_orig / ratio)))

    if n_new == n_orig:
        warped = chroma.copy()
    else:
        old_idx = np.arange(n_orig)
        new_idx = np.linspace(0, n_orig - 1, n_new)
        warped = np.empty((12, n_new), dtype=chroma.dtype)
        for c in range(12):
            warped[c] = np.interp(new_idx, old_idx, chroma[c])
        # Re-normalize after interpolation
        norms = np.linalg.norm(warped, axis=0, keepdims=True) + 1e-8
        warped = warped / norms

    semis = int(round(12 * np.log2(ratio)))
    if semis != 0:
        warped = np.roll(warped, semis, axis=0)

    return warped


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-correlation search
# ═══════════════════════════════════════════════════════════════════════════════

def cross_correlate(set_chroma: np.ndarray, ref_chroma: np.ndarray) -> tuple:
    """Sum-of-pitch-classes cross-correlation between set and ref chroma.

    Returns (score, score_offsets) where score[i] is the correlation when
    ref is placed starting at frame score_offsets[i] in the set.
    """
    n_set = set_chroma.shape[1]
    n_ref = ref_chroma.shape[1]
    # FFT-based correlation per pitch class, summed
    score = np.zeros(n_set + n_ref - 1, dtype=np.float64)
    for c in range(12):
        score += fftconvolve(set_chroma[c], ref_chroma[c, ::-1], mode="full")
    score_offsets = np.arange(score.size) - (n_ref - 1)

    # Average per overlapping frame so longer overlaps don't dominate.
    overlap = np.minimum(score_offsets + n_ref, n_set) - np.maximum(score_offsets, 0)
    overlap = np.clip(overlap, 1, n_ref).astype(np.float64)
    score = score / overlap

    return score, score_offsets


def search_with_tempo_ratios(
    set_chroma:         np.ndarray,
    ref_chroma_orig:    np.ndarray,
    ratios:             list,
    search_start_frame: int,
    search_end_frame:   int,
) -> dict:
    """Search for ref over the set across multiple tempo ratios.

    Returns a dict with the best (highest-confidence) result, including the
    full score curve at the chosen ratio (used later for the heatmap).
    """
    n_set = set_chroma.shape[1]
    s0    = max(0, search_start_frame)
    s1    = min(n_set - 1, search_end_frame)

    best = None
    for ratio in ratios:
        ref_chroma = warp_chroma(ref_chroma_orig, ratio)
        score, score_offsets = cross_correlate(set_chroma, ref_chroma)

        mask = (score_offsets >= s0) & (score_offsets <= s1)
        if not mask.any():
            continue
        windowed = np.where(mask, score, -np.inf)
        best_idx = int(np.argmax(windowed))
        confidence = float(score[best_idx])
        offset = int(score_offsets[best_idx])

        if best is None or confidence > best["confidence"]:
            best = {
                "offset":        offset,
                "confidence":    confidence,
                "ratio":         ratio,
                "score":         score,
                "score_offsets": score_offsets,
                "ref_n_frames":  ref_chroma.shape[1],
            }
    return best


def detect_played_region(
    set_chroma: np.ndarray,
    ref_chroma: np.ndarray,
    offset:     int,
    threshold:  float = PLAYED_THRESHOLD,
) -> tuple:
    """Walk along the alignment, flag frames where similarity > threshold,
    and return (audible_start_frame_in_set, audible_end_frame_in_set,
    mean_similarity_over_played_region).

    The longest run of high-similarity frames is treated as the play period;
    isolated spikes (likely from transitions where chroma briefly aligns with
    a different track) are ignored.  Falls back to the alignment endpoints
    when no run is found.
    """
    n_ref = ref_chroma.shape[1]
    n_set = set_chroma.shape[1]

    sims = np.zeros(n_ref, dtype=np.float32)
    for t in range(n_ref):
        s = offset + t
        if 0 <= s < n_set:
            sims[t] = float(np.dot(set_chroma[:, s], ref_chroma[:, t]))

    # Smooth so brief drops/spikes don't end runs prematurely
    win = max(11, n_ref // 80)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / win
    smooth = np.convolve(sims, kernel, mode="same")

    above = smooth > threshold
    if not above.any():
        return offset, offset + n_ref, float(smooth.mean())

    # Find longest contiguous run
    diffs  = np.diff(above.astype(np.int8))
    starts = list(np.where(diffs ==  1)[0] + 1)
    ends   = list(np.where(diffs == -1)[0] + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [n_ref]
    runs = list(zip(starts, ends))
    longest = max(runs, key=lambda x: x[1] - x[0])

    audible_start = offset + longest[0]
    audible_end   = offset + longest[1]
    mean_sim = float(smooth[longest[0]:longest[1]].mean())
    return audible_start, audible_end, mean_sim


def build_tempo_ratios(max_pct: float, step_pct: float = TEMPO_STEP_PCT) -> list:
    """Build a list of speed ratios spanning ±max_pct in step_pct increments."""
    n_steps = max(1, int(round(max_pct / step_pct)))
    pcts = np.unique(np.concatenate([
        [0.0],
        np.arange(step_pct, max_pct + step_pct / 2, step_pct),
        -np.arange(step_pct, max_pct + step_pct / 2, step_pct),
    ]))
    return [1.0 + p / 100.0 for p in sorted(pcts)]


# ═══════════════════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_time(secs: float) -> str:
    """Format seconds as M:SS.s or H:MM:SS.s."""
    if secs is None:
        return "    ?    "
    secs = max(0.0, secs)
    h = int(secs) // 3600
    m = (int(secs) // 60) % 60
    s = secs - 60 * (h * 60 + m)
    if h:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"


def write_tracklist(tracks: list, out_path: str):
    """Write the human-readable tracklist."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Tracklist\n")
        f.write(f"# Source : {os.path.basename(PLAYLIST_CSV)}\n")
        f.write(f"# Set    : {os.path.basename(SET_WAV)}\n")
        f.write("#\n")
        f.write("# Time      Conf   ±%       #  Title — Artist\n")
        f.write("# ───────   ────   ─────   ──  ─────────────────────────────────────────\n")
        for t in tracks:
            if t.is_mashup:
                f.write(f"# (mashup)                    {t.number:>2}  {t.title} — {t.artist}\n")
                continue
            if t.is_missing:
                f.write(f"# (missing file)              {t.number:>2}  {t.title} — {t.artist}\n")
                continue
            ts        = fmt_time(t.detected_start_secs)
            conf      = f"{t.confidence:.2f}"
            bpm_shift = (t.tempo_ratio - 1.0) * 100
            flag      = "  " if t.confidence >= MIN_CONFIDENCE else "??"
            f.write(
                f"  {ts:<9} {conf:<6} {bpm_shift:+5.1f}%  {t.number:>2}  "
                f"{t.title} — {t.artist} {flag}\n"
            )
    print(f"  Tracklist     : {out_path}")


def write_heatmap(tracks: list, set_duration: float, out_path: str):
    """Render a confidence heatmap: x = set time, y = track index, color = score.

    Each row shows the cross-correlation score curve for that track at its
    chosen tempo ratio.  A red marker on each row marks the detected start.
    Tracks with poor matches show low colour everywhere — easy to eyeball.
    """
    visible = [t for t in tracks if not t.is_mashup and not t.is_missing
               and t.score_curve is not None]
    if not visible:
        print("  (no tracks to plot)")
        return

    # Resample every track's score curve to a common x-axis (set time)
    n_x = min(4000, int(set_duration * 4))   # ~4 px / sec, cap at 4k
    x_secs = np.linspace(0, set_duration, n_x)
    grid = np.full((len(visible), n_x), np.nan, dtype=np.float32)

    for i, t in enumerate(visible):
        x_track = t.score_offsets_secs
        y_track = t.score_curve
        # Clip to valid set range
        mask = (x_track >= 0) & (x_track <= set_duration)
        if mask.sum() < 2:
            continue
        grid[i] = np.interp(x_secs, x_track[mask], y_track[mask],
                            left=np.nan, right=np.nan)

    fig_height = max(4, 0.32 * len(visible) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="upper",
        extent=[0, set_duration / 60, len(visible) - 0.5, -0.5],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, label="cross-correlation (cosine similarity)")

    # Detected-start markers
    for i, t in enumerate(visible):
        if t.detected_start_secs is None:
            continue
        marker_color = "lime" if t.confidence >= MIN_CONFIDENCE else "red"
        ax.plot(t.detected_start_secs / 60, i, marker="o",
                markerfacecolor="none", markeredgecolor=marker_color,
                markeredgewidth=1.4, markersize=8, zorder=5)

    ax.set_yticks(range(len(visible)))
    ax.set_yticklabels([f"{t.number:>2}  {t.title[:38]}" for t in visible], fontsize=8)
    ax.set_xlabel("Set time (minutes)")
    ax.set_title("Tracklist confidence heatmap — green ◯ = locked, red ◯ = low confidence")
    ax.grid(axis="x", color="white", alpha=0.15, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Heatmap       : {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not os.path.isfile(PLAYLIST_CSV):
        sys.exit(f"Error: Playlist CSV not found:\n  {PLAYLIST_CSV}")
    if not os.path.isfile(SET_WAV):
        sys.exit(f"Error: Set WAV not found:\n  {SET_WAV}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading playlist: {PLAYLIST_CSV}")
    tracks = read_playlist(PLAYLIST_CSV)
    print(f"  {len(tracks)} tracks "
          f"({sum(t.is_mashup for t in tracks)} marked as mashup)")

    # Validate file paths
    missing = []
    for t in tracks:
        if not t.is_mashup and not os.path.isfile(t.file_path):
            t.is_missing = True
            missing.append(t)
    if missing:
        print(f"\n  Warning: {len(missing)} source files are missing:")
        for t in missing:
            print(f"    Track {t.number}: {t.file_path}")

    # Load full set
    print(f"\nLoading set audio: {SET_WAV}")
    print(f"  (downsampling to {ANALYSIS_SR} Hz mono)")
    set_audio = load_audio_mono(SET_WAV, sr=ANALYSIS_SR)
    set_duration = len(set_audio) / ANALYSIS_SR
    print(f"  Duration: {set_duration:.1f} s ({set_duration/60:.1f} min)")

    print("Computing set chroma …")
    set_chroma = compute_chroma(set_audio, sr=ANALYSIS_SR, hop=HOP_LENGTH)
    print(f"  {set_chroma.shape[1]} frames @ {ANALYSIS_SR/HOP_LENGTH:.1f} Hz")
    del set_audio   # free memory; we only need chroma from here on

    frames_per_sec = ANALYSIS_SR / HOP_LENGTH

    # ── Sequential search ───────────────────────────────────────────────────
    last_start_secs = 0.0
    print()
    for t in tracks:
        header = f"Track {t.number:>2}: {t.title}"
        if t.is_mashup:
            print(f"{header}  [MASHUP — skipped]")
            continue
        if t.is_missing:
            print(f"{header}  [MISSING file — skipped]")
            continue

        print(f"{header}")

        # Build search window: previous start − lookback ... + lookahead.
        # First track: only constrain to [0, lookahead].
        if t.number == tracks[0].number:
            s0_secs = 0.0
        else:
            s0_secs = max(0.0, last_start_secs - SEARCH_LOOKBACK)
        s1_secs = min(set_duration, last_start_secs + SEARCH_LOOKAHEAD)
        s0_frame = int(s0_secs * frames_per_sec)
        s1_frame = int(s1_secs * frames_per_sec)

        # Load reference and compute chroma once
        try:
            ref_audio = load_audio_mono(t.file_path)
        except Exception as exc:
            print(f"  ERROR loading audio: {exc}")
            continue
        ref_chroma_orig = compute_chroma(ref_audio, sr=ANALYSIS_SR, hop=HOP_LENGTH)
        del ref_audio

        # Search across tempo ratios
        max_pct = TEMPO_OVERRIDES.get(t.number, DEFAULT_TEMPO_PCT)
        ratios  = build_tempo_ratios(max_pct)
        best    = search_with_tempo_ratios(
            set_chroma, ref_chroma_orig, ratios, s0_frame, s1_frame
        )

        # Retry with wide range if confidence too low
        if (best is None or best["confidence"] < MIN_CONFIDENCE) \
                and t.number not in TEMPO_OVERRIDES:
            prev_conf = f"{best['confidence']:.2f}" if best else "n/a"
            print(f"  Low confidence ({prev_conf}), "
                  f"retrying with ±{WIDE_TEMPO_PCT}% tempo …")
            ratios = build_tempo_ratios(WIDE_TEMPO_PCT)
            best   = search_with_tempo_ratios(
                set_chroma, ref_chroma_orig, ratios, s0_frame, s1_frame
            )

        if best is None:
            print("  ERROR: search returned no result")
            continue

        # Refine to audible region (ignore intro silence / pre-hotcue audio)
        ref_chroma_at_best = warp_chroma(ref_chroma_orig, best["ratio"])
        audible_start_f, audible_end_f, played_sim = detect_played_region(
            set_chroma, ref_chroma_at_best, best["offset"]
        )

        t.detected_start_secs = audible_start_f / frames_per_sec
        t.detected_end_secs   = audible_end_f   / frames_per_sec
        t.confidence          = max(best["confidence"], played_sim)
        t.tempo_ratio         = best["ratio"]
        t.score_curve         = best["score"].astype(np.float32)
        t.score_offsets_secs  = best["score_offsets"] / frames_per_sec

        bpm_shift = (best["ratio"] - 1.0) * 100
        flag      = "✓" if t.confidence >= MIN_CONFIDENCE else "?"
        print(f"  {flag} Start: {fmt_time(t.detected_start_secs)}   "
              f"End: {fmt_time(t.detected_end_secs)}   "
              f"conf={t.confidence:.3f}   bpm_shift={bpm_shift:+.1f}%")

        last_start_secs = t.detected_start_secs

    # ── Outputs ─────────────────────────────────────────────────────────────
    print("\nWriting outputs …")
    write_tracklist(tracks, OUTPUT_TRACKLIST)
    write_heatmap(tracks, set_duration, OUTPUT_HEATMAP)


if __name__ == "__main__":
    main()
