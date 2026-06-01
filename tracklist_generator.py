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
        "Install with: pip install librosa soundfile numpy scipy matplotlib mutagen plotly"
    )

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these variables before running
# ═══════════════════════════════════════════════════════════════════════════════

# ── Inputs ────────────────────────────────────────────────────────────────────
PLAYLIST_CSV = os.path.expanduser(
    "~/Git Repos/tracklist-from-full-set/Party Set V4.csv"
)
SET_WAV = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal/GREE/My Sets/Abiball Party Set V4/Party Set V4.wav"
)
ENGINE_DB_PATH = os.path.expanduser("~/Music/Engine Library/Database2/m.db")

# ── Outputs ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-Personal/GREE/My Sets/Abiball Party Set V4"
)
OUTPUT_TRACKLIST = os.path.join(OUTPUT_DIR, "tracklist.txt")
OUTPUT_HEATMAP   = os.path.join(OUTPUT_DIR, "confidence_heatmap.png")
OUTPUT_INTERACTIVE = os.path.join(OUTPUT_DIR, "transitions.html")

# ── Mashup overrides ──────────────────────────────────────────────────────────
# Track numbers (from the CSV "#" column) whose VOCALS only are layered over
# another track's backing.  These tracks are skipped — they don't appear in
# the tracklist and don't advance the timeline.
MASHUP_TRACK_NUMBERS: list = [10,11,30]     # e.g. [12, 18]

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
MIN_RUN_SECS       = 8.0       # ignore high-similarity bursts shorter than this
                               # (filters brief chord-match spikes during transitions)
SEARCH_LOOKBACK    = 20        # search this many seconds before previous track's start
SEARCH_LOOKAHEAD   = 360       # search up to this many seconds after previous start

# ── Output settings ───────────────────────────────────────────────────────────
TRANSITION_WINDOW_SECS = 60.0   # seconds before/after each detected start to show
                                # in the interactive transition plot

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
    # Per-frame similarity at the locked alignment — high while the song is
    # audible, low otherwise.  This is the curve used in the interactive plot
    # and the heatmap; gives much better contrast than the cross-correlation.
    played_curve:         Optional[np.ndarray] = None
    played_times_secs:    Optional[np.ndarray] = None


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


def compute_played_curve(
    set_chroma: np.ndarray,
    ref_chroma: np.ndarray,
    offset:     int,
) -> np.ndarray:
    """Per-frame cosine similarity between set and ref at the given alignment.

    Returns an array indexed by *set frame*: played[s] = sim(set[s], ref[s-offset])
    when the ref overlaps that set frame, else 0.  This is the curve that
    cleanly separates "song audible" from "song silent" — much higher contrast
    than the windowed cross-correlation used during search.
    """
    n_set = set_chroma.shape[1]
    n_ref = ref_chroma.shape[1]
    played = np.zeros(n_set, dtype=np.float32)
    s0 = max(0, offset)
    s1 = min(n_set, offset + n_ref)
    if s1 > s0:
        # Per-frame dot product (chroma is L2-normalized, so this is cosine sim).
        ref_slice = ref_chroma[:, s0 - offset : s1 - offset]
        played[s0:s1] = np.einsum("ij,ij->j", set_chroma[:, s0:s1], ref_slice)
    return played


def detect_played_region(
    played_curve:    np.ndarray,
    sample_rate_fps: float,
    threshold:       float = PLAYED_THRESHOLD,
    min_run_secs:    float = MIN_RUN_SECS,
    seed_frame:      Optional[int] = None,
) -> tuple:
    """From the per-frame played-curve, find the run that represents the song.

    A "run" is a contiguous span of frames whose smoothed similarity exceeds
    `threshold`.  We discard runs shorter than `min_run_secs` (these are brief
    chord matches during transitions, not real playback).  Of the runs that
    remain, we pick the one closest to `seed_frame` (the cross-correlation
    peak) — that is almost always the correct one and avoids the bug where
    the song's outro overlap with the *next* track gets picked as the start.

    Returns (start_frame, end_frame, mean_similarity).
    """
    n = len(played_curve)
    win = max(11, int(round(2.0 * sample_rate_fps)))   # ~2 s smoothing
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / win
    smooth = np.convolve(played_curve, kernel, mode="same")

    above = smooth > threshold
    if not above.any():
        # Nothing reached threshold — fall back to whatever we had
        idx = int(np.argmax(smooth)) if seed_frame is None else seed_frame
        return idx, idx + 1, float(smooth.max())

    diffs  = np.diff(above.astype(np.int8))
    starts = list(np.where(diffs ==  1)[0] + 1)
    ends   = list(np.where(diffs == -1)[0] + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [n]
    runs = list(zip(starts, ends))

    min_run_frames = int(round(min_run_secs * sample_rate_fps))
    long_runs = [(s, e) for s, e in runs if (e - s) >= min_run_frames]
    if not long_runs:
        # Nothing sustained — relax min length but still prefer longest
        long_runs = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:3]

    if seed_frame is not None:
        # Pick the run that contains the seed; if none does, the closest one.
        containing = [r for r in long_runs if r[0] <= seed_frame < r[1]]
        if containing:
            chosen = max(containing, key=lambda r: r[1] - r[0])
        else:
            chosen = min(long_runs,
                         key=lambda r: min(abs(seed_frame - r[0]),
                                           abs(seed_frame - r[1])))
    else:
        chosen = max(long_runs, key=lambda r: r[1] - r[0])

    s, e = chosen
    return s, e, float(smooth[s:e].mean())


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

    Each row shows the *per-frame similarity* between the set and that track at
    its locked alignment — high while the song is audible, low otherwise.
    Much higher contrast than the raw cross-correlation curve, so the
    "song is/isn't playing" boundary is visible at a glance.

    A green ◯ marks each detected start (red ◯ if confidence is low).
    """
    visible = [t for t in tracks if not t.is_mashup and not t.is_missing
               and t.played_curve is not None]
    if not visible:
        print("  (no tracks to plot)")
        return

    # Resample every track's played curve to a common x-axis (set time)
    n_x = min(4000, int(set_duration * 4))   # ~4 px / sec, cap at 4k
    x_secs = np.linspace(0, set_duration, n_x)
    grid = np.full((len(visible), n_x), np.nan, dtype=np.float32)

    for i, t in enumerate(visible):
        x_track = t.played_times_secs
        y_track = t.played_curve
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
    fig.colorbar(im, ax=ax, label="per-frame similarity (cosine)")

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
    ax.set_title("Per-track playback heatmap — green ◯ = locked, red ◯ = low confidence")
    ax.grid(axis="x", color="white", alpha=0.15, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Heatmap       : {out_path}")


def write_interactive_transitions(tracks: list, set_duration: float, out_path: str):
    """Render an interactive HTML plot focused on transition zones.

    For each transition (track N → N+1), shows the played-curves of both
    tracks zoomed into a ±TRANSITION_WINDOW_SECS window around the detected
    start of track N+1.  Plotly handles zoom/pan/hover in the browser.

    Use this when a detected start looks slightly off — you can see exactly
    where each track's audio ramps up/down and pick the right boundary by eye.
    """
    if not _HAS_PLOTLY:
        print("  (plotly not installed — skipping interactive plot. "
              "pip install plotly)")
        return

    visible = [t for t in tracks if not t.is_mashup and not t.is_missing
               and t.played_curve is not None
               and t.detected_start_secs is not None]
    if len(visible) < 2:
        print("  (not enough tracks for interactive transitions)")
        return

    n = len(visible)
    cols = 2
    rows = (n + cols - 1) // cols

    titles = []
    for i, t in enumerate(visible):
        prev_label = f"← {visible[i-1].number}" if i > 0 else "—"
        titles.append(
            f"Transition into #{t.number}: {t.title[:42]}  "
            f"(prev: {prev_label}) @ {fmt_time(t.detected_start_secs)}"
        )

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.04,
        horizontal_spacing=0.06,
    )

    for i, t in enumerate(visible):
        row = i // cols + 1
        col = i % cols + 1

        center = t.detected_start_secs
        x0 = max(0.0, center - TRANSITION_WINDOW_SECS)
        x1 = min(set_duration, center + TRANSITION_WINDOW_SECS)

        # Build the slice for this track
        mask = (t.played_times_secs >= x0) & (t.played_times_secs <= x1)
        fig.add_trace(
            go.Scatter(
                x=t.played_times_secs[mask],
                y=t.played_curve[mask],
                mode="lines",
                name=f"#{t.number} {t.title[:30]}",
                line=dict(width=2, color="#1f77b4"),
                hovertemplate=(
                    "set time=%{x:.2f}s<br>"
                    f"#{t.number} {t.title[:40]}<br>"
                    "similarity=%{y:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row, col=col,
        )

        # Previous track (the one ending around this transition)
        if i > 0:
            prev = visible[i - 1]
            mask_prev = (prev.played_times_secs >= x0) & (prev.played_times_secs <= x1)
            fig.add_trace(
                go.Scatter(
                    x=prev.played_times_secs[mask_prev],
                    y=prev.played_curve[mask_prev],
                    mode="lines",
                    name=f"#{prev.number} {prev.title[:30]}",
                    line=dict(width=2, color="#ff7f0e", dash="dot"),
                    hovertemplate=(
                        "set time=%{x:.2f}s<br>"
                        f"#{prev.number} {prev.title[:40]} (outgoing)<br>"
                        "similarity=%{y:.3f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row, col=col,
            )

        # Detected-start vertical line
        fig.add_shape(
            type="line",
            x0=center, x1=center, y0=0, y1=1,
            xref=f"x{i+1}" if i > 0 else "x",
            yref=f"y{i+1}" if i > 0 else "y",
            line=dict(color="lime" if t.confidence >= MIN_CONFIDENCE else "red",
                      width=1.5, dash="dash"),
        )

        # Threshold line
        fig.add_shape(
            type="line",
            x0=x0, x1=x1, y0=PLAYED_THRESHOLD, y1=PLAYED_THRESHOLD,
            xref=f"x{i+1}" if i > 0 else "x",
            yref=f"y{i+1}" if i > 0 else "y",
            line=dict(color="gray", width=1, dash="dot"),
        )

        fig.update_xaxes(
            title_text="set time (s)" if row == rows else "",
            range=[x0, x1],
            row=row, col=col,
        )
        fig.update_yaxes(range=[0, 1], row=row, col=col)

    fig.update_layout(
        title=("Transition explorer — incoming track (blue solid) and "
               "outgoing track (orange dotted) similarity over set time. "
               "Dashed vertical = detected start. Hover for exact times."),
        height=max(400, rows * 240),
        plot_bgcolor="white",
        font=dict(size=11),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 10

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  Transitions   : {out_path}")


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

        # Per-frame similarity at the locked alignment — high during playback,
        # low otherwise.  This is what gives the heatmap real contrast.
        played_curve = compute_played_curve(
            set_chroma, ref_chroma_at_best, best["offset"]
        )
        seed_frame = best["offset"] + ref_chroma_at_best.shape[1] // 4
        audible_start_f, audible_end_f, played_sim = detect_played_region(
            played_curve, frames_per_sec, seed_frame=seed_frame
        )

        t.detected_start_secs = audible_start_f / frames_per_sec
        t.detected_end_secs   = audible_end_f   / frames_per_sec
        t.confidence          = played_sim
        t.tempo_ratio         = best["ratio"]
        t.score_curve         = best["score"].astype(np.float32)
        t.score_offsets_secs  = best["score_offsets"] / frames_per_sec
        t.played_curve        = played_curve
        t.played_times_secs   = np.arange(len(played_curve)) / frames_per_sec

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
    write_interactive_transitions(tracks, set_duration, OUTPUT_INTERACTIVE)


if __name__ == "__main__":
    main()
