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
OUTPUT_TRACKLIST   = os.path.join(OUTPUT_DIR, "tracklist.csv")
OUTPUT_HEATMAP     = os.path.join(OUTPUT_DIR, "confidence_heatmap.png")
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
PLAYED_THRESHOLD   = 0.55      # high threshold: "song is clearly the lead track"
ONSET_THRESHOLD    = 0.35      # low threshold: "song is becoming/ceasing audible"
                               # used for sub-second-precise transition edges
MIN_RUN_SECS       = 8.0       # ignore high-similarity bursts shorter than this
                               # (filters brief chord-match spikes during transitions)
EDGE_REFINE_SECS   = 6.0       # max distance to walk inward from the broad-detection
                               # edge looking for the precise threshold crossing
COARSE_SMOOTH_SECS = 2.0       # smoothing window for finding the broad audible region
FINE_SMOOTH_SECS   = 0.25      # smoothing window for sub-second edge refinement
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

    # Four transition timestamps, in set-time seconds.
    #   in_start  — first audible frame (start of the in-transition / mix-in)
    #   in_end    — moment the previous track stops being audible (in-transition done)
    #   out_start — moment the next track starts becoming audible (out-transition begins)
    #   out_end   — last audible frame of this track (out-transition done)
    # For the first track in_start may equal in_end (no incoming transition);
    # for the last track out_start may equal out_end (no outgoing transition).
    in_start_secs:   Optional[float] = None
    in_end_secs:     Optional[float] = None
    out_start_secs:  Optional[float] = None
    out_end_secs:    Optional[float] = None


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


def _smooth(curve: np.ndarray, win_frames: int) -> np.ndarray:
    """Boxcar-smooth a 1-D curve.  win_frames is forced odd."""
    win = max(3, int(win_frames))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(curve.astype(np.float32), kernel, mode="same")


def _refine_edge(
    curve:       np.ndarray,
    coarse_idx:  int,
    direction:   int,                 # +1 = walking forward, -1 = walking back
    threshold:   float,
    max_walk:    int,
) -> int:
    """Walk inward from a coarse edge index toward the lower-threshold crossing.

    Sub-second refinement: the coarse smoothed curve's threshold crossing is
    biased by the smoothing window, so we walk from `coarse_idx` along
    `direction` looking for the first frame where the *lightly* smoothed
    curve drops below `threshold`.  The frame just before the drop (in the
    walk direction) is the precise boundary.

    direction = +1 walks forward (used for an audible-region START — we walk
    forward from the run's start until we pass the noise-floor crossing).
    direction = -1 walks backward (audible-region END — we walk backward from
    the run's last frame until we pass the noise-floor crossing).
    """
    n = len(curve)
    walk = 0
    idx = coarse_idx
    if direction == +1:
        # Walk backward from coarse_idx while the curve is still above threshold,
        # then walk forward to the first frame below.  This finds the rising edge.
        while idx - 1 >= 0 and walk < max_walk and curve[idx - 1] >= threshold:
            idx -= 1
            walk += 1
        return max(0, idx)
    else:
        while idx + 1 < n and walk < max_walk and curve[idx + 1] >= threshold:
            idx += 1
            walk += 1
        return min(n - 1, idx)


def detect_played_region(
    played_curve:    np.ndarray,
    sample_rate_fps: float,
    threshold:       float = PLAYED_THRESHOLD,
    onset_threshold: float = ONSET_THRESHOLD,
    min_run_secs:    float = MIN_RUN_SECS,
    seed_frame:      Optional[int] = None,
) -> tuple:
    """From the per-frame played-curve, find the run that represents the song,
    then refine its start/end edges to sub-second precision.

    Two-pass detection:
      1. COARSE — heavy smoothing (~2 s window) + high `threshold` (≥ 0.55) to
         identify the broad audible region.  Filters short transition spikes.
      2. FINE   — light smoothing (~0.25 s) + low `onset_threshold` (≥ 0.35)
         to walk the broad edges out to where the song first becomes / last
         remains audible above the noise floor.  The fine threshold catches
         the actual onset of the song mixing in, not its mid-song peak.

    Returns (start_frame, end_frame, mean_similarity).  start_frame and
    end_frame are in original-curve indices, with frame-level (≈46 ms) precision.
    """
    n = len(played_curve)

    coarse = _smooth(played_curve, int(COARSE_SMOOTH_SECS * sample_rate_fps))
    fine   = _smooth(played_curve, int(FINE_SMOOTH_SECS   * sample_rate_fps))

    above = coarse > threshold
    if not above.any():
        idx = int(np.argmax(coarse)) if seed_frame is None else seed_frame
        return idx, idx + 1, float(coarse.max())

    # Identify all coarse runs
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
        long_runs = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:3]

    if seed_frame is not None:
        containing = [r for r in long_runs if r[0] <= seed_frame < r[1]]
        if containing:
            chosen = max(containing, key=lambda r: r[1] - r[0])
        else:
            chosen = min(long_runs,
                         key=lambda r: min(abs(seed_frame - r[0]),
                                           abs(seed_frame - r[1])))
    else:
        chosen = max(long_runs, key=lambda r: r[1] - r[0])

    coarse_s, coarse_e = chosen
    max_walk_frames = int(round(EDGE_REFINE_SECS * sample_rate_fps))

    # Refine: walk backward from coarse_s while the *fine* curve is still
    # above the noise-floor onset_threshold.  This pulls the start earlier
    # if the song was audibly present (above onset_threshold) before the
    # coarse run began — i.e. it captures the rising edge of the mix-in.
    fine_s = coarse_s
    walk = 0
    while fine_s - 1 >= 0 and walk < max_walk_frames and fine[fine_s - 1] >= onset_threshold:
        fine_s -= 1
        walk += 1

    fine_e = coarse_e
    walk = 0
    while fine_e < n and walk < max_walk_frames and fine[fine_e] >= onset_threshold:
        fine_e += 1
        walk += 1

    return fine_s, fine_e, float(coarse[coarse_s:coarse_e].mean())


def resolve_transitions(tracks: list, frames_per_sec: float) -> None:
    """Compute the four transition timestamps for every track.

    For each consecutive pair (N, N+1) we walk both played-curves between
    track N's coarse-detected end and track N+1's coarse-detected start, and
    locate the moment N+1 first becomes audible above ONSET_THRESHOLD
    (= mix_in) and the moment N falls below ONSET_THRESHOLD (= mix_out).

    The resulting timestamps:
        track N+1.in_start  = track N+1.out_start (of N→N+1 transition) = mix_in
        track N+1.in_end    = track N.out_end                            = mix_out
    so the transition window is [mix_in, mix_out] from both perspectives.

    Mashup tracks are skipped entirely (they don't advance the timeline).
    """
    detectable = [t for t in tracks
                  if not t.is_mashup and not t.is_missing
                  and t.played_curve is not None]

    for i, t in enumerate(detectable):
        # Default: use the detected audible region itself
        t.in_start_secs  = t.detected_start_secs
        t.in_end_secs    = t.detected_start_secs   # placeholder, refined below
        t.out_start_secs = t.detected_end_secs     # placeholder
        t.out_end_secs   = t.detected_end_secs

    for i in range(len(detectable) - 1):
        cur = detectable[i]
        nxt = detectable[i + 1]

        # Search window: from current track's detected start
        # to next track's detected end (covers the whole overlap zone)
        s0 = max(0, int(cur.detected_start_secs * frames_per_sec))
        s1 = min(len(cur.played_curve),
                 int(nxt.detected_end_secs * frames_per_sec))
        if s1 <= s0:
            continue

        cur_fine = _smooth(cur.played_curve, int(FINE_SMOOTH_SECS * frames_per_sec))
        nxt_fine = _smooth(nxt.played_curve, int(FINE_SMOOTH_SECS * frames_per_sec))

        # mix_in: first frame in [cur.detected_start, nxt.detected_end]
        # where the next track rises above ONSET_THRESHOLD.
        nxt_window = nxt_fine[s0:s1]
        above_nxt  = np.where(nxt_window >= ONSET_THRESHOLD)[0]
        if len(above_nxt):
            mix_in_frame = s0 + int(above_nxt[0])
        else:
            mix_in_frame = int(nxt.detected_start_secs * frames_per_sec)

        # mix_out: last frame from mix_in onward where the current track is
        # still above ONSET_THRESHOLD.  The frame after that is when the
        # outgoing track is fully gone.
        cur_after_mixin = cur_fine[mix_in_frame:s1]
        above_cur = np.where(cur_after_mixin >= ONSET_THRESHOLD)[0]
        if len(above_cur):
            mix_out_frame = mix_in_frame + int(above_cur[-1]) + 1
        else:
            mix_out_frame = mix_in_frame + 1

        # Sanity: mix_out cannot precede mix_in
        if mix_out_frame <= mix_in_frame:
            mix_out_frame = mix_in_frame + 1

        mix_in_secs  = mix_in_frame  / frames_per_sec
        mix_out_secs = mix_out_frame / frames_per_sec

        # Assign to both tracks — same physical time points.
        # For the OUTGOING track this transition is its OUT-transition.
        cur.out_start_secs = mix_in_secs
        cur.out_end_secs   = mix_out_secs
        # For the INCOMING track this transition is its IN-transition.
        nxt.in_start_secs  = mix_in_secs
        nxt.in_end_secs    = mix_out_secs


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
    """Write the tracklist as a CSV.

    Columns:
        # — playlist position from the source CSV
        title, artist
        status — "ok" (mapped), "low_confidence", "mashup", "missing"
        in_start, in_end       — incoming transition window (set time)
        out_start, out_end     — outgoing transition window (set time)
        in_start_hms, in_end_hms, out_start_hms, out_end_hms — same, formatted
        confidence             — locked-region mean similarity
        bpm_shift_pct          — playback-speed shift (%, vs original)

    All timestamps are set-time seconds (decimals, ms precision).
    Mashup and missing tracks have empty timestamp cells.
    """
    fieldnames = [
        "#", "title", "artist", "status",
        "in_start", "in_end", "out_start", "out_end",
        "in_start_hms", "in_end_hms", "out_start_hms", "out_end_hms",
        "confidence", "bpm_shift_pct",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in tracks:
            if t.is_mashup:
                status = "mashup"
            elif t.is_missing:
                status = "missing"
            elif t.confidence < MIN_CONFIDENCE:
                status = "low_confidence"
            else:
                status = "ok"

            row = {
                "#":      t.number,
                "title":  t.title,
                "artist": t.artist,
                "status": status,
            }
            if status in ("ok", "low_confidence"):
                row.update({
                    "in_start":      f"{t.in_start_secs:.3f}"  if t.in_start_secs  is not None else "",
                    "in_end":        f"{t.in_end_secs:.3f}"    if t.in_end_secs    is not None else "",
                    "out_start":     f"{t.out_start_secs:.3f}" if t.out_start_secs is not None else "",
                    "out_end":       f"{t.out_end_secs:.3f}"   if t.out_end_secs   is not None else "",
                    "in_start_hms":  fmt_time(t.in_start_secs),
                    "in_end_hms":    fmt_time(t.in_end_secs),
                    "out_start_hms": fmt_time(t.out_start_secs),
                    "out_end_hms":   fmt_time(t.out_end_secs),
                    "confidence":    f"{t.confidence:.3f}",
                    "bpm_shift_pct": f"{(t.tempo_ratio - 1.0) * 100:+.2f}",
                })
            else:
                for k in fieldnames[4:]:
                    row[k] = ""
            w.writerow(row)
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

        # Vertical markers — incoming track's transition windows.
        # in_start / in_end mark when the *incoming* track first becomes
        # audible and when the *outgoing* track stops being audible.
        marker_color_lock = "lime" if t.confidence >= MIN_CONFIDENCE else "red"
        for x_val, color, dash, label in [
            (t.in_start_secs,  "lime",   "dash", "in_start (mix-in begins)"),
            (t.in_end_secs,    "orange", "dash", "in_end (prev faded out)"),
        ]:
            if x_val is None or not (x0 <= x_val <= x1):
                continue
            fig.add_shape(
                type="line",
                x0=x_val, x1=x_val, y0=0, y1=1,
                xref=f"x{i+1}" if i > 0 else "x",
                yref=f"y{i+1}" if i > 0 else "y",
                line=dict(color=color, width=1.5, dash=dash),
            )

        # Threshold lines (high = lock, low = onset)
        for thr, color in [(PLAYED_THRESHOLD, "gray"),
                           (ONSET_THRESHOLD,  "lightgray")]:
            fig.add_shape(
                type="line",
                x0=x0, x1=x1, y0=thr, y1=thr,
                xref=f"x{i+1}" if i > 0 else "x",
                yref=f"y{i+1}" if i > 0 else "y",
                line=dict(color=color, width=1, dash="dot"),
            )

        fig.update_xaxes(
            title_text="set time (s)" if row == rows else "",
            range=[x0, x1],
            row=row, col=col,
        )
        fig.update_yaxes(range=[0, 1], row=row, col=col)

    fig.update_layout(
        title=("Transition explorer — incoming (blue solid) vs outgoing (orange dotted) "
               "track similarity. Green dashed = in_start (mix-in begins), "
               "orange dashed = in_end (previous track gone). "
               "Gray dotted = lock/onset thresholds."),
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

    # ── Resolve transition timestamps ───────────────────────────────────────
    print("\nResolving transition windows …")
    resolve_transitions(tracks, frames_per_sec)

    # ── Outputs ─────────────────────────────────────────────────────────────
    print("\nWriting outputs …")
    write_tracklist(tracks, OUTPUT_TRACKLIST)
    write_heatmap(tracks, set_duration, OUTPUT_HEATMAP)
    write_interactive_transitions(tracks, set_duration, OUTPUT_INTERACTIVE)


if __name__ == "__main__":
    main()
