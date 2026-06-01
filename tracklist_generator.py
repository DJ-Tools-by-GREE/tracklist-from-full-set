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
import json
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
OUTPUT_REVIEW_UI   = os.path.join(OUTPUT_DIR, "review.html")

# ── Mashup overrides ──────────────────────────────────────────────────────────
# Track numbers (from the CSV "#" column) whose VOCALS only are layered over
# another track's backing.  These tracks are skipped — they don't appear in
# the tracklist and don't advance the timeline.
MASHUP_TRACK_NUMBERS: list = [10,11,30]     # e.g. [12, 18]

# ── Tempo search ──────────────────────────────────────────────────────────────
# DJ pitch-bend can shift speed/pitch by a few percent.  The search runs in two
# stages: a coarse sweep at TEMPO_STEP_PCT, then a fine refinement around the
# winning ratio at REFINE_STEP_PCT for sub-step precision (~0.05 %).
# WIDE_TEMPO_PCT covers e.g. 121 → 130 BPM ≈ +7.4 %.
DEFAULT_TEMPO_PCT = 3.0     # ±%   coarse search range (default)
WIDE_TEMPO_PCT    = 10.0    # ±%   coarse search range (auto-retry on low confidence)
TEMPO_STEP_PCT    = 0.5     # %    coarse grid step
REFINE_STEP_PCT   = 0.05    # %    fine grid step around the coarse winner
REFINE_HALF_WIDTH = 0.6     # %    fine grid spans ±this around the coarse winner

# Per-track explicit override (CSV track number → max ±% to search).
# Use this to force a wide search for a transition you know is dramatic.
TEMPO_OVERRIDES: dict = {}  # e.g. {15: 12.0}

# Per-segment ratio drift refinement — after the global lock, the played region
# is split into chunks and each chunk re-aligned at sub-step resolution so that
# slow speed changes during a track (DJ rides the pitch fader) don't cause the
# played-curve to drift out of phase mid-track.
SEGMENT_REFINE_ENABLED   = True
SEGMENT_LENGTH_SECS      = 30.0   # length of each per-segment ratio probe
SEGMENT_RATIO_HALF_WIDTH = 1.0    # ±% search around the global ratio per segment
SEGMENT_RATIO_STEP       = 0.05   # % step inside each segment

# ── Transition BPM sync ───────────────────────────────────────────────────────
# When two tracks beatmatch through a transition, both decks play at the SAME
# BPM by definition (the DJ has synced them).  After resolve_transitions has
# fixed in/out timestamps, we re-fit the segments inside each transition
# window with a SHARED ratio per pair of tracks — and we let it drift linearly
# from the outgoing track's solo BPM at out_start to the incoming track's
# solo BPM at in_end (since DJs commonly ride the pitch fader during the mix).
#
# This:
#   • Removes spurious BPM noise that the per-track segment fitter produces
#     during the overlap (where the chroma is a mix of two tracks).
#   • Makes the played-curve cleaner through transitions (both refs are warped
#     at the actual played BPM rather than each at its solo BPM).
#   • Reveals the DJ's chosen transition BPM as a clean ramp on the plot.
TRANSITION_BPM_SYNC_ENABLED = True
SOLO_MARGIN_SECS  = 2.0    # ignore segments within this many sec of in_end /
                           # out_start when picking each track's "solo" ratio
                           # — those are still partly mixed.
TRANSITION_RAMP_STEP_PCT  = 0.05   # step size for ramp-target probe
TRANSITION_RAMP_HALF_PCT  = 1.5    # max ±% deviation of ramp endpoint from the
                                   # straight-line interpolation between
                                   # neighbour solo BPMs

# ── Cue alignment ────────────────────────────────────────────────────────────
# When comparing Engine DJ hotcue positions to detected transition markers,
# a cue is considered "aligned" if it is within this many seconds of the marker.
CUE_ALIGN_THRESHOLD_SECS = 3.0

# ── Engine DJ sample rate ─────────────────────────────────────────────────────
# Hotcue positions are stored in samples at this rate.
ENGINE_SAMPLE_RATE = 44100

# ── Hard-cut detection ────────────────────────────────────────────────────────
# When the outgoing track has clearly stopped before the incoming one starts
# (no overlap), we treat it as a hard cut: the in-transition / out-transition
# windows collapse to a single instant.  Both endpoints get the same value so
# the heatmap and CSV show a single marker per cut, not a near-zero range that
# looks like a measurement artefact.
HARDCUT_GAP_SECS = 0.30   # gap between cur-fade-out and nxt-fade-in below
                          # which the transition is treated as overlapping
HARDCUT_OVERLAP_SECS = 0.30   # overlap above which the transition is a real
                              # mix (anything between these two thresholds is
                              # ambiguous → still treated as a hard cut)

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

    # Hard-cut flags — True when the in/out transition is essentially a clean
    # cut (no overlap between songs) rather than a real mix.  In that case
    # in_start == in_end (or out_start == out_end) and the UI/CSV should show
    # a single marker for the cut, not a zero-width range.
    is_hardcut_in:   bool = False
    is_hardcut_out:  bool = False

    # Engine DJ hotcue positions (set-time seconds), None if not present.
    # cue1 = in-transition start, cue3 = in-transition end,
    # cue6 = out-transition start, cue8 = out-transition end.
    cue1_secs: Optional[float] = None
    cue3_secs: Optional[float] = None
    cue6_secs: Optional[float] = None
    cue8_secs: Optional[float] = None

    # Position inside the SET (in chroma frames) where this track's locked
    # alignment begins.  Needed by align_transition_bpms() to rewrite the
    # played-curve and per-segment ratios inside transition windows after the
    # initial per-track segment refinement has run.
    lock_offset_frames: int = 0
    # The original (pre-warp) reference chroma — kept around so transition
    # re-fitting can probe new ratios without re-decoding the audio file.
    ref_chroma_orig:    Optional[np.ndarray] = None

    # Per-segment ratio drift — captures slow speed changes within a track.
    # segment_centers_secs[i] is the set-time at the centre of segment i,
    # segment_ratios[i] is that segment's best playback ratio relative to the
    # source file (1.0 = original speed; >1 = faster).  Used to render the
    # BPM-over-time line in the heatmap and to back the "real" tempo column
    # in the CSV (mean ± stdev).
    segment_centers_secs: Optional[np.ndarray] = None
    segment_ratios:       Optional[np.ndarray] = None


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
# Engine DJ hotcue reader
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_hotcue_blob(blob: bytes) -> dict:
    """Decompress and parse an Engine DJ PerformanceData.quickCues blob.

    Returns {cue_number: position_secs} for cues with a valid (>=0) position.
    Cue numbers are 1-indexed as stored in the binary data.
    """
    if not blob or len(blob) < 5:
        return {}
    try:
        raw = zlib.decompress(blob[4:])
    except zlib.error:
        return {}

    pos = 0
    if len(raw) < 8:
        return {}
    n_cues = struct.unpack_from(">q", raw, pos)[0]
    pos += 8

    cues = {}
    for cue_idx in range(n_cues):
        if pos >= len(raw):
            break
        name_len = struct.unpack_from("B", raw, pos)[0]
        pos += 1
        if pos + name_len > len(raw):
            break
        pos += name_len   # skip name bytes
        if pos + 12 > len(raw):
            break
        position_samples = struct.unpack_from(">d", raw, pos)[0]
        pos += 8
        pos += 4   # skip ARGB color
        if position_samples >= 0:
            cue_number = cue_idx + 1   # 1-indexed
            cues[cue_number] = position_samples / ENGINE_SAMPLE_RATE
    return cues


def _track_id_for_path(db_path: str, file_path: str) -> Optional[int]:
    """Look up the Engine DJ track ID for a given file path.

    Tries the exact path first, then falls back to matching just the filename.
    """
    if not os.path.isfile(db_path):
        return None
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        # Normalise separators for the comparison
        norm = file_path.replace("\\", "/")
        cur.execute("SELECT id FROM Track WHERE filename = ?", (norm,))
        row = cur.fetchone()
        if row:
            con.close()
            return int(row[0])
        # Fall back: match by the last path component (filename only)
        basename = os.path.basename(norm)
        cur.execute("SELECT id FROM Track WHERE filename LIKE ?",
                    (f"%/{basename}",))
        row = cur.fetchone()
        con.close()
        return int(row[0]) if row else None
    except Exception:
        return None


def _hotcues_for_track_id(db_path: str, track_id: int) -> dict:
    """Query PerformanceData.quickCues for track_id and parse the blob."""
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(
            "SELECT quickCues FROM PerformanceData WHERE trackId = ?",
            (track_id,)
        )
        row = cur.fetchone()
        con.close()
        if not row or row[0] is None:
            return {}
        return _parse_hotcue_blob(bytes(row[0]))
    except Exception:
        return {}


def read_engine_hotcues(tracks: list, db_path: str) -> None:
    """Populate cue1/cue3/cue6/cue8_secs on each Track from Engine DJ's DB.

    The cue positions are stored in samples at ENGINE_SAMPLE_RATE; we convert
    to seconds and store relative to the *track file* start (not set time).
    The caller is responsible for converting to set-time by adding the track's
    detected_start_secs offset when needed (done in write_tracklist /
    write_review_ui).

    Cue mapping:
        cue1 → in-transition start
        cue3 → in-transition end
        cue6 → out-transition start
        cue8 → out-transition end
    """
    if not os.path.isfile(db_path):
        print(f"  Engine DB not found, skipping hotcues: {db_path}")
        return

    for t in tracks:
        if t.is_mashup or t.is_missing:
            continue
        track_id = _track_id_for_path(db_path, t.file_path)
        if track_id is None:
            continue
        cues = _hotcues_for_track_id(db_path, track_id)
        if 1 in cues:
            t.cue1_secs = cues[1]
        if 3 in cues:
            t.cue3_secs = cues[3]
        if 6 in cues:
            t.cue6_secs = cues[6]
        if 8 in cues:
            t.cue8_secs = cues[8]


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

    Joint detection — each transition (N → N+1) produces two anchor times that
    are SHARED between both tracks by construction:

        T_overlap_start  =  cur.out_start  =  nxt.in_start
        T_overlap_end    =  cur.out_end    =  nxt.in_end

    so the in-transition of track N+1 occupies the same set-time interval as
    the out-transition of track N.  This makes adjacent rows in the heatmap
    line up cleanly: one marker per shared edge, not two near-equal ones.

    The two anchors are found by analysing both played-curves over the same
    overlap window:

      • T_overlap_start  =  the latest of (last_cur_below_onset_before_overlap,
                                            first_nxt_above_onset_in_overlap)
                            — i.e. the moment BOTH tracks are audible.

      • T_overlap_end    =  the earliest of (last_cur_above_onset,
                                              first_nxt_above_played_threshold_solo)
                            — i.e. the moment the outgoing track has cleared
                              and the incoming one is the lead.

    HARD CUTS — when the outgoing fade-out and incoming fade-in barely overlap
    (gap larger than HARDCUT_GAP_SECS, or overlap shorter than
    HARDCUT_OVERLAP_SECS), it's a clean cut: both anchors collapse to a single
    boundary instant (the midpoint of cur_silenced and nxt_audible).  The
    track is flagged is_hardcut_out / is_hardcut_in so the UI shows one marker
    instead of two near-identical ones.

    Mashup tracks are skipped (they don't advance the timeline).
    """
    detectable = [t for t in tracks
                  if not t.is_mashup and not t.is_missing
                  and t.played_curve is not None]

    # Defaults — used for the very first track's in_* and very last out_*.
    for t in detectable:
        t.in_start_secs  = t.detected_start_secs
        t.in_end_secs    = t.detected_start_secs
        t.out_start_secs = t.detected_end_secs
        t.out_end_secs   = t.detected_end_secs

    BUFFER_SECS = 30.0

    for i in range(len(detectable) - 1):
        cur = detectable[i]
        nxt = detectable[i + 1]

        # Search window: the overlap zone, padded.
        win_start = max(cur.detected_start_secs,
                        nxt.detected_start_secs - BUFFER_SECS)
        win_end   = min(nxt.detected_end_secs,
                        cur.detected_end_secs + BUFFER_SECS)
        s0 = max(0, int(win_start * frames_per_sec))
        s1 = min(len(cur.played_curve),
                 len(nxt.played_curve),
                 int(win_end * frames_per_sec))
        if s1 <= s0:
            continue

        cur_fine = _smooth(cur.played_curve, int(FINE_SMOOTH_SECS * frames_per_sec))
        nxt_fine = _smooth(nxt.played_curve, int(FINE_SMOOTH_SECS * frames_per_sec))

        # Audibility masks for both tracks across the search window
        cur_audible = cur_fine[s0:s1] >= ONSET_THRESHOLD
        nxt_audible = nxt_fine[s0:s1] >= ONSET_THRESHOLD

        # The "moment the outgoing track stops being audible" — last cur_audible frame.
        cur_audible_idx = np.where(cur_audible)[0]
        if len(cur_audible_idx):
            cur_silenced_local = int(cur_audible_idx[-1]) + 1   # frame after last
        else:
            cur_silenced_local = 0

        # The "moment the incoming track first becomes audible" — first nxt_audible frame.
        nxt_audible_idx = np.where(nxt_audible)[0]
        if len(nxt_audible_idx):
            nxt_first_local = int(nxt_audible_idx[0])
        else:
            nxt_first_local = s1 - s0 - 1

        # ── Build BOTH-AUDIBLE region — frames where cur and nxt overlap.
        both = cur_audible & nxt_audible
        both_idx = np.where(both)[0]
        if len(both_idx):
            overlap_start_local = int(both_idx[0])
            overlap_end_local   = int(both_idx[-1]) + 1
            overlap_secs = (overlap_end_local - overlap_start_local) / frames_per_sec
            gap_secs     = 0.0
        else:
            # No frame where both are audible.  Either there's a gap (cur ends
            # before nxt starts) or the audibility windows don't agree at all.
            overlap_secs = 0.0
            gap_secs     = (nxt_first_local - cur_silenced_local) / frames_per_sec
            overlap_start_local = cur_silenced_local
            overlap_end_local   = nxt_first_local

        is_hardcut = (overlap_secs < HARDCUT_OVERLAP_SECS or
                      gap_secs    > HARDCUT_GAP_SECS)

        if is_hardcut:
            # Single boundary instant — the midpoint of "cur stopped" and
            # "nxt started".  When there's a gap that's the silence midpoint;
            # when they brush past each other it's an instantaneous cut.
            boundary_frame = s0 + (cur_silenced_local + nxt_first_local) // 2
            t_secs = boundary_frame / frames_per_sec
            cur.out_start_secs = t_secs
            cur.out_end_secs   = t_secs
            cur.is_hardcut_out = True
            nxt.in_start_secs  = t_secs
            nxt.in_end_secs    = t_secs
            nxt.is_hardcut_in  = True
        else:
            # Real mix: the overlap window has positive length.
            t_start = (s0 + overlap_start_local) / frames_per_sec
            t_end   = (s0 + overlap_end_local)   / frames_per_sec
            cur.out_start_secs = t_start
            cur.out_end_secs   = t_end
            cur.is_hardcut_out = False
            nxt.in_start_secs  = t_start
            nxt.in_end_secs    = t_end
            nxt.is_hardcut_in  = False


def _solo_ratio_at_boundary(
    seg_centers_secs: np.ndarray,
    seg_ratios:       np.ndarray,
    boundary_secs:    float,
    side:             str,                # "before" or "after"
    margin_secs:      float = SOLO_MARGIN_SECS,
) -> float:
    """Return the segment ratio that best represents the track's *solo* BPM
    just before / after a transition boundary.

    side="before" — segments whose centre is at most `boundary_secs - margin`
                    (i.e. before the track starts handing off).  Returns the
                    LATEST such ratio.
    side="after"  — segments whose centre is at least `boundary_secs + margin`
                    (i.e. after the track has finished mixing in).  Returns
                    the EARLIEST such ratio.

    Falls back to the closest available segment ratio when nothing qualifies
    (e.g. the track is too short to have a clean solo region).  Returns NaN
    if there are no segments at all.
    """
    if seg_centers_secs is None or len(seg_centers_secs) == 0:
        return float("nan")
    if side == "before":
        mask = seg_centers_secs <= (boundary_secs - margin_secs)
        if mask.any():
            return float(seg_ratios[mask][-1])
        # Fall back to nearest segment
        idx = int(np.argmin(np.abs(seg_centers_secs - boundary_secs)))
        return float(seg_ratios[idx])
    else:  # "after"
        mask = seg_centers_secs >= (boundary_secs + margin_secs)
        if mask.any():
            return float(seg_ratios[mask][0])
        idx = int(np.argmin(np.abs(seg_centers_secs - boundary_secs)))
        return float(seg_ratios[idx])


def _piecewise_warp_for_transition(*args, **kwargs):
    """Deprecated; superseded by _build_ramp_warp.  Kept as a stub so any
    earlier call sites raise loudly during development."""
    raise NotImplementedError(
        "_piecewise_warp_for_transition has been replaced by _build_ramp_warp"
    )


def align_transition_bpms(
    tracks:         list,
    set_chroma:     np.ndarray,
    frames_per_sec: float,
) -> None:
    """Enforce same-BPM-for-both-tracks across each (non-hardcut) transition.

    Physics: when two tracks beatmatch through a transition, both decks are
    locked at the SAME tempo by definition (sync button or pitch-matched).
    But the per-track segment fitter (refine_tempo_per_segment) sees the two
    tracks' chromas SUPERIMPOSED inside the overlap window, so its segment
    ratios there are corrupted by the other track.

    This pass corrects that:
      1. Read each track's "solo" boundary ratios:
         • r_cur_out  = cur's segment ratio just BEFORE out_start
                        (the BPM cur was playing at when the mix began)
         • r_nxt_in   = nxt's segment ratio just AFTER in_end
                        (the BPM nxt settled into after the mix)
      2. Choose a single ramp (r_cur_ramp_end → r_nxt_ramp_start) that BOTH
         tracks must follow inside the transition window.  Default endpoints
         are r_cur_out and r_nxt_in (a linear bridge).  The endpoints are
         then jointly optimised within ±TRANSITION_RAMP_HALF_PCT to maximise
         the SUM of cur+nxt similarity inside the overlap window.
      3. Replace the segment ratios of cur (inside out_start..out_end) and
         nxt (inside in_start..in_end) with samples from that shared ramp.
      4. Rebuild each track's warped ref chroma in the transition region
         using the shared ramp, then recompute their played_curves so the
         heatmap reflects the corrected warp.

    Hard cuts are skipped — there's no shared BPM there.  Mashup / missing
    tracks were already filtered out before this runs.
    """
    if not TRANSITION_BPM_SYNC_ENABLED:
        return

    detectable = [t for t in tracks
                  if not t.is_mashup and not t.is_missing
                  and t.played_curve is not None
                  and t.segment_ratios is not None
                  and t.ref_chroma_orig is not None]

    candidate_pcts = np.arange(
        -TRANSITION_RAMP_HALF_PCT,
        TRANSITION_RAMP_HALF_PCT + TRANSITION_RAMP_STEP_PCT / 2,
        TRANSITION_RAMP_STEP_PCT,
    )

    for i in range(len(detectable) - 1):
        cur = detectable[i]
        nxt = detectable[i + 1]
        if cur.is_hardcut_out or nxt.is_hardcut_in:
            continue
        if cur.out_start_secs is None or cur.out_end_secs is None:
            continue
        if cur.out_end_secs - cur.out_start_secs < 0.5:
            # Window too short — not enough audio for a meaningful re-fit
            continue

        # Boundary solo ratios (defaults for the ramp endpoints)
        r_cur_solo = _solo_ratio_at_boundary(
            cur.segment_centers_secs, cur.segment_ratios,
            cur.out_start_secs, "before")
        r_nxt_solo = _solo_ratio_at_boundary(
            nxt.segment_centers_secs, nxt.segment_ratios,
            nxt.in_end_secs, "after")
        if not (np.isfinite(r_cur_solo) and np.isfinite(r_nxt_solo)):
            continue

        t_start_f = int(round(cur.out_start_secs * frames_per_sec))
        t_end_f   = int(round(cur.out_end_secs   * frames_per_sec))
        if t_end_f - t_start_f < 4:
            continue
        n_set = set_chroma.shape[1]
        t_start_f = max(0, t_start_f)
        t_end_f   = min(n_set, t_end_f)
        set_overlap = set_chroma[:, t_start_f:t_end_f]
        n_overlap = set_overlap.shape[1]

        # ── Joint ramp probe ─────────────────────────────────────────────
        # The ramp's ENDPOINTS are searched in a small box around the
        # neighbour solo ratios.  This lets the BPM drift inside the
        # transition (DJ rides the fader) be discovered, but constrains
        # the result to be physically plausible.
        #
        # For each (ra, rb) candidate we synthesise the chroma each track
        # would produce inside the transition window if it were warped at
        # that ramp, and score the SUM of (set·cur_warped) + (set·nxt_warped)
        # — both decks contribute to the audio inside the overlap, so the
        # joint score is the right thing to maximise.
        cur_orig_anchor = max(0, int(round(
            (t_start_f - cur.lock_offset_frames) * r_cur_solo
        )))
        cur_orig_anchor = min(cur_orig_anchor, cur.ref_chroma_orig.shape[1] - 4)
        nxt_orig_anchor = max(0, int(round(
            (t_start_f - nxt.lock_offset_frames) * r_nxt_solo
        )))
        nxt_orig_anchor = min(nxt_orig_anchor, nxt.ref_chroma_orig.shape[1] - 4)

        best_score = -np.inf
        best_endpoints = (r_cur_solo, r_nxt_solo)

        for pct_a in candidate_pcts:
            for pct_b in candidate_pcts:
                ra = r_cur_solo + pct_a / 100.0
                rb = r_nxt_solo + pct_b / 100.0

                cur_warped = _build_ramp_warp(
                    cur.ref_chroma_orig, n_overlap,
                    cur_orig_anchor, ra, rb, frames_per_sec,
                )
                nxt_warped = _build_ramp_warp(
                    nxt.ref_chroma_orig, n_overlap,
                    nxt_orig_anchor, ra, rb, frames_per_sec,
                )

                score_cur = float(np.einsum(
                    "ij,ij->", set_overlap, cur_warped[:, :n_overlap]
                )) / n_overlap
                score_nxt = float(np.einsum(
                    "ij,ij->", set_overlap, nxt_warped[:, :n_overlap]
                )) / n_overlap
                score = score_cur + score_nxt
                if score > best_score:
                    best_score = score
                    best_endpoints = (ra, rb)

        # ── Apply the winning ramp to BOTH tracks ────────────────────────
        ra, rb = best_endpoints

        # Rewrite segment ratios that fall inside the transition with the
        # ramp values, so the CSV / heatmap reflect the corrected BPM.
        for trk, t0, t1 in [
            (cur, cur.out_start_secs, cur.out_end_secs),
            (nxt, nxt.in_start_secs,  nxt.in_end_secs),
        ]:
            if trk.segment_centers_secs is None:
                continue
            mask = ((trk.segment_centers_secs >= t0) &
                    (trk.segment_centers_secs <= t1))
            if mask.any():
                centers = trk.segment_centers_secs[mask]
                rel = (centers - t0) / max(1e-6, t1 - t0)
                trk.segment_ratios[mask] = ra + (rb - ra) * rel
            else:
                # No segment centre inside the window — append synthetic ones
                # so the CSV / plot have something to show.
                center = (t0 + t1) / 2
                rel = 0.5
                value = ra + (rb - ra) * rel
                trk.segment_centers_secs = np.append(trk.segment_centers_secs, center)
                trk.segment_ratios       = np.append(trk.segment_ratios, value)
                order = np.argsort(trk.segment_centers_secs)
                trk.segment_centers_secs = trk.segment_centers_secs[order]
                trk.segment_ratios       = trk.segment_ratios[order]

        # Rebuild each track's played_curve over [t_start_f, t_end_f] using
        # the shared ramp so the heatmap and review UI show clean alignment.
        for trk, neighbour_solo in [(cur, r_cur_solo), (nxt, r_nxt_solo)]:
            ref_orig = trk.ref_chroma_orig
            n_orig   = ref_orig.shape[1]
            ref_start_in_warped = t_start_f - trk.lock_offset_frames
            if ref_start_in_warped < 0 or ref_start_in_warped >= len(trk.played_curve):
                continue
            cur_orig_pos = int(round(ref_start_in_warped * neighbour_solo))
            cur_orig_pos = max(0, min(cur_orig_pos, n_orig - 4))
            warped_seg = _build_ramp_warp(
                ref_orig, n_overlap, cur_orig_pos, ra, rb, frames_per_sec,
            )
            # Recompute per-frame similarity for the transition slice.
            sims = np.einsum(
                "ij,ij->j", set_overlap, warped_seg[:, :n_overlap]
            ).astype(np.float32)
            # Splice into the played_curve at the transition's set-time range.
            trk.played_curve[t_start_f:t_start_f + len(sims)] = sims


def _build_ramp_warp(
    ref_orig:        np.ndarray,
    n_out:           int,
    orig_anchor:     int,
    r_start:         float,
    r_end:           float,
    frames_per_sec:  float,
) -> np.ndarray:
    """Produce a (12 × n_out) chroma slice by walking ref_orig from
    orig_anchor with a linearly varying ratio r_start → r_end.

    Used by align_transition_bpms() to score and apply ramp warps without
    going through the full piecewise-warp helper.  Each ~1-second sub-step
    uses a fixed local ratio (linear interp at sub-step centre).
    """
    out = np.zeros((12, n_out), dtype=np.float32)
    n_orig = ref_orig.shape[1]
    sub_n = max(4, int(round(1.0 * frames_per_sec)))
    cur_out  = 0
    cur_orig = orig_anchor
    while cur_out < n_out and cur_orig < n_orig - 2:
        this_n = min(sub_n, n_out - cur_out)
        rel = (cur_out + this_n / 2) / max(1, n_out)
        local_r = r_start + (r_end - r_start) * rel
        orig_slice_n = max(2, int(round(this_n * local_r)))
        orig_slice = ref_orig[:, cur_orig:min(cur_orig + orig_slice_n, n_orig)]
        if orig_slice.shape[1] < 2:
            break
        w = warp_chroma(orig_slice, local_r)
        n_write = min(this_n, w.shape[1])
        if n_write <= 0:
            break
        out[:, cur_out:cur_out + n_write] = w[:, :n_write]
        cur_out  += n_write
        cur_orig += orig_slice_n
    return out


def build_tempo_ratios(
    max_pct:    float,
    step_pct:   float = TEMPO_STEP_PCT,
    center_pct: float = 0.0,
) -> list:
    """Build a list of speed ratios spanning center ±max_pct in step_pct increments."""
    n_steps = max(1, int(round(max_pct / step_pct)))
    pcts = np.unique(np.concatenate([
        [center_pct],
        center_pct + np.arange(step_pct, max_pct + step_pct / 2, step_pct),
        center_pct - np.arange(step_pct, max_pct + step_pct / 2, step_pct),
    ]))
    return [1.0 + p / 100.0 for p in sorted(pcts)]


def search_tempo_two_stage(
    set_chroma:         np.ndarray,
    ref_chroma_orig:    np.ndarray,
    coarse_max_pct:     float,
    search_start_frame: int,
    search_end_frame:   int,
) -> dict:
    """Two-stage tempo search: coarse sweep, then fine refinement around the winner.

    Stage 1 — coarse: ±coarse_max_pct in TEMPO_STEP_PCT steps (default 0.5 %).
    Stage 2 — fine:   ±REFINE_HALF_WIDTH around the coarse winner in
                      REFINE_STEP_PCT steps (default 0.05 %).

    Stage 2 typically gains 0.0–0.3 % accuracy and noticeably improves the
    played-curve stability through the middle of long tracks where even a
    0.25 % bias accumulates into many seconds of drift.
    """
    coarse_ratios = build_tempo_ratios(coarse_max_pct, TEMPO_STEP_PCT)
    coarse_best = search_with_tempo_ratios(
        set_chroma, ref_chroma_orig, coarse_ratios,
        search_start_frame, search_end_frame,
    )
    if coarse_best is None:
        return None

    coarse_pct = (coarse_best["ratio"] - 1.0) * 100.0
    fine_ratios = build_tempo_ratios(
        REFINE_HALF_WIDTH, REFINE_STEP_PCT, center_pct=coarse_pct,
    )
    fine_best = search_with_tempo_ratios(
        set_chroma, ref_chroma_orig, fine_ratios,
        search_start_frame, search_end_frame,
    )
    if fine_best is None or fine_best["confidence"] <= coarse_best["confidence"]:
        return coarse_best
    return fine_best


def refine_tempo_per_segment(
    set_chroma:      np.ndarray,
    ref_chroma_orig: np.ndarray,
    global_offset:   int,
    global_ratio:    float,
    frames_per_sec:  float,
) -> tuple:
    """Find a separate best playback ratio for each segment of the played region.

    Slow speed changes within a track (DJ rides the pitch fader, or cues a
    new beatgrid) cause a single global ratio to drift out of phase with the
    actual audio after a few minutes.  Per-segment refinement fixes this:

        1. Slice the time-warped ref into SEGMENT_LENGTH_SECS chunks.
        2. For each chunk, sweep ±SEGMENT_RATIO_HALF_WIDTH around the global
           ratio in SEGMENT_RATIO_STEP increments and pick the local winner.
        3. Use those per-segment ratios to build a piecewise-warped ref
           chroma whose timing follows the actual set audio frame-for-frame.

    Returns (segment_centers_set_secs, segment_ratios, warped_ref_chroma).
    The warped_ref_chroma can then be passed to compute_played_curve() to
    produce a much cleaner (less drifty) played-curve.
    """
    ref_warped = warp_chroma(ref_chroma_orig, global_ratio)
    n_ref      = ref_warped.shape[1]
    n_set      = set_chroma.shape[1]

    seg_frames = int(round(SEGMENT_LENGTH_SECS * frames_per_sec))
    if seg_frames <= 0 or n_ref < 2 * seg_frames:
        # Track too short — fall back to global ratio everywhere.
        return (np.array([global_offset / frames_per_sec]),
                np.array([global_ratio]),
                ref_warped)

    # Iterate over set-frame segments where this track is supposed to be playing.
    seg_starts_set = list(range(global_offset,
                                min(global_offset + n_ref, n_set) - seg_frames,
                                seg_frames))
    if not seg_starts_set:
        return (np.array([global_offset / frames_per_sec]),
                np.array([global_ratio]),
                ref_warped)

    centers_set, ratios = [], []
    rebuilt = ref_warped.copy()
    n_set_chroma = set_chroma.shape[1]

    # Track how far the cumulative drift has shifted us in the original ref.
    # We re-anchor each segment at the previous segment's end in set time so
    # short-term ratio changes don't accumulate position errors.
    cur_ref_pos = 0  # position inside ref_warped that maps to seg_starts_set[0]

    candidate_pcts = np.arange(
        -SEGMENT_RATIO_HALF_WIDTH,
        SEGMENT_RATIO_HALF_WIDTH + SEGMENT_RATIO_STEP / 2,
        SEGMENT_RATIO_STEP,
    )

    for s_set in seg_starts_set:
        # Set-side window for this segment
        set_seg = set_chroma[:, s_set : s_set + seg_frames]
        if set_seg.shape[1] < 4:
            continue

        # For each candidate local ratio, time-stretch the ORIGINAL ref's
        # corresponding slice and score it against set_seg.
        # The original-ref slice covers seg_frames * global_ratio frames
        # of the unwarped reference, anchored at cur_ref_pos / global_ratio.
        ref_orig_anchor = int(round(cur_ref_pos * global_ratio))
        ref_orig_len    = int(round(seg_frames * global_ratio))
        ref_orig_slice  = ref_chroma_orig[
            :,
            ref_orig_anchor : min(ref_orig_anchor + ref_orig_len,
                                  ref_chroma_orig.shape[1]),
        ]
        if ref_orig_slice.shape[1] < 4:
            continue

        best_pct, best_score = 0.0, -np.inf
        for pct in candidate_pcts:
            local_ratio = global_ratio + pct / 100.0
            warped_seg = warp_chroma(ref_orig_slice, local_ratio)
            n_seg = min(seg_frames, warped_seg.shape[1])
            if n_seg < 4:
                continue
            score = float(np.einsum(
                "ij,ij->",
                set_seg[:, :n_seg],
                warped_seg[:, :n_seg],
            )) / n_seg
            if score > best_score:
                best_score = score
                best_pct   = pct

        local_ratio = global_ratio + best_pct / 100.0
        ratios.append(local_ratio)
        centers_set.append((s_set + seg_frames / 2) / frames_per_sec)

        # Rebuild this segment of ref_warped using the local ratio so the
        # final played-curve aligns frame-for-frame.
        warped_seg = warp_chroma(ref_orig_slice, local_ratio)
        n_seg = min(seg_frames, warped_seg.shape[1], n_ref - cur_ref_pos)
        if n_seg > 0:
            rebuilt[:, cur_ref_pos : cur_ref_pos + n_seg] = warped_seg[:, :n_seg]

        cur_ref_pos += seg_frames

    if not ratios:
        return (np.array([global_offset / frames_per_sec]),
                np.array([global_ratio]),
                ref_warped)

    return (np.array(centers_set, dtype=np.float64),
            np.array(ratios,      dtype=np.float64),
            rebuilt)


# ═══════════════════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_time(secs: float) -> str:
    """Format seconds as M:SS.ss or H:MM:SS.ss."""
    if secs is None:
        return "    ?    "
    secs = max(0.0, secs)
    h = int(secs) // 3600
    m = (int(secs) // 60) % 60
    s = secs - 60 * (h * 60 + m)
    if h:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"


def fmt_time_hms(secs: float, with_hours: Optional[bool] = None) -> str:
    """Format seconds as (hh:)mm:ss.  Used for plot-axis tick labels.

    `with_hours=True` forces HH:MM:SS even when secs < 1 h (so all ticks line
    up nicely on a long set).  with_hours=None auto-decides based on value.
    """
    if secs is None:
        return "?"
    secs = max(0.0, secs)
    h = int(secs) // 3600
    m = (int(secs) // 60) % 60
    s = int(round(secs)) % 60
    show_h = with_hours if with_hours is not None else (h > 0)
    if show_h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _hms_ticks(x0: float, x1: float, n_ticks: int = 8) -> tuple:
    """Build (tick_values, tick_labels) for an axis spanning [x0, x1] seconds.

    Picks a "round" tick spacing — 1, 2, 5, 10, 15, 30 seconds, or 1, 2, 5,
    10, 15, 30 minutes — so labels land on memorable times.  Forces HH:MM:SS
    formatting if the window covers an hour or more.
    """
    span = max(1e-3, x1 - x0)
    target = span / max(1, n_ticks)
    candidates_secs = [
        1, 2, 5, 10, 15, 30,
        60, 120, 300, 600, 900, 1800,
        3600, 7200,
    ]
    step = next((c for c in candidates_secs if c >= target), candidates_secs[-1])
    first = int(np.ceil(x0 / step)) * step
    vals = list(np.arange(first, x1 + step / 2, step))
    if not vals:
        vals = [x0, x1]
    use_hours = x1 >= 3600
    labels = [fmt_time_hms(v, with_hours=use_hours) for v in vals]
    return vals, labels


def set_to_file_time(set_time: float, track) -> float:
    """Convert a set-time (seconds) to the position in the original track file."""
    lock_secs = track.lock_offset_frames / (ANALYSIS_SR / HOP_LENGTH)
    return (set_time - lock_secs) * track.tempo_ratio


def write_tracklist(tracks: list, out_path: str):
    """Write the tracklist as a CSV.

    Columns:
        # — playlist position from the source CSV
        title, artist
        status — "ok" (mapped), "low_confidence", "mashup", "missing"
        in_start, in_end       — incoming transition window (set time)
        out_start, out_end     — outgoing transition window (set time)
        in_start_hms, in_end_hms, out_start_hms, out_end_hms — same, formatted
        in_start_file, in_end_file, out_start_file, out_end_file — file time (secs)
        in_start_file_hms, …   — same, formatted
        confidence             — locked-region mean similarity
        bpm_shift_pct          — global playback-speed shift (%, vs original)
        bpm_shift_min_pct      — slowest segment within the track
        bpm_shift_max_pct      — fastest segment within the track
        bpm_shift_stdev_pct    — stdev across segments (intra-track variation)
        original_bpm           — the BPM column from the playlist CSV
        played_bpm_mean        — original_bpm × mean_segment_ratio (empty if BPM=0)

    All timestamps are set-time seconds (decimals, ms precision).
    File-time timestamps are seconds into the original track file (tempo-corrected).
    Mashup and missing tracks have empty timestamp cells.
    """
    fieldnames = [
        "#", "title", "artist", "status",
        "in_start", "in_end", "out_start", "out_end",
        "in_start_hms", "in_end_hms", "out_start_hms", "out_end_hms",
        "in_start_file", "in_end_file", "out_start_file", "out_end_file",
        "in_start_file_hms", "in_end_file_hms", "out_start_file_hms", "out_end_file_hms",
        "hardcut_in", "hardcut_out",
        "confidence",
        "bpm_shift_pct", "bpm_shift_min_pct", "bpm_shift_max_pct",
        "bpm_shift_stdev_pct",
        "original_bpm", "played_bpm_mean",
        "cue1_set_secs", "cue3_set_secs", "cue6_set_secs", "cue8_set_secs",
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
                if t.segment_ratios is not None and len(t.segment_ratios):
                    seg_pcts = (t.segment_ratios - 1.0) * 100
                    seg_min  = f"{seg_pcts.min():+.2f}"
                    seg_max  = f"{seg_pcts.max():+.2f}"
                    seg_std  = f"{seg_pcts.std():.2f}"
                    mean_ratio = float(t.segment_ratios.mean())
                else:
                    seg_min = seg_max = seg_std = ""
                    mean_ratio = t.tempo_ratio

                played_bpm_mean = (
                    f"{t.bpm * mean_ratio:.2f}" if t.bpm > 0 else ""
                )

                # Convert track-relative cue positions to set time.
                # Cue seconds are in original-speed track time; dividing by
                # tempo_ratio converts to played duration, then add track start.
                def _cue_set_secs(cue_track_secs):
                    if cue_track_secs is None or t.detected_start_secs is None:
                        return ""
                    ratio = mean_ratio if mean_ratio else 1.0
                    return f"{t.detected_start_secs + cue_track_secs / ratio:.3f}"

                in_start_f  = set_to_file_time(t.in_start_secs,  t) if t.in_start_secs  is not None else None
                in_end_f    = set_to_file_time(t.in_end_secs,    t) if t.in_end_secs    is not None else None
                out_start_f = set_to_file_time(t.out_start_secs, t) if t.out_start_secs is not None else None
                out_end_f   = set_to_file_time(t.out_end_secs,   t) if t.out_end_secs   is not None else None

                row.update({
                    "in_start":         f"{t.in_start_secs:.3f}"  if t.in_start_secs  is not None else "",
                    "in_end":           f"{t.in_end_secs:.3f}"    if t.in_end_secs    is not None else "",
                    "out_start":        f"{t.out_start_secs:.3f}" if t.out_start_secs is not None else "",
                    "out_end":          f"{t.out_end_secs:.3f}"   if t.out_end_secs   is not None else "",
                    "in_start_hms":     fmt_time(t.in_start_secs),
                    "in_end_hms":       fmt_time(t.in_end_secs),
                    "out_start_hms":    fmt_time(t.out_start_secs),
                    "out_end_hms":      fmt_time(t.out_end_secs),
                    "in_start_file":    f"{in_start_f:.3f}"  if in_start_f  is not None else "",
                    "in_end_file":      f"{in_end_f:.3f}"    if in_end_f    is not None else "",
                    "out_start_file":   f"{out_start_f:.3f}" if out_start_f is not None else "",
                    "out_end_file":     f"{out_end_f:.3f}"   if out_end_f   is not None else "",
                    "in_start_file_hms":  fmt_time(in_start_f)  if in_start_f  is not None else "",
                    "in_end_file_hms":    fmt_time(in_end_f)    if in_end_f    is not None else "",
                    "out_start_file_hms": fmt_time(out_start_f) if out_start_f is not None else "",
                    "out_end_file_hms":   fmt_time(out_end_f)   if out_end_f   is not None else "",
                    "hardcut_in":       "1" if t.is_hardcut_in  else "",
                    "hardcut_out":      "1" if t.is_hardcut_out else "",
                    "confidence":       f"{t.confidence:.3f}",
                    "bpm_shift_pct":    f"{(t.tempo_ratio - 1.0) * 100:+.2f}",
                    "bpm_shift_min_pct":   seg_min,
                    "bpm_shift_max_pct":   seg_max,
                    "bpm_shift_stdev_pct": seg_std,
                    "original_bpm":     f"{t.bpm:.2f}" if t.bpm > 0 else "",
                    "played_bpm_mean":  played_bpm_mean,
                    "cue1_set_secs":    _cue_set_secs(t.cue1_secs),
                    "cue3_set_secs":    _cue_set_secs(t.cue3_secs),
                    "cue6_set_secs":    _cue_set_secs(t.cue6_secs),
                    "cue8_set_secs":    _cue_set_secs(t.cue8_secs),
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
    fig, ax = plt.subplots(figsize=(42, fig_height))   # 3× the original 14" wide
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="upper",
        extent=[0, set_duration, len(visible) - 0.5, -0.5],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="per-frame similarity (cosine)")

    # Marker set per row.  In addition to the triangle glyph, each marker
    # gets a short vertical line that overhangs the row by EXTEND fraction
    # of the row height — this makes neighbour rows line up visually so you
    # can confirm that "out_end of N" really sits at the same set time as
    # "in_end of N+1" (joint detection guarantees they're equal, but the
    # overhang makes it readable on the rendered heatmap).
    #
    # Triangle / line styling:
    #   in_start  — green ▶  (mix-in begins)
    #   in_end    — orange ▶ (previous track gone)        — also = cur.out_end
    #   out_start — orange ◀ (next track audible)         — also = nxt.in_start
    #   out_end   — red ◀    (this track gone)
    # Hard cuts collapse the in/out window to a single instant; in that case
    # only the *_end marker is drawn (the orange ▶ / red ◀ pair) so the row
    # doesn't show two near-identical glyphs stacked on top of each other.
    EXTEND = 0.35           # how far the line extends past the row band
    LINE_HALF = 0.5 + EXTEND # half-height of the marker line
    for i, t in enumerate(visible):
        if t.detected_start_secs is None:
            continue

        marker_specs = []
        # in-transition markers — only on first/non-hardcut tracks
        if not t.is_hardcut_in and t.in_start_secs != t.in_end_secs:
            marker_specs.append((t.in_start_secs, "lime",   ">"))
        marker_specs.append((t.in_end_secs,    "orange",     ">"))
        # out-transition markers — only when not a hard cut
        if not t.is_hardcut_out and t.out_start_secs != t.out_end_secs:
            marker_specs.append((t.out_start_secs, "darkorange", "<"))
        marker_specs.append((t.out_end_secs,   "red",        "<"))

        for x_secs_val, color, marker in marker_specs:
            if x_secs_val is None:
                continue
            # Vertical line that overhangs the row above and below
            ax.plot([x_secs_val, x_secs_val],
                    [i - LINE_HALF, i + LINE_HALF],
                    color=color, linewidth=1.0, alpha=0.65,
                    zorder=4, solid_capstyle="butt")
            # Triangle glyph at the row centre
            ax.plot(x_secs_val, i, marker=marker, color=color,
                    markersize=7, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.4)

    tick_vals, tick_labels = _hms_ticks(0, set_duration, n_ticks=24)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(range(len(visible)))
    ax.set_yticklabels([f"{t.number:>2}  {t.title[:38]}" for t in visible], fontsize=8)
    ax.set_xlabel("Set time (h:mm:ss)" if set_duration >= 3600 else "Set time (m:ss)")
    ax.set_title(
        "Per-track playback heatmap — "
        "▶ green/orange = in-transition window, ◀ orange/red = out-transition window"
    )
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
        specs=[[{"secondary_y": True}] * cols for _ in range(rows)],
    )

    for i, t in enumerate(visible):
        row = i // cols + 1
        col = i % cols + 1

        center = t.detected_start_secs
        x0 = max(0.0, center - TRANSITION_WINDOW_SECS)
        x1 = min(set_duration, center + TRANSITION_WINDOW_SECS)

        # Build the slice for this track.  customdata carries the formatted
        # h:mm:ss timestamp so the hover readout matches the axis ticks.
        mask = (t.played_times_secs >= x0) & (t.played_times_secs <= x1)
        x_secs    = t.played_times_secs[mask]
        x_labels  = [fmt_time_hms(s, with_hours=set_duration >= 3600) for s in x_secs]
        fig.add_trace(
            go.Scatter(
                x=x_secs,
                y=t.played_curve[mask],
                customdata=x_labels,
                mode="lines",
                name=f"#{t.number} {t.title[:30]}",
                line=dict(width=2, color="#1f77b4"),
                hovertemplate=(
                    "set time=%{customdata}<br>"
                    f"#{t.number} {t.title[:40]}<br>"
                    "similarity=%{y:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row, col=col, secondary_y=False,
        )

        # Previous track (the one ending around this transition)
        if i > 0:
            prev = visible[i - 1]
            mask_prev = (prev.played_times_secs >= x0) & (prev.played_times_secs <= x1)
            x_prev   = prev.played_times_secs[mask_prev]
            lbl_prev = [fmt_time_hms(s, with_hours=set_duration >= 3600) for s in x_prev]
            fig.add_trace(
                go.Scatter(
                    x=x_prev,
                    y=prev.played_curve[mask_prev],
                    customdata=lbl_prev,
                    mode="lines",
                    name=f"#{prev.number} {prev.title[:30]}",
                    line=dict(width=2, color="#ff7f0e", dash="dot"),
                    hovertemplate=(
                        "set time=%{customdata}<br>"
                        f"#{prev.number} {prev.title[:40]} (outgoing)<br>"
                        "similarity=%{y:.3f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row, col=col, secondary_y=False,
            )

        # BPM-shift line — per-segment % deviation from original speed.  Plotted
        # on a secondary y-axis so it doesn't visually compete with the
        # similarity curves but is still readable on hover.
        if t.segment_centers_secs is not None and len(t.segment_centers_secs) > 1:
            seg_mask = ((t.segment_centers_secs >= x0) &
                        (t.segment_centers_secs <= x1))
            if seg_mask.any():
                seg_x   = t.segment_centers_secs[seg_mask]
                seg_pct = (t.segment_ratios[seg_mask] - 1.0) * 100.0
                seg_lbl = [fmt_time_hms(s, with_hours=set_duration >= 3600)
                           for s in seg_x]
                fig.add_trace(
                    go.Scatter(
                        x=seg_x,
                        y=seg_pct,
                        customdata=seg_lbl,
                        mode="lines+markers",
                        line=dict(width=1.2, color="#9467bd"),
                        marker=dict(size=4, color="#9467bd"),
                        hovertemplate=(
                            "set time=%{customdata}<br>"
                            "bpm shift=%{y:+.2f}%<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=row, col=col, secondary_y=True,
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

        # h:mm:ss tick labels for this panel's x-axis
        tick_vals, tick_labels = _hms_ticks(x0, x1, n_ticks=6)
        fig.update_xaxes(
            title_text="set time (h:mm:ss)" if row == rows else "",
            range=[x0, x1],
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_labels,
            row=row, col=col,
        )
        fig.update_yaxes(
            title_text="similarity" if col == 1 else "",
            range=[0, 1],
            row=row, col=col, secondary_y=False,
        )
        fig.update_yaxes(
            title_text="bpm shift %" if col == cols else "",
            row=row, col=col, secondary_y=True,
            showgrid=False,
            color="#9467bd",
            tickformat="+.2f",
        )

    fig.update_layout(
        title=("Transition explorer — incoming (blue solid) vs outgoing (orange dotted) "
               "track similarity. Purple = per-segment BPM shift % (right axis). "
               "Green dashed = in_start, orange dashed = in_end. "
               "Gray dotted = lock/onset thresholds."),
        height=max(400, rows * 240),
        plot_bgcolor="white",
        font=dict(size=11),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 10

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  Transitions   : {out_path}")


def _downsample_curve(curve: np.ndarray, times: np.ndarray, target_n: int) -> tuple:
    """Downsample a (curve, times) pair to ~target_n points for embedding.

    Plain stride sampling — preserves the shape well enough at 1500 points to
    show transitions clearly without bloating the embedded JSON.
    """
    n = len(curve)
    if n <= target_n:
        return curve.tolist(), times.tolist()
    step = max(1, n // target_n)
    return curve[::step].tolist(), times[::step].tolist()


def write_review_ui(tracks: list, set_duration: float, out_path: str):
    """Write a self-contained HTML page for reviewing the tracklist.

    Two panes side-by-side:
      • LEFT  — sortable, searchable table of every track from the CSV.
                Click any row to focus that track in the right pane.
      • RIGHT — three stacked plotly charts: previous track, current track,
                next track.  Each shows the per-frame similarity curve
                centred on the current track's transition zone, with the
                four transition markers (in_start, in_end, out_start, out_end)
                drawn as vertical dashed lines and the lock and onset
                thresholds drawn as horizontal dotted lines.

    Everything is inlined into a single HTML file using plotly.js from a CDN —
    no server, no extra files; just open it in a browser.
    """
    if not _HAS_PLOTLY:
        # We don't strictly need plotly to write the file (the HTML loads it
        # from a CDN at view time), but we lean on go.Scatter shapes to keep
        # this consistent with the other interactive plot.  Skip if missing
        # so the user still gets a clear install hint.
        print("  (plotly not installed — skipping review UI. pip install plotly)")
        return

    detectable = [t for t in tracks
                  if not t.is_mashup and not t.is_missing
                  and t.played_curve is not None]
    if not detectable:
        print("  (no detectable tracks — skipping review UI)")
        return

    # Build a lookup so the JS can pull a track's curve by its CSV number.
    # Curves are stride-downsampled to keep the page light (~50 KB / track
    # at 1500 points instead of ~1 MB at 100 k frames).
    curves = {}
    for t in detectable:
        y_ds, x_ds = _downsample_curve(t.played_curve, t.played_times_secs, 1500)
        # Per-segment BPM shift trace (small, no need to downsample)
        if t.segment_centers_secs is not None and len(t.segment_centers_secs):
            seg_x = t.segment_centers_secs.tolist()
            seg_y = ((t.segment_ratios - 1.0) * 100.0).tolist()
        else:
            seg_x, seg_y = [], []
        # Convert track-relative cue positions to set time
        mean_ratio = (float(t.segment_ratios.mean())
                      if t.segment_ratios is not None and len(t.segment_ratios)
                      else t.tempo_ratio) or 1.0
        def _to_set(cue):
            if cue is None or t.detected_start_secs is None:
                return None
            return t.detected_start_secs + cue / mean_ratio
        curves[str(t.number)] = {
            "x":          x_ds,
            "y":          y_ds,
            "seg_x":      seg_x,
            "seg_y":      seg_y,
            "in_start":   t.in_start_secs,
            "in_end":     t.in_end_secs,
            "out_start":  t.out_start_secs,
            "out_end":    t.out_end_secs,
            "hardcut_in":  bool(t.is_hardcut_in),
            "hardcut_out": bool(t.is_hardcut_out),
            "lock":       t.detected_start_secs,
            "confidence": float(t.confidence),
            "title":      t.title,
            "artist":     t.artist,
            "cue1":       _to_set(t.cue1_secs),
            "cue3":       _to_set(t.cue3_secs),
            "cue6":       _to_set(t.cue6_secs),
            "cue8":       _to_set(t.cue8_secs),
        }

    # Table rows — same fields the CSV gets, plus a clickable status badge.
    rows = []
    for t in tracks:
        if t.is_mashup:
            status, status_class = "mashup", "mashup"
        elif t.is_missing:
            status, status_class = "missing", "missing"
        elif t.confidence < MIN_CONFIDENCE:
            status, status_class = "low_confidence", "low"
        else:
            status, status_class = "ok", "ok"

        if status in ("ok", "low_confidence"):
            seg_std = ""
            mean_ratio = t.tempo_ratio
            if t.segment_ratios is not None and len(t.segment_ratios):
                seg_std = f"{((t.segment_ratios - 1.0) * 100).std():.2f}"
                mean_ratio = float(t.segment_ratios.mean())
            mean_ratio = mean_ratio or 1.0

            def _to_set_r(cue):
                if cue is None or t.detected_start_secs is None:
                    return None
                return t.detected_start_secs + cue / mean_ratio

            cue1_set = _to_set_r(t.cue1_secs)
            cue3_set = _to_set_r(t.cue3_secs)
            cue6_set = _to_set_r(t.cue6_secs)
            cue8_set = _to_set_r(t.cue8_secs)

            def _aligned(cue_set, marker_secs):
                if cue_set is None or marker_secs is None:
                    return None
                return abs(cue_set - marker_secs) <= CUE_ALIGN_THRESHOLD_SECS

            rows.append({
                "num":           t.number,
                "title":         t.title,
                "artist":        t.artist,
                "status":        status,
                "status_class":  status_class,
                "in_start_hms":  fmt_time(t.in_start_secs),
                "in_end_hms":    fmt_time(t.in_end_secs),
                "out_start_hms": fmt_time(t.out_start_secs),
                "out_end_hms":   fmt_time(t.out_end_secs),
                "confidence":    f"{t.confidence:.3f}",
                "bpm_shift":     f"{(t.tempo_ratio - 1.0) * 100:+.2f}",
                "bpm_stdev":     seg_std,
                "selectable":    True,
                "cue1_hms":  fmt_time(cue1_set) if cue1_set is not None else "",
                "cue3_hms":  fmt_time(cue3_set) if cue3_set is not None else "",
                "cue6_hms":  fmt_time(cue6_set) if cue6_set is not None else "",
                "cue8_hms":  fmt_time(cue8_set) if cue8_set is not None else "",
                "cue1_ok":   _aligned(cue1_set, t.in_start_secs),
                "cue3_ok":   _aligned(cue3_set, t.in_end_secs),
                "cue6_ok":   _aligned(cue6_set, t.out_start_secs),
                "cue8_ok":   _aligned(cue8_set, t.out_end_secs),
            })
        else:
            rows.append({
                "num":          t.number,
                "title":        t.title,
                "artist":       t.artist,
                "status":       status,
                "status_class": status_class,
                "in_start_hms":  "",
                "in_end_hms":    "",
                "out_start_hms": "",
                "out_end_hms":   "",
                "confidence":    "",
                "bpm_shift":     "",
                "bpm_stdev":     "",
                "selectable":    False,
            })

    # Order of detectable tracks (used for prev/next navigation)
    detectable_nums = [t.number for t in detectable]

    payload = {
        "set_duration":     set_duration,
        "min_confidence":   MIN_CONFIDENCE,
        "played_threshold": PLAYED_THRESHOLD,
        "onset_threshold":  ONSET_THRESHOLD,
        "transition_window_secs": TRANSITION_WINDOW_SECS,
        "cue_align_threshold_secs": CUE_ALIGN_THRESHOLD_SECS,
        "rows":             rows,
        "curves":           curves,
        "detectable_nums":  detectable_nums,
    }

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tracklist review</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    --bg: #1a1a1a; --panel: #232323; --border: #333; --text: #e0e0e0;
    --muted: #888; --accent: #4a9eff;
    --ok: #4ade80; --low: #fbbf24; --mashup: #818cf8; --missing: #f87171;
  }
  body { margin:0; padding:0; background:var(--bg); color:var(--text);
         font:13px/1.4 -apple-system, BlinkMacSystemFont, sans-serif; }
  #app { display:grid; grid-template-columns: 820px 1fr; height:100vh; }
  #left { border-right:1px solid var(--border); overflow:hidden;
          display:flex; flex-direction:column; }
  #search { padding:10px; border-bottom:1px solid var(--border);
            background:var(--panel); }
  #search input { width:100%; box-sizing:border-box; padding:6px 8px;
                  background:#0f0f0f; color:var(--text);
                  border:1px solid var(--border); border-radius:4px; }
  #table-wrap { flex:1; overflow:auto; }
  table { border-collapse:collapse; width:100%; }
  thead th { position:sticky; top:0; background:var(--panel);
             text-align:left; padding:6px 8px; font-weight:500;
             border-bottom:1px solid var(--border); cursor:pointer;
             user-select:none; }
  thead th:hover { color:var(--accent); }
  tbody tr { border-bottom:1px solid #2a2a2a; cursor:pointer; }
  tbody tr.selectable:hover { background:#2a2a2a; }
  tbody tr.selected { background:#1e3a5f !important; }
  tbody tr.unselectable { color:var(--muted); cursor:default; }
  td { padding:5px 8px; vertical-align:top; }
  td.num { color:var(--muted); width:28px; text-align:right; }
  td.title { max-width:200px; overflow:hidden; text-overflow:ellipsis;
             white-space:nowrap; }
  td.artist { color:var(--muted); max-width:140px; overflow:hidden;
              text-overflow:ellipsis; white-space:nowrap; }
  td.mono { font-family:ui-monospace, monospace; font-size:12px;
            color:#bbb; }
  td.cue-bad { background:rgba(248,113,113,0.28) !important; color:#f87171; }
  .badge { display:inline-block; padding:1px 6px; border-radius:3px;
           font-size:10px; text-transform:uppercase; font-weight:600; }
  .badge.ok      { background:rgba(74, 222, 128, 0.18); color:var(--ok); }
  .badge.low     { background:rgba(251, 191, 36, 0.18); color:var(--low); }
  .badge.mashup  { background:rgba(129, 140, 248, 0.18); color:var(--mashup); }
  .badge.missing { background:rgba(248, 113, 113, 0.18); color:var(--missing); }

  #right { display:flex; flex-direction:column; overflow:hidden; }
  #header { padding:10px 16px; border-bottom:1px solid var(--border);
            background:var(--panel); }
  #header h2 { margin:0 0 4px 0; font-size:16px; }
  #header .meta { color:var(--muted); font-size:12px; }
  #charts { flex:1; overflow:auto; padding:6px; }
  .chart { height:240px; margin-bottom:6px;
           border:1px solid var(--border); border-radius:4px;
           background:#181818; }
  .chart-label { font-size:11px; color:var(--muted); padding:4px 10px 0; }
  #empty { padding:40px; color:var(--muted); text-align:center; }
</style>
</head>
<body>
<div id="app">
  <div id="left">
    <div id="search">
      <input type="text" id="filter" placeholder="filter by title, artist, # or status…">
    </div>
    <div id="table-wrap">
      <table>
        <thead><tr>
          <th data-sort="num">#</th>
          <th data-sort="title">title</th>
          <th data-sort="artist">artist</th>
          <th data-sort="status">status</th>
          <th data-sort="in_start_hms">in_start</th>
          <th data-sort="in_end_hms">in_end</th>
          <th data-sort="out_start_hms">out_start</th>
          <th data-sort="out_end_hms">out_end</th>
          <th data-sort="confidence">conf</th>
          <th data-sort="bpm_shift">bpm%</th>
          <th data-sort="bpm_stdev">σ%</th>
          <th data-sort="cue1_hms" title="Engine DJ cue1 vs in_start">cue1</th>
          <th data-sort="cue3_hms" title="Engine DJ cue3 vs in_end">cue3</th>
          <th data-sort="cue6_hms" title="Engine DJ cue6 vs out_start">cue6</th>
          <th data-sort="cue8_hms" title="Engine DJ cue8 vs out_end">cue8</th>
        </tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>
  <div id="right">
    <div id="header">
      <h2 id="track-title">— select a track —</h2>
      <div class="meta" id="track-meta"></div>
    </div>
    <div id="charts">
      <div id="empty">Click a row in the table to inspect that track's transition zone.<br>
        The three charts will show the previous, current, and next song's similarity curve
        with markers for in_start, in_end, out_start, and out_end.</div>
    </div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;

function fmtTime(secs) {
  if (secs == null || isNaN(secs)) return '?';
  secs = Math.max(0, secs);
  const h = Math.floor(secs / 3600);
  const m = Math.floor(secs / 60) % 60;
  const s = secs - 60 * (60 * h + m);
  if (h > 0) return `${h}:${String(m).padStart(2,'0')}:${s.toFixed(2).padStart(5,'0')}`;
  return `${m}:${s.toFixed(2).padStart(5,'0')}`;
}

function fmtTickHms(secs, withHours) {
  secs = Math.max(0, secs);
  const h = Math.floor(secs / 3600);
  const m = Math.floor(secs / 60) % 60;
  const s = Math.round(secs) % 60;
  if (withHours || h > 0) return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
  return `${m}:${String(s).padStart(2,'0')}`;
}

function hmsTicks(x0, x1, n) {
  const span = Math.max(0.001, x1 - x0);
  const target = span / Math.max(1, n);
  const candidates = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
  const step = candidates.find(c => c >= target) ?? candidates[candidates.length - 1];
  const first = Math.ceil(x0 / step) * step;
  const vals = [];
  for (let v = first; v <= x1 + step / 2; v += step) vals.push(v);
  const useH = x1 >= 3600;
  return [vals, vals.map(v => fmtTickHms(v, useH))];
}

// ── Table render + filter + sort ────────────────────────────────────────
let sortKey = 'num';
let sortDir = 1;
let filterText = '';

function renderTable() {
  const tbody = document.getElementById('tbody');
  const q = filterText.toLowerCase();
  const rows = DATA.rows.filter(r => {
    if (!q) return true;
    return [r.num, r.title, r.artist, r.status]
      .some(v => String(v).toLowerCase().includes(q));
  });
  rows.sort((a, b) => {
    let av = a[sortKey], bv = b[sortKey];
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) { av = an; bv = bn; }
    if (av < bv) return -sortDir;
    if (av > bv) return  sortDir;
    return 0;
  });
  tbody.innerHTML = rows.map(r => `
    <tr class="${r.selectable ? 'selectable' : 'unselectable'}" data-num="${r.num}">
      <td class="num">${r.num}</td>
      <td class="title" title="${r.title.replace(/"/g,'&quot;')}">${r.title}</td>
      <td class="artist" title="${r.artist.replace(/"/g,'&quot;')}">${r.artist}</td>
      <td><span class="badge ${r.status_class}">${r.status}</span></td>
      <td class="mono">${r.in_start_hms}</td>
      <td class="mono">${r.in_end_hms}</td>
      <td class="mono">${r.out_start_hms}</td>
      <td class="mono">${r.out_end_hms}</td>
      <td class="mono">${r.confidence}</td>
      <td class="mono">${r.bpm_shift}</td>
      <td class="mono">${r.bpm_stdev}</td>
      <td class="mono${r.cue1_ok === false ? ' cue-bad' : ''}" title="cue1 vs in_start">${r.cue1_hms || '—'}</td>
      <td class="mono${r.cue3_ok === false ? ' cue-bad' : ''}" title="cue3 vs in_end">${r.cue3_hms || '—'}</td>
      <td class="mono${r.cue6_ok === false ? ' cue-bad' : ''}" title="cue6 vs out_start">${r.cue6_hms || '—'}</td>
      <td class="mono${r.cue8_ok === false ? ' cue-bad' : ''}" title="cue8 vs out_end">${r.cue8_hms || '—'}</td>
    </tr>`).join('');

  document.querySelectorAll('tbody tr.selectable').forEach(tr => {
    tr.addEventListener('click', () => selectTrack(parseInt(tr.dataset.num, 10)));
  });
}

document.querySelectorAll('th[data-sort]').forEach(th => {
  th.addEventListener('click', () => {
    const k = th.dataset.sort;
    if (sortKey === k) sortDir = -sortDir; else { sortKey = k; sortDir = 1; }
    renderTable();
  });
});
document.getElementById('filter').addEventListener('input', e => {
  filterText = e.target.value;
  renderTable();
});

// ── Right pane: three charts (prev / current / next) ────────────────────
function buildChart(divId, label, curveData, windowCenter, windowSecs, role) {
  if (!curveData) {
    document.getElementById(divId).innerHTML =
      `<div class='chart-label'>${label}: (none)</div>`;
    return;
  }
  const x0 = Math.max(0, windowCenter - windowSecs);
  const x1 = Math.min(DATA.set_duration, windowCenter + windowSecs);

  // Slice curve to window for performance
  const xAll = curveData.x, yAll = curveData.y;
  const xs = [], ys = [];
  for (let i = 0; i < xAll.length; i++) {
    if (xAll[i] >= x0 && xAll[i] <= x1) { xs.push(xAll[i]); ys.push(yAll[i]); }
  }
  const customLabels = xs.map(s => fmtTickHms(s, DATA.set_duration >= 3600));

  const colorByRole = { prev: '#ff7f0e', current: '#1f77b4', next: '#2ca02c' };
  const dashByRole  = { prev: 'dot',     current: 'solid',  next: 'dash' };
  const lineColor = colorByRole[role] || '#1f77b4';

  const traces = [{
    x: xs, y: ys, customdata: customLabels,
    mode: 'lines',
    line: { color: lineColor, width: 2, dash: dashByRole[role] || 'solid' },
    hovertemplate:
      'set time=%{customdata}<br>' +
      'similarity=%{y:.3f}<extra></extra>',
    name: label,
  }];

  // Per-segment BPM-shift line on secondary y-axis
  if (curveData.seg_x && curveData.seg_x.length) {
    const sx = [], sy = [];
    for (let i = 0; i < curveData.seg_x.length; i++) {
      if (curveData.seg_x[i] >= x0 && curveData.seg_x[i] <= x1) {
        sx.push(curveData.seg_x[i]); sy.push(curveData.seg_y[i]);
      }
    }
    if (sx.length) {
      traces.push({
        x: sx, y: sy,
        customdata: sx.map(s => fmtTickHms(s, DATA.set_duration >= 3600)),
        mode: 'lines+markers', yaxis: 'y2',
        line: { color: '#9467bd', width: 1.2 },
        marker: { size: 4, color: '#9467bd' },
        hovertemplate:
          'set time=%{customdata}<br>bpm shift=%{y:+.2f}%<extra></extra>',
        showlegend: false,
      });
    }
  }

  const shapes = [];
  // Markers: in_start / in_end / out_start / out_end + lock.
  // Hard cuts collapse a transition window to one instant — for those we draw
  // only the *_end marker so the chart doesn't show two near-identical lines.
  const markerSpecs = [];
  if (!curveData.hardcut_in && curveData.in_start !== curveData.in_end) {
    markerSpecs.push({ val: curveData.in_start, color: '#4ade80', label: 'in_start' });
  }
  markerSpecs.push({ val: curveData.in_end, color: '#fbbf24', label: 'in_end' });
  if (!curveData.hardcut_out && curveData.out_start !== curveData.out_end) {
    markerSpecs.push({ val: curveData.out_start, color: '#fb923c', label: 'out_start' });
  }
  markerSpecs.push({ val: curveData.out_end, color: '#f87171', label: 'out_end' });
  markerSpecs.forEach(m => {
    if (m.val == null || m.val < x0 || m.val > x1) return;
    shapes.push({
      type: 'line', x0: m.val, x1: m.val, y0: 0, y1: 1, yref: 'y',
      line: { color: m.color, width: 1.5, dash: 'dash' },
    });
  });
  // Engine DJ hotcue lines (dotted, thinner, white/cyan)
  const cueSpecs = [
    { val: curveData.cue1, label: 'cue1 (in_start)' },
    { val: curveData.cue3, label: 'cue3 (in_end)' },
    { val: curveData.cue6, label: 'cue6 (out_start)' },
    { val: curveData.cue8, label: 'cue8 (out_end)' },
  ];
  cueSpecs.forEach(c => {
    if (c.val == null || c.val < x0 || c.val > x1) return;
    shapes.push({
      type: 'line', x0: c.val, x1: c.val, y0: 0, y1: 1, yref: 'y',
      line: { color: '#38bdf8', width: 1, dash: 'dot' },
    });
  });
  // Threshold lines
  shapes.push({
    type: 'line', x0: x0, x1: x1, y0: DATA.played_threshold, y1: DATA.played_threshold,
    line: { color: '#666', width: 1, dash: 'dot' },
  });
  shapes.push({
    type: 'line', x0: x0, x1: x1, y0: DATA.onset_threshold, y1: DATA.onset_threshold,
    line: { color: '#444', width: 1, dash: 'dot' },
  });

  const [tickVals, tickLabels] = hmsTicks(x0, x1, 8);

  const layout = {
    title: { text: label, font: { size: 12, color: '#bbb' }, x: 0.01, y: 0.98 },
    xaxis: {
      range: [x0, x1], tickmode: 'array', tickvals: tickVals, ticktext: tickLabels,
      gridcolor: '#2a2a2a', color: '#aaa',
    },
    yaxis: {
      range: [0, 1], gridcolor: '#2a2a2a', color: '#aaa',
      title: { text: 'similarity', font: { size: 10, color: '#aaa' } },
    },
    yaxis2: {
      overlaying: 'y', side: 'right', showgrid: false,
      color: '#9467bd', tickformat: '+.2f',
      title: { text: 'bpm %', font: { size: 10, color: '#9467bd' } },
    },
    shapes: shapes,
    margin: { l: 50, r: 55, t: 24, b: 28 },
    paper_bgcolor: '#181818', plot_bgcolor: '#181818',
    font: { color: '#bbb' },
    showlegend: false,
  };
  Plotly.react(divId, traces, layout, { responsive: true, displaylogo: false });
}

function selectTrack(num) {
  document.querySelectorAll('tbody tr').forEach(tr => tr.classList.remove('selected'));
  document.querySelectorAll(`tbody tr[data-num="${num}"]`).forEach(tr =>
    tr.classList.add('selected'));

  const cur = DATA.curves[String(num)];
  if (!cur) return;
  const idx = DATA.detectable_nums.indexOf(num);
  const prevNum = idx > 0 ? DATA.detectable_nums[idx - 1] : null;
  const nextNum = idx < DATA.detectable_nums.length - 1 ? DATA.detectable_nums[idx + 1] : null;
  const prev = prevNum != null ? DATA.curves[String(prevNum)] : null;
  const next = nextNum != null ? DATA.curves[String(nextNum)] : null;

  // Show transition window: from in_start - W to out_end + W (covers both ends)
  const w = DATA.transition_window_secs;
  const center = cur.lock ?? cur.in_start ?? 0;

  document.getElementById('track-title').textContent =
    `#${num}  ${cur.title} — ${cur.artist}`;
  const conf = cur.confidence != null ? cur.confidence.toFixed(3) : '—';
  const inDesc = cur.hardcut_in
    ? `cut @ ${fmtTime(cur.in_end)}`
    : `[${fmtTime(cur.in_start)} → ${fmtTime(cur.in_end)}]`;
  const outDesc = cur.hardcut_out
    ? `cut @ ${fmtTime(cur.out_end)}`
    : `[${fmtTime(cur.out_start)} → ${fmtTime(cur.out_end)}]`;
  document.getElementById('track-meta').innerHTML =
    `lock @ ${fmtTime(cur.lock)} · conf ${conf} · ` +
    `in ${inDesc} · out ${outDesc}`;

  const charts = document.getElementById('charts');
  charts.innerHTML = `
    <div class="chart" id="chart-prev"></div>
    <div class="chart" id="chart-cur"></div>
    <div class="chart" id="chart-next"></div>`;

  // Prev track is shown around the CURRENT track's in_start (the prev's out)
  const prevCenter = cur.in_start ?? center;
  buildChart('chart-prev',
    prev ? `← previous: #${prevNum} ${prev.title}` : '← previous: —',
    prev, prevCenter, w, 'prev');
  buildChart('chart-cur',
    `■ current: #${num} ${cur.title}`,
    cur, center, w, 'current');
  // Next track is shown around the CURRENT track's out_start (the next's in)
  const nextCenter = cur.out_start ?? center;
  buildChart('chart-next',
    next ? `→ next: #${nextNum} ${next.title}` : '→ next: —',
    next, nextCenter, w, 'next');
}

// Initial render — table populated, no track selected
renderTable();
// Auto-select the first detectable track for convenience
if (DATA.detectable_nums.length) selectTrack(DATA.detectable_nums[0]);
</script>
</body>
</html>"""
    html = html.replace("__PAYLOAD__", json.dumps(payload))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Review UI     : {out_path}")


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

        # Search across tempo ratios — coarse sweep then fine refinement.
        max_pct = TEMPO_OVERRIDES.get(t.number, DEFAULT_TEMPO_PCT)
        best = search_tempo_two_stage(
            set_chroma, ref_chroma_orig, max_pct, s0_frame, s1_frame
        )

        # Retry with wide range if confidence too low
        if (best is None or best["confidence"] < MIN_CONFIDENCE) \
                and t.number not in TEMPO_OVERRIDES:
            prev_conf = f"{best['confidence']:.2f}" if best else "n/a"
            print(f"  Low confidence ({prev_conf}), "
                  f"retrying with ±{WIDE_TEMPO_PCT}% tempo …")
            best = search_tempo_two_stage(
                set_chroma, ref_chroma_orig, WIDE_TEMPO_PCT, s0_frame, s1_frame
            )

        if best is None:
            print("  ERROR: search returned no result")
            continue

        # Refine to audible region (ignore intro silence / pre-hotcue audio)
        ref_chroma_at_best = warp_chroma(ref_chroma_orig, best["ratio"])

        # Per-segment ratio refinement — captures slow speed changes within a
        # track so the played-curve stays sharp through long mixes.
        if SEGMENT_REFINE_ENABLED:
            seg_centers, seg_ratios, ref_chroma_at_best = refine_tempo_per_segment(
                set_chroma, ref_chroma_orig,
                global_offset=best["offset"],
                global_ratio=best["ratio"],
                frames_per_sec=frames_per_sec,
            )
            t.segment_centers_secs = seg_centers
            t.segment_ratios       = seg_ratios

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
        t.lock_offset_frames  = int(best["offset"])
        t.ref_chroma_orig     = ref_chroma_orig

        bpm_shift = (best["ratio"] - 1.0) * 100
        flag      = "✓" if t.confidence >= MIN_CONFIDENCE else "?"
        if t.segment_ratios is not None and len(t.segment_ratios) > 1:
            seg_pcts = (t.segment_ratios - 1.0) * 100
            seg_info = (f"   seg_shift=[{seg_pcts.min():+.2f}%, "
                        f"{seg_pcts.max():+.2f}%] "
                        f"σ={seg_pcts.std():.2f}%")
        else:
            seg_info = ""
        print(f"  {flag} Start: {fmt_time(t.detected_start_secs)}   "
              f"End: {fmt_time(t.detected_end_secs)}   "
              f"conf={t.confidence:.3f}   "
              f"bpm_shift={bpm_shift:+.2f}%{seg_info}")

        last_start_secs = t.detected_start_secs

    # ── Resolve transition timestamps ───────────────────────────────────────
    print("\nResolving transition windows …")
    resolve_transitions(tracks, frames_per_sec)

    # ── Align BPMs across each transition ───────────────────────────────────
    if TRANSITION_BPM_SYNC_ENABLED:
        print("Aligning transition BPMs (shared ramp per pair) …")
        align_transition_bpms(tracks, set_chroma, frames_per_sec)
    # Free the per-track original chroma — heatmap / CSV / UI don't need it.
    for t in tracks:
        t.ref_chroma_orig = None

    # ── Outputs ─────────────────────────────────────────────────────────────
    print("\nWriting outputs …")
    print("Reading Engine DJ hotcues …")
    read_engine_hotcues(tracks, ENGINE_DB_PATH)
    write_tracklist(tracks, OUTPUT_TRACKLIST)
    write_heatmap(tracks, set_duration, OUTPUT_HEATMAP)
    write_interactive_transitions(tracks, set_duration, OUTPUT_INTERACTIVE)
    write_review_ui(tracks, set_duration, OUTPUT_REVIEW_UI)


if __name__ == "__main__":
    main()
