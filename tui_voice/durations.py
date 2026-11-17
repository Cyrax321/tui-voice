"""Turn CTC phone onsets into per-token frame durations on tiny-voca's grid
(256 samples at 24 kHz = 10.67 ms). numpy only.

Rules copied from tiny-voca's annotations-v3 notes:
- the wav2vec2 phoneme CTC emits a label at the START of the phone; a phone really starts
  ~19 ms before its first emission frame (ONSET_SHIFT_S)
- a phone lasts from its onset to the next phone's onset
- silence >= 75 ms before a pause-capable token (BOS, EOS, punctuation, word boundary) goes
  to that token; shorter gaps stay with the phone (stop closures)
- durations sum to the frame count exactly and every token has >= 1 frame
"""
import numpy as np

SR, HOP = 24000, 256
FRAME_S = HOP / SR
ONSET_SHIFT_S = 0.019
MIN_PAUSE_S = 0.075
SIL_DB_BELOW_LOUD = 45.0
ONSET_TOL = 3


def num_frames(n_samples: int) -> int:
    return int(np.ceil(n_samples / HOP))        # upstream: math.ceil(len(wave) / hop)


def silence_mask(y: np.ndarray, n_frames: int) -> np.ndarray:
    """True where the frame level is more than 45 dB below the recording's loud level."""
    lvl = np.zeros(n_frames)
    half = HOP
    for i in range(n_frames):
        c = i * HOP
        seg = y[max(0, c - half): c + half]
        lvl[i] = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if len(seg) else -120
    loud = np.percentile(lvl, 95)
    return lvl < (loud - SIL_DB_BELOW_LOUD)


def build_durations(is_phone, phone_onsets_s, total_frames, sil=None):
    """is_phone: list[bool] over ALL tokens (BOS ... EOS), in order.
    phone_onsets_s: onset (seconds, emission time) for each phone token, in order.
    sil: optional bool mask per frame. Returns int array of durations (len = len(is_phone))."""
    n = len(is_phone)
    assert sum(is_phone) == len(phone_onsets_s)
    on = np.zeros(n)
    it = iter(phone_onsets_s)
    ph_on = [None] * n
    for i, p in enumerate(is_phone):
        if p:
            ph_on[i] = max(0.0, next(it) - ONSET_SHIFT_S) / FRAME_S
    # pause tokens inherit the next phone's onset; EOS sits at the end
    nxt = float(total_frames)
    for i in range(n - 1, -1, -1):
        if ph_on[i] is None:
            ph_on[i] = nxt
        else:
            nxt = ph_on[i]
    on = np.array(ph_on, dtype=float)
    on[0] = 0.0                                   # BOS starts the clip
    if sil is not None:
        min_f = int(np.ceil(MIN_PAUSE_S / FRAME_S))
        for i in range(1, n):
            if is_phone[i]:
                continue
            end = int(min(round(on[i]), len(sil)))   # silent run just before the next onset
            o = end
            while end > 0 and not sil[end - 1] and o - end < ONSET_TOL:
                end -= 1                              # onsets can lag the true edge by a few frames
            k = end
            while k > 0 and sil[k - 1]:
                k -= 1
            if end - k >= min_f:
                on[i] = float(k)
    on = np.round(on).astype(int)
    for i in range(1, n):                          # strictly increasing: >= 1 frame each
        on[i] = max(on[i], on[i - 1] + 1)
    for i in range(n - 1, -1, -1):                 # leave room for the tokens after
        on[i] = min(on[i], total_frames - (n - i))
    on = np.maximum.accumulate(on)
    on[0] = 0
    ends = np.append(on[1:], total_frames)
    d = ends - on
    assert d.min() >= 1 and d.sum() == total_frames, (d.min(), d.sum(), total_frames)
    return d
