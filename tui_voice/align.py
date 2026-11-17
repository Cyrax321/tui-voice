"""CTC forced alignment with facebook/wav2vec2-lv-60-espeak-cv-ft (alignment only; none of
its weights go into the TTS model). Needs torch + torchaudio + transformers. NOT run in the
authoring sandbox (no torch); first real test is Gate 2 in Colab."""
import numpy as np

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
W2V_STRIDE_S = 0.02


class Aligner:
    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.torch = torch
        self.device = device
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.proc.tokenizer.pad_token_id

    def label_id(self, ph):
        """Map one eSpeak phone to this model's label id; fall back to stripped forms."""
        for cand in (ph, ph.replace("ː", ""), ph.replace("ʰ", "").replace("ʱ", ""), ph[:1]):
            if cand in self.vocab:
                return self.vocab[cand]
        return None

    def phone_onsets(self, wav16k: np.ndarray, phones):
        """Never raises: any failure returns (None, 0.0, n_unknown) so one bad clip cannot stop a run."""
        try:
            return self._phone_onsets(wav16k, phones)
        except Exception:
            return None, 0.0, len(phones)

    def _phone_onsets(self, wav16k: np.ndarray, phones):
        """Returns (onsets_seconds list, mean_confidence, n_unknown). phones: list of eSpeak phones."""
        import torchaudio.functional as AF
        torch = self.torch
        ids = [self.label_id(p) for p in phones]
        n_unknown = sum(i is None for i in ids)
        keep = [(k, i) for k, i in enumerate(ids) if i is not None]
        if not keep:
            return None, 0.0, n_unknown
        x = self.proc(wav16k, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            logp = torch.log_softmax(self.model(x).logits, dim=-1)
        targets = torch.tensor([[i for _, i in keep]], dtype=torch.int32, device=self.device)
        if logp.shape[1] < targets.shape[1] * 1:
            return None, 0.0, n_unknown
        ali, scores = AF.forced_align(logp, targets, blank=self.blank)
        ali, scores = ali[0].cpu().numpy(), scores[0].exp().cpu().numpy()
        # first frame of each target occurrence (CTC emits a label at the start of the phone)
        onsets, conf, pos, prev = [], [], -1, self.blank
        for t, lab in enumerate(ali):
            if lab != self.blank and (lab != prev or prev == self.blank):
                pos += 1
                onsets.append(t * W2V_STRIDE_S)
                conf.append(float(scores[t]))
            prev = lab
        # repeated identical labels need a blank between them in CTC, so the rule above is exact
        assert len(onsets) == len(keep), (len(onsets), len(keep))
        full = [None] * len(phones)
        for (k, _), o in zip(keep, onsets):
            full[k] = o
        # unknown phones: interpolate between neighbours
        known = [k for k, o in enumerate(full) if o is not None]
        for k in range(len(full)):
            if full[k] is None:
                l = max([j for j in known if j < k], default=None)
                r = min([j for j in known if j > k], default=None)
                full[k] = full[l] if r is None else full[r] if l is None else (full[l] + full[r]) / 2
        return full, float(np.mean(conf)), n_unknown
