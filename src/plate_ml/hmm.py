"""Template-constrained character HMM, using OCR posteriors as pseudo-emissions.

This is a discriminative sequence postprocessor, not an image-generating HMM.
The templates describe research formats, not exhaustive registration rules.
"""
import numpy as np

from .heuristics import DEFAULT_TEMPLATES


class VietnamesePlateHMM:
    def __init__(self, templates=DEFAULT_TEMPLATES, smoothing=1.0, max_log_penalty=0.35):
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be positive")
        if not np.isfinite(max_log_penalty) or max_log_penalty < 0:
            raise ValueError("max_log_penalty must be nonnegative")
        self.templates, self.smoothing = templates, smoothing
        self.max_log_penalty = max_log_penalty
        self.alphabet_ = np.asarray(sorted(set(''.join(a for t in templates for a in t.positions))))
        self.initial_ = np.full(len(self.alphabet_), smoothing)
        self.transitions_ = np.full((len(self.alphabet_), len(self.alphabet_)), smoothing)
        self.training_sequences_ = 0

    def fit(self, texts):
        """Estimate smoothed character bigrams from training plate text only.

        Repeated plate identities count once. No validation/test text is needed.
        Uniform priors remain usable for the notebook's synthetic examples.
        """
        self.initial_.fill(self.smoothing)
        self.transitions_.fill(self.smoothing)
        self.training_sequences_ = 0
        lookup = {c: i for i, c in enumerate(self.alphabet_)}
        for text in sorted(set(texts)):
            if not text or any(c not in lookup for c in text):
                continue
            indices = [lookup[c] for c in text]
            self.initial_[indices[0]] += 1
            for a, b in zip(indices, indices[1:]):
                self.transitions_[a, b] += 1
            self.training_sequences_ += 1
        return self

    def decode(self, probabilities, classes, row_lengths, templates=None):
        p, classes = np.asarray(probabilities, dtype=float), np.asarray(classes)
        if p.ndim != 2 or p.shape[1] != len(classes) or len(set(classes)) != len(classes):
            raise ValueError("Probability dimensions/classes are invalid")
        if any(n <= 0 for n in row_lengths) or sum(row_lengths) != len(p):
            raise ValueError("Row lengths do not match characters")
        if not np.isfinite(p).all() or (p < 0).any() or (p.sum(axis=1) <= 0).any():
            raise ValueError("Expected finite nonnegative probability rows with positive mass")
        if not len(p):
            return dict(raw='', text='', template=None, changed=False, mean_log_penalty=0.0)
        p = p / p.sum(axis=1, keepdims=True)
        logp = np.log(np.maximum(p, 1e-12))
        raw_indices = p.argmax(axis=1)
        raw = ''.join(classes[raw_indices])
        raw_score = float(logp[np.arange(len(p)), raw_indices].sum())
        lookup = {c: i for i, c in enumerate(self.alphabet_)}
        options = []
        for template in self.templates if templates is None else templates:
            if tuple(row_lengths) != template.rows or len(p) != len(template.positions):
                continue
            valid = [np.flatnonzero(np.isin(classes, list(a))) for a in template.positions]
            if any(not len(v) for v in valid):
                continue
            states = [[lookup[classes[j]] for j in v] for v in valid]
            initial = self.initial_[states[0]]
            scores = logp[0, valid[0]] + np.log(initial / initial.sum())
            back = []
            for t in range(1, len(p)):
                transition = self.transitions_[np.ix_(states[t-1], states[t])]
                transition = transition / transition.sum(axis=1, keepdims=True)
                paths = scores[:, None] + np.log(transition)
                parent = paths.argmax(axis=0)
                scores = paths[parent, np.arange(len(valid[t]))] + logp[t, valid[t]]
                back.append(parent)
            end = int(scores.argmax())
            path = [end]
            for parent in reversed(back):
                path.append(int(parent[path[-1]]))
            indices = [valid[t][j] for t, j in enumerate(reversed(path))]
            emission_score = float(logp[np.arange(len(p)), indices].sum())
            options.append((float(scores[end]), template.name, ''.join(classes[indices]), emission_score))
        if not options:
            return dict(raw=raw, text=raw, template=None, changed=False, mean_log_penalty=0.0)
        _, name, text, emission_score = max(options)
        penalty = (raw_score - emission_score) / len(p)
        accepted = penalty <= self.max_log_penalty
        return dict(raw=raw, text=text if accepted else raw, template=name if accepted else None,
                    changed=accepted and text != raw, mean_log_penalty=penalty)

    __call__ = decode


def decode_plate_hmm(probabilities, classes, row_lengths, templates=DEFAULT_TEMPLATES):
    return VietnamesePlateHMM(templates).decode(probabilities, classes, row_lengths)
