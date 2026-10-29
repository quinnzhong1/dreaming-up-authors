"""Author-name matching and hallucination metrics for the Section 7 stages.

``NameSplitter`` and ``NameMatcher`` are distilled from the module 06 evaluation
code (``06_llm_recall_hardcoded_eval/name_matcher.py``) and handle reversed, bibliographic, unicode, particle, concatenated-initial
and suffix forms.

Match levels, strongest first: L1 exact, L2 unicode-normalised, L3 alpha-only,
L3.5 first and last, L4 surname and first initial, L5 surname only. L1-L4 count
as correct; L5 counts the ground-truth author as missed and the LLM name as
hallucinated.

    matcher = NameMatcher()
    result  = score_authors(gt_str, llm_str, matcher)
    is_hal  = per_name_is_hallucinated(name, gt_str, matcher)
"""

import re
from typing import List, Set, Dict, Optional, Tuple

from nameparser import HumanName
from unidecode import unidecode


# ─────────────────────────────────────────────────────────────────────────────
# Threshold
# ─────────────────────────────────────────────────────────────────────────────

MATCH_LEVEL_THRESHOLD = 4   # L4 and below = strong match


# ─────────────────────────────────────────────────────────────────────────────
# NameSplitter
# ─────────────────────────────────────────────────────────────────────────────

class NameSplitter:
    """Split a raw author string into full names, correcting reversed and
    bibliographic ordering."""

    def __init__(self):
        self.surname_particles: Set[str] = {
            'van', 'von', 'de', 'der', 'den', 'del', 'della', 'di', 'da',
            'do', 'dos', 'du', 'la', 'le', 'van der', 'van den', 'von der',
        }
        self.name_suffixes: Set[str] = {
            'jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'v',
            'phd', 'ph.d', 'ph.d.', 'md', 'm.d', 'm.d.', 'esq', 'esq.',
        }

    def _is_initial(self, word: str) -> bool:
        return len(word) == 2 and word[1] == '.' and word[0].isalpha() and word[0].isupper()

    def _looks_like_initial_plus_surname(self, segment: str) -> bool:
        words = segment.strip().split()
        if len(words) < 2:
            return False
        for i in range(len(words) - 1):
            if not self._is_initial(words[i]):
                return False
        last = words[-1]
        return not self._is_initial(last) and last.lower() not in self.surname_particles

    def _detect_reversed_format(self, segments: List[str]) -> bool:
        """Detect alternating Surname, Firstname format (stricter rules)."""
        if len(segments) < 2 or len(segments) % 2 != 0:
            return False

        single_word_even = 0
        single_word_odd  = 0
        has_initials     = False

        for i in range(0, len(segments), 2):
            words = segments[i].strip().split()
            for w in words:
                if self._is_initial(w):
                    has_initials = True
            non_particle = [w for w in words if w.lower() not in self.surname_particles]
            if len(non_particle) == 1:
                single_word_even += 1
            if len(non_particle) > 1:
                return False

        for i in range(1, len(segments), 2):
            seg   = segments[i].strip()
            words = seg.split()
            for w in words:
                if self._is_initial(w):
                    has_initials = True
            if self._looks_like_initial_plus_surname(seg):
                return False
            if len(words) == 1:
                single_word_odd += 1
            if len(words) > 1:
                has_non_initial = any(not self._is_initial(w) and w.lower() not in self.surname_particles
                                      for w in words)
                if has_non_initial and any(self._is_initial(w) for w in words):
                    return False

        n = len(segments) // 2
        if has_initials:
            return single_word_even == n and single_word_odd == n
        return (single_word_even / n) >= 0.75 and (single_word_odd / n) >= 0.75

    def _parse_reversed_format(self, segments: List[str]) -> List[str]:
        return [f"{segments[i+1]} {segments[i]}" for i in range(0, len(segments), 2)]

    def _detect_lastname_firstname_middle_format(self, segments: List[str]) -> bool:
        """Detect bibliographic format: LastName FirstName Initial(s)."""
        if len(segments) < 2:
            return False
        biblio_count = 0
        normal_count = 0
        for seg in segments:
            words = seg.strip().split()
            if len(words) < 2:
                continue
            first_word = words[0]
            last_word  = words[-1]
            last_is_initials = (
                (len(last_word) == 2 and last_word[1] == '.' and last_word[0].isalpha()) or
                bool(re.match(r'^[A-Z](\.[A-Z])+\.?$', last_word))
            )
            first_is_namelike = (
                not self._is_initial(first_word)
                and '.' not in first_word
                and len(first_word) > 2
            )
            if first_is_namelike and last_is_initials:
                biblio_count += 1
            elif not last_is_initials:
                normal_count += 1

        total = len(segments)
        return (biblio_count / total) >= 0.6 and (normal_count / total) < 0.2

    def _correct_lastname_firstname_middle_format(self, segments: List[str]) -> List[str]:
        corrected = []
        for seg in segments:
            words = seg.strip().split()
            if len(words) < 2:
                corrected.append(seg)
                continue
            surname_parts = []
            given_start   = 0
            for i, word in enumerate(words):
                if i == 0 and word.lower() in self.surname_particles:
                    surname_parts.append(word)
                elif i > 0 and surname_parts and word.lower() not in self.surname_particles:
                    surname_parts.append(word)
                    given_start = i + 1
                    break
                elif i == 0:
                    surname_parts.append(word)
                    given_start = 1
                    break
            if given_start < len(words):
                given   = ' '.join(words[given_start:])
                surname = ' '.join(surname_parts)
                corrected.append(f"{given} {surname}")
            else:
                corrected.append(seg)
        return corrected

    def _merge_suffixes(self, segments: List[str]) -> List[str]:
        if len(segments) <= 1:
            return segments
        merged = []
        i = 0
        while i < len(segments):
            if i + 1 < len(segments) and segments[i + 1].strip().lower() in self.name_suffixes:
                merged.append(f"{segments[i]}, {segments[i+1].strip()}")
                i += 2
            else:
                merged.append(segments[i])
                i += 1
        return merged

    def split(self, author_string: str) -> List[str]:
        """
        Split a raw author string into individual names.

        If semicolons are present they are used as the author separator first
        (each semicolon-delimited chunk is then processed independently with
        the comma-based format-detection pipeline).  This avoids reversed-format
        false positives when the model uses 'First Last; First Last' output.

        Falls back to comma-only splitting when no semicolons are present.
        """
        if not author_string or not author_string.strip():
            return []

        # ── Semicolon-separated output (preferred, instructed in prompt) ──────
        if ';' in author_string:
            raw_chunks = [c.strip() for c in author_string.split(';') if c.strip()]
            names = []
            for chunk in raw_chunks:
                # Each chunk should be one name; run through comma pipeline
                # in case a chunk itself has internal comma (e.g. Jr. suffix)
                chunk_names = self._split_comma_chunk(chunk)
                names.extend(chunk_names)
            return names

        # ── Comma-only fallback (model ignored instruction) ───────────────────
        return self._split_comma_chunk(author_string)

    def _split_comma_chunk(self, author_string: str) -> List[str]:
        """Comma-based split with reversed/bibliographic format detection."""
        processed = re.sub(r'\s+and\s+', ', ', author_string)
        segments  = [s.strip() for s in processed.split(',')
                     if s.strip() and s.strip() != '...']
        segments  = self._merge_suffixes(segments)
        if self._detect_lastname_firstname_middle_format(segments):
            return self._correct_lastname_firstname_middle_format(segments)
        if self._detect_reversed_format(segments):
            return self._parse_reversed_format(segments)
        return segments


# ─────────────────────────────────────────────────────────────────────────────
# NameMatcher
# ─────────────────────────────────────────────────────────────────────────────

class NameMatcher:
    """
    Hierarchical author-name matcher with unicode normalization, particle
    stripping, and first/last swap rescue strategies.
    """

    def __init__(self):
        self.splitter = NameSplitter()
        self.surname_particles: Set[str] = {
            'van', 'von', 'de', 'der', 'den', 'del', 'della', 'di', 'da',
            'do', 'dos', 'du', 'la', 'le', 'van der', 'van den', 'von der',
            'van de', 'von dem', 'de la', 'de las', 'de los',
        }

    def _strip_particles(self, lastname: str) -> str:
        if not lastname:
            return ''
        words = lastname.lower().split()
        while words and words[0] in self.surname_particles:
            words.pop(0)
        i = 0
        while i < len(words):
            if i + 1 < len(words) and f"{words[i]} {words[i+1]}" in self.surname_particles:
                words.pop(i); words.pop(i)
            elif words[i] in self.surname_particles:
                words.pop(i)
            else:
                i += 1
        return ' '.join(words)

    def _parse_single_name(self, name_str: str) -> Dict:
        original    = name_str.strip()
        preprocessed = re.sub(r'([A-Z])\.([A-Z])', r'\1. \2', original)
        parsed      = HumanName(preprocessed)
        norm_full   = unidecode(str(parsed))
        norm_parsed = HumanName(norm_full)
        norm_alpha  = ' '.join(''.join(c for c in norm_full if c.isalpha() or c.isspace()).split())
        na_parsed   = HumanName(norm_alpha)
        return {
            'original':       original,
            'normalized':     norm_full,
            'normalized_alpha': norm_alpha.lower(),
            'parsed_alpha': {
                'first':        na_parsed.first.lower(),
                'middle':       na_parsed.middle.lower(),
                'last':         na_parsed.last.lower(),
                'last_stripped': self._strip_particles(na_parsed.last.lower()),
            },
        }

    def _get_match(self, gt: Dict, llm: Dict, swapped: bool = False) -> Optional[Tuple[str, str]]:
        sfx = '_swapped' if swapped else ''

        if gt['original'] == llm['original']:
            return f'L1_exact_original{sfx}', llm['original']
        if gt['normalized'] == llm['normalized']:
            return f'L2_normalized_full{sfx}', llm['original']
        if gt['normalized_alpha'] == llm['normalized_alpha']:
            return f'L3_normalized_alpha{sfx}', llm['original']

        gt_last  = gt['parsed_alpha']['last'];  llm_last  = llm['parsed_alpha']['last']
        gt_ls    = gt['parsed_alpha']['last_stripped']
        llm_ls   = llm['parsed_alpha']['last_stripped']
        gt_first = gt['parsed_alpha']['first']; llm_first = llm['parsed_alpha']['first']

        last_match = (gt_last and llm_last and gt_last == llm_last) or \
                     (gt_ls and llm_ls and gt_ls == llm_ls)
        if not last_match:
            return None

        if gt_first and llm_first and gt_first == llm_first:
            return f'L3.5_full_component{sfx}', llm['original']
        if gt_first and llm_first and gt_first[0] == llm_first[0]:
            return f'L4_lastname_initial{sfx}', llm['original']

        # L5 weak — block if initials contradict
        if gt_first and llm_first:
            gt_ini = len(gt_first) == 1; llm_ini = len(llm_first) == 1
            if (gt_ini or llm_ini) and gt_first[0] != llm_first[0]:
                return None
        return f'L5_lastname_only{sfx}', llm['original']

    def _create_swapped_name(self, parsed: Dict) -> Dict:
        first = parsed['parsed_alpha']['first']
        mid   = parsed['parsed_alpha']['middle']
        last  = parsed['parsed_alpha']['last']
        new_last = ' '.join(p for p in [first, mid] if p)
        return {**parsed, 'parsed_alpha': {
            'first': last, 'middle': '',
            'last': new_last,
            'last_stripped': self._strip_particles(new_last),
        }}

    def _create_bibliographic_name(self, parsed: Dict) -> Dict:
        first = parsed['parsed_alpha']['first']
        mid   = parsed['parsed_alpha']['middle']
        last  = parsed['parsed_alpha']['last']
        parts = [p for p in [mid, last] if p]
        if not parts:
            return parsed
        return {**parsed, 'parsed_alpha': {
            'first': parts[0],
            'middle': ' '.join(parts[1:]),
            'last': first,
            'last_stripped': self._strip_particles(first),
        }}

    def find_matches(self, gt_authors_list: List[str], llm_authors_str: str) -> Dict:
        """
        Match each GT author against parsed LLM output.

        Returns dict with:
          matches        — list of {gt_author, gt_index, llm_author, match_level}
          unmatched_gt   — list of {gt_author, gt_index}
          hallucinated_llm — list of unmatched LLM author strings
        """
        gt_parsed  = [{**self._parse_single_name(n), 'gt_index': i}
                      for i, n in enumerate(gt_authors_list)]
        llm_split  = self.splitter.split(llm_authors_str)
        llm_pool   = [{**self._parse_single_name(n), 'llm_index': i}
                      for i, n in enumerate(llm_split)]

        matches      = []
        unmatched_gt = []

        for gt in gt_parsed:
            best_result = None
            best_level  = 10.0
            best_idx    = None

            for llm in llm_pool:
                for attempt_fn, swapped in [
                    (lambda l: self._get_match(gt, l, swapped=False), False),
                    (lambda l: self._get_match(gt, self._create_swapped_name(l), swapped=True), True),
                    (lambda l: self._get_match(gt, self._create_bibliographic_name(l), swapped=False), False),
                ]:
                    res = attempt_fn(llm)
                    if res:
                        level = float(res[0].split('_')[0][1:])
                        if level < best_level:
                            best_level  = level
                            best_result = res
                            best_idx    = llm['llm_index']
                        break   # don't try weaker rescue if a match was found

            if best_result:
                match_level, llm_original = best_result
                matches.append({
                    'gt_author':   gt['original'],
                    'gt_index':    gt['gt_index'],
                    'llm_author':  llm_original,
                    'match_level': match_level,
                })
                llm_pool = [p for p in llm_pool if p['llm_index'] != best_idx]
            else:
                unmatched_gt.append({'gt_author': gt['original'], 'gt_index': gt['gt_index']})

        return {
            'matches':          matches,
            'unmatched_gt':     unmatched_gt,
            'hallucinated_llm': [p['original'] for p in llm_pool],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring API
# ─────────────────────────────────────────────────────────────────────────────

def _match_level_value(match_level_str: str) -> float:
    """Extract numeric level from e.g. 'L3.5_full_component_swapped' → 3.5"""
    return float(match_level_str.split('_')[0][1:])


def score_authors(gt_str: str, llm_str: str, matcher: NameMatcher) -> dict:
    """Score an LLM author-list string against ground truth.

    X is strong matches (level <= MATCH_LEVEL_THRESHOLD), Y hallucinations and
    Z missed authors, both of which absorb weak (L5) matches. Returns hr2, hr4,
    hr5 and the underlying counts.
    """
    gt_list = matcher.splitter.split(gt_str) if gt_str else []
    results = matcher.find_matches(gt_list, llm_str)

    strong, weak = [], []
    for m in results['matches']:
        (strong if _match_level_value(m['match_level']) <= MATCH_LEVEL_THRESHOLD else weak).append(m)

    n_M = len(strong)
    n_H = len(results['hallucinated_llm']) + len(weak)
    n_U = len(results['unmatched_gt'])     + len(weak)
    denom = n_M + n_H + n_U
    hr2   = (n_H + n_U) / denom          if denom        > 0 else 0.0
    hr4   = n_H / (n_M + n_H)            if (n_M + n_H)  > 0 else 0.0
    hr5   = n_U / (n_M + n_U)            if (n_M + n_U)  > 0 else 0.0

    all_predicted = (
        [m['llm_author'] for m in results['matches']] +
        results['hallucinated_llm']
    )

    return {
        'predicted_authors': ', '.join(all_predicted),
        'hr2':               round(hr2, 6),
        'hr4':               round(hr4, 6),
        'hr5':               round(hr5, 6),
        'n_match':           n_M,
        'n_hal':             n_H,
        'n_unretr':          n_U,
        'is_hallucinated':   int(hr2 > 0.5),
        'n_gt_authors':      len(gt_list),
        'n_pred_authors':    len(matcher.splitter.split(llm_str)) if llm_str else 0,
    }


def per_name_is_hallucinated(name_str: str, gt_str: str, matcher: NameMatcher) -> bool:
    """
    Return True if name_str does not match any author in gt_str at level ≤ threshold.
    Used for labeling per-name P9 embeddings.
    """
    gt_list = matcher.splitter.split(gt_str) if gt_str else []
    results = matcher.find_matches(gt_list, name_str)
    # If any strong match found, name is correct; otherwise hallucinated
    for m in results['matches']:
        if _match_level_value(m['match_level']) <= MATCH_LEVEL_THRESHOLD:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_auroc(scores, labels) -> float:
    """AUROC for binary labels. Returns nan if only one class present."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    scores = np.asarray(scores); labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float('nan')
    return float(roc_auc_score(labels, scores))


def spearman_r(x, y) -> float:
    """Spearman rank correlation, ignoring NaN pairs."""
    import numpy as np
    from scipy.stats import spearmanr
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float('nan')
    r, _ = spearmanr(x[mask], y[mask])
    return float(r)
