"""Deterministic fuzzy matching of LLM author names to ground truth.

Ported verbatim from ``name_matcher_v3.NameMatcher``, the version that produced
every evaluated CSV in ``release/responses/``.

Each ground-truth author keeps its strongest match, lower level being stronger:
L1 exact, L2 Unicode-folded, L3 also depunctuated, L3.5 surname and full given
name, L4 surname and first initial, L5 surname only with non-contradictory
given names. Surnames compare with and without leading particles, so ``van
Oudenaarden`` matches ``Oudenaarden``.

Section 4.5's threshold is L1-L4. L5 is produced here and rejected by
:mod:`score_answers`, which counts it as both a hallucination and an
unretrieved author; the cutoff is ``MATCH_LEVEL_THRESHOLD`` in ``common.py``.

Matching is greedy in ground-truth order and consumes matched names by index,
so duplicates on either side are counted one-for-one.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from nameparser import HumanName
from unidecode import unidecode

from name_splitter import NameSplitter

class NameMatcher:
    """Matches author names by the L1-L5 hierarchy described in the module docstring."""
    def __init__(self):
        self.splitter = NameSplitter()
        self.surname_particles: Set[str] = {
            'van', 'von', 'de', 'der', 'den', 'del', 'della', 'di', 'da',
            'do', 'dos', 'du', 'la', 'le', 'van der', 'van den', 'von der',
            'van de', 'von dem', 'de la', 'de las', 'de los'
        }

    def _strip_particles(self, lastname: str) -> str:
        """Strip surname particles: ``van der Meyden`` -> ``meyden``."""
        if not lastname:
            return ''
        
        lastname_lower = lastname.lower()
        words = lastname_lower.split()
        
        # Remove particles from the beginning
        while words and words[0] in self.surname_particles:
            words.pop(0)
        
        # Handle multi-word particles (e.g., "van der")
        i = 0
        while i < len(words):
            # Check if current word combines with next to form a particle
            if i + 1 < len(words):
                two_word = f"{words[i]} {words[i+1]}"
                if two_word in self.surname_particles:
                    words.pop(i)
                    words.pop(i)
                    continue
            # Check single word particle
            if words[i] in self.surname_particles:
                words.pop(i)
            else:
                i += 1
        
        return ' '.join(words)

    def _parse_single_name(self, name_str: str) -> Dict:
        """Parse one name into its normalised forms, including a particle-stripped
        surname and a middle name for the swap rescue.

        Concatenated initials (``W.M.``) are spaced out first so they parse.
        """
        # Original string
        original = name_str.strip()
        
        # Preprocess to handle concatenated initials (e.g., "W.M." -> "W. M.")
        # This regex finds patterns like "W.M." or "J.K." and adds spaces
        preprocessed = re.sub(r'([A-Z])\.([A-Z])', r'\1. \2', original)

        # Parsed version of preprocessed string
        parsed_name = HumanName(preprocessed)

        # Normalized version (handles unicode)
        normalized_full = unidecode(str(parsed_name))
        normalized_parsed = HumanName(normalized_full)

        # Alpha-only version (handles punctuation)
        normalized_alpha = ''.join(c for c in normalized_full if c.isalpha() or c.isspace())
        normalized_alpha = ' '.join(normalized_alpha.split())
        normalized_alpha_parsed = HumanName(normalized_alpha)

        # Particle-stripped last name
        last_stripped = self._strip_particles(normalized_alpha_parsed.last.lower())

        return {
            'original': original,  # Keep the original for tracking
            'normalized': normalized_full,
            'normalized_alpha': normalized_alpha.lower(),
            'parsed_alpha': {
                'first': normalized_alpha_parsed.first.lower(),
                'middle': normalized_alpha_parsed.middle.lower(),
                'last': normalized_alpha_parsed.last.lower(),
                'last_stripped': last_stripped
            }
        }

    def _get_match(self, gt_name: Dict, llm_name: Dict, swapped: bool = False) -> Optional[Tuple[str, str]]:
        """Strongest level at which two parsed names match, or ``None``.

        Returns ``(match_level, llm_original_name)``. Levels are defined in the
        module docstring.
        """
        # Level 1: Exact Original Match
        if gt_name['original'] == llm_name['original']:
            suffix = '_swapped' if swapped else ''
            return f'L1_exact_original{suffix}', llm_name['original']

        # Level 2: Normalized Full Match
        if gt_name['normalized'] == llm_name['normalized']:
            suffix = '_swapped' if swapped else ''
            return f'L2_normalized_full{suffix}', llm_name['original']
            
        # Level 3: Normalized Alpha-Only Match
        if gt_name['normalized_alpha'] == llm_name['normalized_alpha']:
            suffix = '_swapped' if swapped else ''
            return f'L3_normalized_alpha{suffix}', llm_name['original']

        # Prepare for component-based matching
        gt_last = gt_name['parsed_alpha']['last']
        llm_last = llm_name['parsed_alpha']['last']
        gt_last_stripped = gt_name['parsed_alpha']['last_stripped']
        llm_last_stripped = llm_name['parsed_alpha']['last_stripped']
        gt_first = gt_name['parsed_alpha']['first']
        llm_first = llm_name['parsed_alpha']['first']

        # Try matching with particle-stripped last names
        last_names_match = False
        if gt_last and llm_last:
            # First try exact match
            if gt_last == llm_last:
                last_names_match = True
            # Then try particle-stripped match
            elif gt_last_stripped and llm_last_stripped and gt_last_stripped == llm_last_stripped:
                last_names_match = True

        if not last_names_match:
            return None

        # Level 3.5: Full Component Match (Last Name + Full First Name)
        # This catches cases where components match perfectly after normalization/swapping
        if gt_first and llm_first and gt_first == llm_first:
            suffix = '_swapped' if swapped else ''
            return f'L3.5_full_component{suffix}', llm_name['original']

        # Level 4: Last Name and First Initial Match
        if gt_first and llm_first and gt_first[0] == llm_first[0]:
            suffix = '_swapped' if swapped else ''
            return f'L4_lastname_initial{suffix}', llm_name['original']

        # Level 5: Last Name Only Match (Weak match)
        # Check for contradictory initials - but only when at least one is just an initial
        if gt_first and llm_first:
            # Check if either is just an initial (length 1)
            gt_is_initial = len(gt_first) == 1
            llm_is_initial = len(llm_first) == 1
            
            # Only check for contradiction if at least one is an initial
            if (gt_is_initial or llm_is_initial) and gt_first[0] != llm_first[0]:
                # Initial contradiction (e.g., "R. Keefe" vs "Eve Keefe")
                return None
            # Otherwise, different full names are OK for L5 (e.g., "Peter Wurman" vs "Adam Wurman")
        
        # Return L5 match if no contradiction
        suffix = '_swapped' if swapped else ''
        return f'L5_lastname_only{suffix}', llm_name['original']

    def _create_swapped_name(self, parsed_name: Dict) -> Dict:
        """Rescue for ``LastName FirstName`` with no comma.

        Moves the last word to the front: ``van Oudenaarden Alexander`` becomes
        first ``alexander``, last ``van oudenaarden``.
        """
        first = parsed_name['parsed_alpha']['first']
        middle = parsed_name['parsed_alpha']['middle']
        last = parsed_name['parsed_alpha']['last']
        
        # New first name is the old last name
        swapped_first = last
        
        # New last name is everything else (first + middle)
        parts = []
        if first:
            parts.append(first)
        if middle:
            parts.append(middle)
        swapped_last = ' '.join(parts) if parts else ''
        
        # Create new stripped version for the swapped last name
        swapped_last_stripped = self._strip_particles(swapped_last)
        
        return {
            'original': parsed_name['original'],  # Keep original for tracking
            'normalized': parsed_name['normalized'],
            'normalized_alpha': parsed_name['normalized_alpha'],
            'parsed_alpha': {
                'first': swapped_first,
                'middle': '',  # No middle after swap
                'last': swapped_last,
                'last_stripped': swapped_last_stripped
            }
        }
    def _create_bibliographic_name(self, parsed_name: Dict) -> Dict:
        """Rescue for bibliographic order: ``Dimotakis Paul E.`` -> ``Paul E. Dimotakis``.

        The first word is taken as the surname and the rest as given names.
        """
        first = parsed_name['parsed_alpha']['first']
        middle = parsed_name['parsed_alpha']['middle']
        last = parsed_name['parsed_alpha']['last']
        
        # In bibliographic format, the parsed "first" is actually the surname
        biblio_last = first
        
        # Remaining components become given names
        parts = []
        if middle:
            parts.append(middle)
        if last:
            parts.append(last)
        
        # Need at least one given name component
        if not parts:
            return parsed_name
        
        biblio_first = parts[0]
        biblio_middle = ' '.join(parts[1:]) if len(parts) > 1 else ''
        
        # Create stripped version for the new last name
        biblio_last_stripped = self._strip_particles(biblio_last)
        
        return {
            'original': parsed_name['original'],
            'normalized': parsed_name['normalized'],
            'normalized_alpha': parsed_name['normalized_alpha'],
            'parsed_alpha': {
                'first': biblio_first,
                'middle': biblio_middle,
                'last': biblio_last,
                'last_stripped': biblio_last_stripped
            }
        }
        
    def find_matches(self, gt_authors_list: List[str], llm_authors_str: str) -> Dict:
        """Best match for each ground-truth author, consuming the LLM pool by index.

        Each author tries the normal orientation first, then the two rescues,
        and keeps the lowest level found. Returns ``matches``, ``unmatched_gt``,
        and ``hallucinated_llm``.
        """
        # Parse all ground truth authors with indices
        gt_parsed = [
            {**self._parse_single_name(name), 'gt_index': i} 
            for i, name in enumerate(gt_authors_list)
        ]
        
        # Split and parse LLM string with indices
        llm_split = self.splitter.split(llm_authors_str)
        llm_pool = [
            {**self._parse_single_name(name), 'llm_index': i} 
            for i, name in enumerate(llm_split)
        ]
        
        matches = []  # Changed from dict to list
        unmatched_gt = []
        
        # Try to match each GT author
        for gt_author in gt_parsed:
            best_match = None
            best_match_level = 10  # High number for sorting (lower is better)
            best_match_index = None  # Track which LLM author matched
            
            # Try matching against all LLM authors in pool
            for llm_author in llm_pool:
                # Attempt 1: Normal orientation
                match_result = self._get_match(gt_author, llm_author, swapped=False)
                if match_result:
                    level_str = match_result[0].split('_')[0][1:]  # Extract "3.5" from "L3.5"
                    level = float(level_str)
                    if level < best_match_level:
                        best_match_level = level
                        best_match = match_result
                        best_match_index = llm_author['llm_index']
                
                # Attempt 2: Swapped orientation (rescue strategy)
                if not match_result:
                    llm_swapped = self._create_swapped_name(llm_author)
                    match_result = self._get_match(gt_author, llm_swapped, swapped=True)
                    if match_result:
                        level_str = match_result[0].split('_')[0][1:]
                        level = float(level_str)
                        if level < best_match_level:
                            best_match_level = level
                            best_match = match_result
                            best_match_index = llm_author['llm_index']
                
                # Attempt 3: Bibliographic orientation (rescue strategy)
                # Handles "LastName FirstName Initial" format from LLM
                if not match_result:
                    llm_biblio = self._create_bibliographic_name(llm_author)
                    match_result = self._get_match(gt_author, llm_biblio, swapped=False)
                    if match_result:
                        level_str = match_result[0].split('_')[0][1:]
                        level = float(level_str)
                        if level < best_match_level:
                            best_match_level = level
                            # Mark this as bibliographic rescue in the match level
                            match_level, llm_orig = match_result
                            best_match = (f"{match_level}_biblio", llm_orig)
                            best_match_index = llm_author['llm_index']

            
            # Store best match or mark as unmatched
            if best_match:
                match_level, llm_original = best_match
                matches.append({
                    'gt_author': gt_author['original'],
                    'gt_index': gt_author['gt_index'],
                    'llm_author': llm_original,
                    # ADDED (not in the original script): the position of the
                    # matched name within the LLM's answer. Needed for the
                    # author-position analysis (Section 8, Figure A6). Purely additive --
                    # no existing key or code path changes.
                    'llm_index': best_match_index,
                    'match_level': match_level
                })
                # Remove from pool by INDEX
                llm_pool = [p for p in llm_pool if p['llm_index'] != best_match_index]
            else:
                unmatched_gt.append({
                    'gt_author': gt_author['original'],
                    'gt_index': gt_author['gt_index']
                })
        
        # Remaining LLM authors are hallucinations
        hallucinated = [p['original'] for p in llm_pool]

        return {
            'matches': matches,
            'unmatched_gt': unmatched_gt,
            'hallucinated_llm': hallucinated,
            # ADDED: the same hallucinations carrying their position in the
            # LLM's answer, for the author-position analysis. Purely additive.
            'hallucinated_llm_detail': [
                {'llm_author': p['original'], 'llm_index': p['llm_index']} for p in llm_pool
            ],
            'n_llm_names': len(llm_split),
        }