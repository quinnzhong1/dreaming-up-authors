"""Splitting a raw author string into individual names.

Ported verbatim from ``name_matcher_v3.NameSplitter``, the version that
produced every evaluated CSV in ``release/responses/``.

Not a comma split: three formats occur -- normal, bibliographic (``Posner
George J.``), and alternating (``Wurman, Peter, Mountz, Mick``). The last two
are detected conservatively, because a misfire reorders correct names into
wrong ones and inflates the hallucination rate.
"""

from __future__ import annotations

import re
from typing import List, Set

class NameSplitter:
    """
    Splits a raw author string into a list of individual full names.
    Enhanced to detect and correct "LastName FirstName MiddleName" format.
    """
    def __init__(self):
        self.surname_particles: Set[str] = {
            'van', 'von', 'de', 'der', 'den', 'del', 'della', 'di', 'da',
            'do', 'dos', 'du', 'la', 'le', 'van der', 'van den', 'von der'
        }

        self.name_suffixes: Set[str] = {
            'jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'v',
            'phd', 'ph.d', 'ph.d.', 'md', 'm.d', 'm.d.',
            'esq', 'esq.'
        }

    def _is_initial(self, word: str) -> bool:
        """True if the word is a single letter followed by a period."""
        return len(word) == 2 and word[1] == '.' and word[0].isalpha() and word[0].isupper()

    def _looks_like_initial_plus_surname(self, segment: str) -> bool:
        """True if the segment reads as ``Initial(s) Surname``, e.g. ``J. P. Tu``.

        Its presence rules out the alternating format, where a segment in the
        first-name position would never carry initials.
        """
        words = segment.strip().split()
        
        # Must have at least 2 words
        if len(words) < 2:
            return False
        
        # Check if all words EXCEPT the last are initials
        for i in range(len(words) - 1):
            if not self._is_initial(words[i]):
                return False
        
        # Last word must be a full word (not an initial, not a particle)
        last_word = words[-1]
        
        if self._is_initial(last_word):
            return False
        
        if last_word.lower() in self.surname_particles:
            return False
        
        # This looks like "Initial(s) Surname" pattern
        return True

    def _detect_reversed_format(self, segments: List[str]) -> bool:
        """True for alternating ``Surname, Firstname`` lists, e.g. ``Wurman, Peter, Mountz, Mick``.

        Requires positive evidence -- all segments single-word, none carrying
        initials -- because a false positive reorders every name in the list.
        """
        # Must have even number of segments to pair up
        if len(segments) < 2 or len(segments) % 2 != 0:
            return False
        
        # Track evidence for reversed format
        single_word_even_count = 0
        single_word_odd_count = 0
        has_initials_anywhere = False
        
        # Check even-indexed segments (potential surnames)
        for i in range(0, len(segments), 2):
            potential_surname = segments[i].strip()
            words = potential_surname.split()
            
            # Check for initials in this segment
            for word in words:
                if self._is_initial(word):
                    has_initials_anywhere = True
            
            # Count if it's a single word (allowing for particles)
            non_particle_words = [w for w in words if w.lower() not in self.surname_particles]
            if len(non_particle_words) == 1:
                single_word_even_count += 1
            
            # STRICTER: If it has multiple non-particle words, likely not a surname
            if len(non_particle_words) > 1:
                return False
        
        # Check odd-indexed segments (potential first names)
        for i in range(1, len(segments), 2):
            potential_firstname = segments[i].strip()
            words = potential_firstname.split()
            
            # Check for initials
            for word in words:
                if self._is_initial(word):
                    has_initials_anywhere = True
            
            # If it looks like "Initial Surname", definitely not reversed format
            if self._looks_like_initial_plus_surname(potential_firstname):
                return False
            
            # Count single-word first names
            if len(words) == 1:
                single_word_odd_count += 1
            
            # If it has multiple words and any are initials, probably not a first name
            if len(words) > 1:
                has_non_initial = False
                for word in words:
                    if not self._is_initial(word) and word.lower() not in self.surname_particles:
                        has_non_initial = True
                # Multiple words with mix of initials and names = likely full name, not just first name
                if has_non_initial and any(self._is_initial(w) for w in words):
                    return False
        
        # STRICTER CRITERIA: Require strong positive evidence
        # 1. Most segments should be single words
        # 2. Prefer no initials anywhere (they indicate normal format)
        # 3. At least 75% of even and odd segments should be single words
        
        total_segments = len(segments)
        even_segments_count = total_segments // 2
        odd_segments_count = total_segments // 2
        
        # If we have initials anywhere, be very conservative
        if has_initials_anywhere:
            # Only accept if ALL segments are single words (no middle initials)
            return (single_word_even_count == even_segments_count and 
                    single_word_odd_count == odd_segments_count)
        
        # Without initials, require most segments to be single words
        even_ratio = single_word_even_count / even_segments_count
        odd_ratio = single_word_odd_count / odd_segments_count
        
        return even_ratio >= 0.75 and odd_ratio >= 0.75

    def _parse_reversed_format(self, segments: List[str]) -> List[str]:
        """Parse alternating surname, firstname format"""
        names = []
        for i in range(0, len(segments), 2):
            names.append(f"{segments[i+1]} {segments[i]}")
        return names
    
    def _detect_lastname_firstname_middle_format(self, segments: List[str]) -> bool:
        """True for bibliographic segments like ``Posner George J.``.

        Keys on most segments ending in initials while starting with a full
        surname, and on that pattern holding across segments.
        """
        if len(segments) < 2:  # Need multiple names to detect pattern
            return False
        
        # Count how many segments match the specific bibliographic pattern
        bibliographic_pattern_count = 0
        normal_format_count = 0
        
        for segment in segments:
            words = segment.strip().split()
            if len(words) < 2:
                continue
            
            first_word = words[0]
            last_word = words[-1]
            
            # Check if LAST word is initials (single or concatenated)
            last_is_initials = False
            
            # Single initial: exactly 2 chars, second is period
            if len(last_word) == 2 and last_word[1] == '.' and last_word[0].isalpha():
                last_is_initials = True
            # Concatenated initials: alternating letters and periods
            elif '.' in last_word and re.match(r'^[A-Z](\.[A-Z])+\.?$', last_word):
                last_is_initials = True
            
            # Check if FIRST word looks like a first name or initial
            first_is_name_like = (
                not self._is_initial(first_word) and 
                '.' not in first_word and 
                len(first_word) > 2
            )
            
            # Bibliographic pattern: Surname FirstName Initial(s)
            # First word is full surname, last word is initials
            if first_is_name_like and last_is_initials:
                bibliographic_pattern_count += 1
            
            # Normal pattern: FirstName/Initial ... Surname
            # First word is initial or name, last is NOT initials
            elif (self._is_initial(first_word) or first_is_name_like) and not last_is_initials:
                normal_format_count += 1
        
        # Only trigger if most segments follow bibliographic pattern
        # AND there are few normal format segments
        total_segments = len(segments)
        bibliographic_ratio = bibliographic_pattern_count / total_segments
        normal_ratio = normal_format_count / total_segments
        
        # Require strong evidence: >60% bibliographic and <20% normal
        return bibliographic_ratio >= 0.6 and normal_ratio < 0.2

    def _correct_lastname_firstname_middle_format(self, segments: List[str]) -> List[str]:
        """Reorder bibliographic names, keeping surname particles attached."""
        corrected = []
        for segment in segments:
            words = segment.strip().split()
            if len(words) >= 2:
                # Identify the surname part (which may include particles)
                surname_parts = []
                given_name_start_idx = 0
                
                # Collect particle + surname
                for i, word in enumerate(words):
                    if i == 0 and word.lower() in self.surname_particles:
                        # First word is a particle, keep collecting
                        surname_parts.append(word)
                    elif i > 0 and len(surname_parts) > 0 and word.lower() not in self.surname_particles:
                        # Found the actual surname after particle(s)
                        surname_parts.append(word)
                        given_name_start_idx = i + 1
                        break
                    elif i == 0:
                        # First word is the surname (no particle)
                        surname_parts.append(word)
                        given_name_start_idx = 1
                        break
                
                # Get the given names
                if given_name_start_idx < len(words):
                    given_names = ' '.join(words[given_name_start_idx:])
                    surname = ' '.join(surname_parts)
                    corrected.append(f"{given_names} {surname}")
                else:
                    # No given names found, keep as is
                    corrected.append(segment)
            else:
                # If only one word, keep as is
                corrected.append(segment)
        return corrected

    def _merge_suffixes(self, segments: List[str]) -> List[str]:
        """Reattach ``Jr.``, ``Sr.``, ``III`` to the preceding segment.

        Splitting on commas otherwise turns one author into two.
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # Check if we have a next segment and it's a suffix
            if i + 1 < len(segments):
                next_segment = segments[i + 1].strip()
                
                # Check if next segment is a known suffix
                if next_segment.lower() in self.name_suffixes:
                    # Merge current with suffix
                    merged.append(f"{current}, {next_segment}")
                    i += 2  # Skip both current and suffix
                    continue
            
            # Not followed by a suffix, add as-is
            merged.append(current)
            i += 1
        
        return merged

    def split(self, author_string: str) -> List[str]:
        """Split a raw LLM author string into individual names.

        Normalises `` and `` to a comma, splits, reattaches suffixes, then
        applies the bibliographic and alternating corrections if detected.
        """
        # Handle empty input
        if not author_string or not author_string.strip():
            return []
        
        # Normalize separators: convert "and" to comma
        processed_string = re.sub(r'\s+and\s+', ', ', author_string)
        
        # Split by comma and clean up
        segments = [s.strip() for s in processed_string.split(',') 
                if s.strip() and s.strip() != '...']
        
        # Merge suffixes back with previous names
        segments = self._merge_suffixes(segments)
        
        # Detection pipeline: check bibliographic format first
        if self._detect_lastname_firstname_middle_format(segments):
            return self._correct_lastname_firstname_middle_format(segments)
        
        # Then check alternating format (with stricter detection)
        elif self._detect_reversed_format(segments):
            return self._parse_reversed_format(segments)
        
        # No special format detected, return as-is
        else:
            return segments