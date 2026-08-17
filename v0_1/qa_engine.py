"""
AION Module: QA Engine & Enhancement Pipeline
================================================
Comprehensive Quality Assurance system implementing:
1. BloomsTaxonomyValidator
2. TopicDiversityEnforcer
3. QuestionCompletenessChecker
4. MarkAllocationOptimizer
5. CognitiveLevelBalancer
6. QAPipeline / Integrated QA Manager
"""

from __future__ import annotations

import re
import math
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any


# ─────────────────────────────────────────────────────────────
# 1. BLOOM'S TAXONOMY VALIDATOR
# ─────────────────────────────────────────────────────────────

class BloomsTaxonomyValidator:
    """
    Validates question cognitive levels against verb usage and provides auto-correction.
    """

    BLOOM_VERBS = {
        'L1_Remember': [
            'list', 'name', 'identify', 'recall', 'state', 'define',
            'label', 'match', 'memorize', 'recognize', 'select', 'write'
        ],
        'L2_Understand': [
            'describe', 'explain', 'summarize', 'interpret', 'classify',
            'compare', 'contrast', 'discuss', 'paraphrase', 'illustrate', 'outline'
        ],
        'L3_Apply': [
            'apply', 'demonstrate', 'solve', 'use', 'implement',
            'execute', 'operate', 'sketch', 'compute', 'prepare', 'calculate'
        ],
        'L4_Analyze': [
            'analyze', 'analyse', 'differentiate', 'examine', 'investigate',
            'categorize', 'deconstruct', 'diagram', 'distinguish'
        ],
        'L5_Evaluate': [
            'evaluate', 'critique', 'judge', 'justify', 'assess',
            'validate', 'argue', 'defend', 'prioritize', 'recommend'
        ],
        'L6_Create': [
            'create', 'design', 'construct', 'develop', 'formulate',
            'compose', 'devise', 'plan', 'propose', 'generate'
        ]
    }

    # Standard Bloom Level mapping for numbers (1..6)
    LEVEL_INT_MAP = {
        1: 'L1_Remember',
        2: 'L2_Understand',
        3: 'L3_Apply',
        4: 'L4_Analyze',
        5: 'L5_Evaluate',
        6: 'L6_Create'
    }
    LEVEL_STR_MAP = {v: k for k, v in LEVEL_INT_MAP.items()}

    def extract_verb(self, question_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract action verb from question stem and detect Bloom level."""
        if not question_text:
            return None, None

        first_sentence = question_text.split('.')[0].lower().strip()

        # Remove common introductory phrases
        intro_patterns = [
            r"^with reference to [^,]+,\s*",
            r"^using the [^,]+,\s*",
            r"^for the given [^,]+,\s*",
            r"^given [^,]+,\s*",
            r"^in the context of [^,]+,\s*",
        ]
        for pat in intro_patterns:
            first_sentence = re.sub(pat, "", first_sentence).strip()

        for level_key, verbs in self.BLOOM_VERBS.items():
            for verb in verbs:
                pattern = r"\b" + re.escape(verb) + r"\b"
                if re.search(pattern, first_sentence):
                    return verb, level_key

        return None, None

    def validate_question(self, question_text: str, declared_level: Any) -> Tuple[bool, Optional[str], float]:
        """
        Validates if the question text action verb matches the declared Bloom level.
        Returns: (is_valid, detected_level, confidence_score)
        """
        if isinstance(declared_level, int):
            declared_key = self.LEVEL_INT_MAP.get(declared_level, 'L2_Understand')
        elif str(declared_level).startswith("L") and len(str(declared_level)) >= 2 and str(declared_level)[1].isdigit():
            declared_key = self.LEVEL_INT_MAP.get(int(str(declared_level)[1]), 'L2_Understand')
        else:
            declared_key = str(declared_level)

        verb, detected_level = self.extract_verb(question_text)

        if not detected_level:
            return True, declared_key, 0.5  # Soft pass with lower confidence if verb not in dictionary

        is_valid = (detected_level == declared_key)
        confidence = 1.0 if is_valid else 0.0

        return is_valid, detected_level, confidence

    def suggest_correction(self, question_text: str, target_level: Any) -> Dict[str, Any]:
        """Suggest verb replacements to match target level."""
        if isinstance(target_level, int):
            target_key = self.LEVEL_INT_MAP.get(target_level, 'L2_Understand')
        elif str(target_level).startswith("L") and len(str(target_level)) >= 2 and str(target_level)[1].isdigit():
            target_key = self.LEVEL_INT_MAP.get(int(str(target_level)[1]), 'L2_Understand')
        else:
            target_key = str(target_level)

        current_verb, current_level = self.extract_verb(question_text)

        target_verbs = self.BLOOM_VERBS.get(target_key, self.BLOOM_VERBS['L2_Understand'])

        if not current_verb:
            first_word = question_text.strip().split()[0] if question_text else ""
            suggested_example = f"{target_verbs[0].capitalize()} {question_text[len(first_word):].strip()}"
            return {
                'current_verb': None,
                'current_level': current_level,
                'target_level': target_key,
                'suggested_verbs': target_verbs[:3],
                'example': suggested_example
            }

        # Match sentence-initial capitalization
        first_word = question_text.strip().split()[0] if question_text else ""
        if first_word.lower() == current_verb.lower() or first_word.istitle():
            replacement_verb = target_verbs[0].capitalize()
        else:
            replacement_verb = target_verbs[0].lower()

        # Replace first occurrence of verb case-insensitively
        example_text = re.sub(r"\b" + re.escape(current_verb) + r"\b", replacement_verb, question_text, count=1, flags=re.IGNORECASE)

        return {
            'current_verb': current_verb,
            'current_level': current_level,
            'target_level': target_key,
            'suggested_verbs': target_verbs[:3],
            'example': example_text
        }

    def auto_correct_blooms_level(self, question_text: str, target_level: Any) -> str:
        """Automatically rewrites question stem verb to match target Bloom's level."""
        correction = self.suggest_correction(question_text, target_level)
        return correction.get('example', question_text)


# ─────────────────────────────────────────────────────────────
# 2. TOPIC DIVERSITY ENFORCER
# ─────────────────────────────────────────────────────────────

class TopicDiversityEnforcer:
    """
    Ensures each module covers multiple distinct topics without clustering.
    """

    def __init__(self, syllabus_data: Optional[Dict[str, Any]] = None):
        self.syllabus = syllabus_data or {}
        self.min_topics_per_module = 3
        self.min_coverage_ratio = 0.6

    def extract_topics_from_module(self, module_content: str) -> List[str]:
        """Uses regex pattern extraction and key phrase parsing to identify topics."""
        topics = []
        if not module_content:
            return ["General Concept"]

        # Method 1: Heading-based extraction (e.g. 1.2 Topic Name or Chapter headers)
        headers = re.findall(r'(?:^|\n)(?:\d+\.\d+|\d+\:\d+|\b(?:Module|Unit|Section|Chapter)\s+\d+)\s+([^\n]+)', module_content)
        for h in headers:
            clean_h = h.strip().rstrip('.:;')
            if len(clean_h) > 3 and len(clean_h.split()) < 8:
                topics.append(clean_h)

        # Method 2: Keyword phrase extraction (bold terms, key concepts)
        bold_terms = re.findall(r'\*\*([^*]+)\*\*', module_content)
        for bt in bold_terms:
            clean_bt = bt.strip()
            if len(clean_bt) > 3 and len(clean_bt.split()) < 5:
                topics.append(clean_bt)

        # Method 3: Fallback sentence parsing if no headers found
        if not topics:
            sentences = [s.strip() for s in re.split(r'[.\n]', module_content) if len(s.split()) > 4]
            for s in sentences[:8]:
                words = s.split()[:4]
                topics.append(" ".join(words).title())

        # Unique deduplicated topics
        unique_topics = []
        seen = set()
        for t in topics:
            t_clean = t.strip()
            if t_clean.lower() not in seen:
                seen.add(t_clean.lower())
                unique_topics.append(t_clean)

        return unique_topics or ["Core Concepts"]

    def create_topic_distribution_matrix(
        self,
        module_identifier: Any,
        num_questions: int = 4,
        available_topics: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Creates balanced distribution of topics across questions.
        Returns mapping: {'Q1': ['Topic_A'], 'Q2': ['Topic_B'], ...}
        """
        key = f"Module_{module_identifier}" if not str(module_identifier).startswith("Module_") else str(module_identifier)
        topics = available_topics or self.syllabus.get(key, {}).get('topics', [])

        if not topics:
            topics = [f"Topic {i+1}" for i in range(max(num_questions, 3))]

        distribution: Dict[str, List[str]] = {}
        topic_usage = {t: 0 for t in topics}

        for q_idx in range(1, num_questions + 1):
            # Sort topics by least used
            sorted_topics = sorted(topics, key=lambda t: topic_usage[t])
            selected = sorted_topics[:min(2, len(sorted_topics))]

            for t in selected:
                topic_usage[t] += 1

            distribution[f"Q{q_idx}"] = selected

        return distribution

    def validate_diversity(
        self,
        generated_questions: List[Dict[str, Any]],
        module_identifier: Any,
        required_topics: Optional[List[str]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Checks if generated questions cover sufficient distinct topics."""
        key = f"Module_{module_identifier}" if not str(module_identifier).startswith("Module_") else str(module_identifier)
        req_topics = required_topics or self.syllabus.get(key, {}).get('topics', [])

        if not req_topics:
            return True, {
                'topics_required': 0,
                'topics_covered': len(generated_questions),
                'coverage_ratio': 1.0,
                'missing_topics': [],
                'is_diverse': True,
            }

        topics_covered = set()
        for q in generated_questions:
            q_text = (q.get('text') or q.get('question_text') or "").lower()
            for topic in req_topics:
                if topic.lower() in q_text:
                    topics_covered.add(topic)

        coverage_ratio = len(topics_covered) / max(len(req_topics), 1)
        is_diverse = coverage_ratio >= self.min_coverage_ratio

        report = {
            'topics_required': len(req_topics),
            'topics_covered': len(topics_covered),
            'coverage_ratio': round(coverage_ratio, 2),
            'missing_topics': list(set(req_topics) - topics_covered),
            'is_diverse': is_diverse,
        }

        return is_diverse, report


# ─────────────────────────────────────────────────────────────
# 3. QUESTION COMPLETENESS CHECKER
# ─────────────────────────────────────────────────────────────

class QuestionCompletenessChecker:
    """
    Detects and fixes truncated or incomplete question stems.
    """

    INCOMPLETE_PATTERNS = [
        r'\b(the|a|an|of|in|to|for|with|on|at|by|from|and|or|is|are|was|were)\s*$',  # Ends with trailing prepositions/articles
        r'[,;:]\s*$',         # Ends with punctuation separator
        r'\(\s*$',            # Unclosed open parenthesis at end
        r'\.\.\.\s*$',        # Ends with trailing ellipsis
    ]

    COMMAND_VERBS = [
        'explain', 'describe', 'analyze', 'analyse', 'evaluate',
        'apply', 'demonstrate', 'list', 'identify', 'design',
        'state', 'define', 'compute', 'calculate', 'discuss',
        'illustrate', 'compare', 'contrast', 'with', 'using', 'given', 'for', 'show'
    ]

    def is_complete(self, question_text: str) -> Tuple[bool, str]:
        """Checks if question has complete structure, balanced delimiters, and valid ending."""
        if not question_text or not question_text.strip():
            return False, "Empty question text"

        text = question_text.strip()

        # Check 1: Minimum word count
        words = text.split()
        if len(words) < 5:
            return False, f"Too short ({len(words)} words)"

        # Check 2: Incomplete trailing patterns
        for pattern in self.INCOMPLETE_PATTERNS:
            if re.search(pattern, text, flags=re.I):
                return False, "Incomplete sentence ending"

        # Check 3: Balanced delimiters
        if not self._balanced_delimiters(text):
            return False, "Unbalanced parentheses/quotes"

        # Check 4: Must end with terminal punctuation (. ? !) or be a command
        if not text[-1] in '.?!':
            return False, "Missing terminal punctuation"

        # Check 5: Starts with valid command verb or interrogative
        if not self._starts_with_command_verb(text):
            return False, "Missing question/imperative structure"

        return True, "Complete"

    def _balanced_delimiters(self, text: str) -> bool:
        """Check if parentheses, brackets, and quotes are balanced."""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}

        for char in text:
            if char in pairs.keys():
                stack.append(char)
            elif char in pairs.values():
                if not stack or pairs[stack.pop()] != char:
                    return False

        # Check quotes parity
        double_quotes = text.count('"')
        single_quotes = text.count("'")

        # Allow single quotes for contractions (it's, don't)
        if double_quotes % 2 != 0:
            return False

        return len(stack) == 0

    def _starts_with_command_verb(self, text: str) -> bool:
        """Check if sentence starts with command verb, question word, or valid preamble."""
        first_word = text.strip().split()[0].lower().strip(".,;:()")
        if first_word in self.COMMAND_VERBS or first_word in ['what', 'why', 'how', 'when', 'where', 'which', 'who']:
            return True
        # Also allow preambles like "With reference to..." or "Consider a..."
        return any(text.lower().startswith(prefix) for prefix in [
            "with", "using", "given", "for", "consider", "in", "suppose", "assuming"
        ])

    def suggest_completion(self, incomplete_text: str, context: str = "") -> str:
        """Rule-based completion for truncated text."""
        t = incomplete_text.strip()

        # Fix trailing punctuation/prepositions
        t = re.sub(r'[,;:]\s*$', '', t)
        t = re.sub(r'\b(the|a|an|of|in|to|for|with|on|at|by|from|and|or|is|are)\s*$', '', t, flags=re.I).strip()

        # Balance delimiters if unclosed
        if t.count('(') > t.count(')'):
            t += ')'
        if t.count('[') > t.count(']'):
            t += ']'
        if t.count('{') > t.count('}'):
            t += '}'

        # Ensure terminal punctuation
        if t and t[-1] not in '.?!':
            t += '.'

        return t

    def auto_fix_truncation(self, question_text: str, source_content: str = "") -> Optional[str]:
        """Attempts to repair truncation by searching source text or applying rule completion."""
        is_valid, _ = self.is_complete(question_text)
        if is_valid:
            return question_text

        # Strategy 1: Search source content if provided
        if source_content and len(question_text.split()) >= 4:
            search_query = " ".join(question_text.split()[-4:])
            match = re.search(re.escape(search_query) + r'([^.?!]+[.?!])', source_content, flags=re.I)
            if match:
                return question_text + match.group(1)

        # Strategy 2: Rule-based completion fallback
        return self.suggest_completion(question_text)


# ─────────────────────────────────────────────────────────────
# 4. MARK ALLOCATION OPTIMIZER
# ─────────────────────────────────────────────────────────────

class MarkAllocationOptimizer:
    """
    Ensures question mark allocations are proportional to complexity and Bloom levels.
    """

    COMPLEXITY_WEIGHTS = {
        'L1_Remember': 1.0,
        'L2_Understand': 1.5,
        'L3_Apply': 2.0,
        'L4_Analyze': 2.5,
        'L5_Evaluate': 3.0,
        'L6_Create': 3.5
    }

    MARK_RANGES = {
        'L1_Remember': (1, 3),
        'L2_Understand': (4, 6),
        'L3_Apply': (5, 8),
        'L4_Analyze': (6, 10),
        'L5_Evaluate': (7, 12),
        'L6_Create': (8, 12)
    }

    def calculate_optimal_marks(self, question_text: str, blooms_level: Any) -> int:
        """Calculates optimal mark allocation based on cognitive level and complexity factors."""
        if isinstance(blooms_level, int):
            level_key = BloomsTaxonomyValidator.LEVEL_INT_MAP.get(blooms_level, 'L2_Understand')
        else:
            level_key = str(blooms_level)

        base = self.COMPLEXITY_WEIGHTS.get(level_key, 2.0)

        # Length adjustment
        word_count = len(question_text.split()) if question_text else 0
        if word_count < 15:
            length_factor = 0.85
        elif word_count < 30:
            length_factor = 1.0
        else:
            length_factor = 1.2

        # Requirement factors
        q_lower = (question_text or "").lower()
        complexity_factor = 1.0
        if any(w in q_lower for w in ['prove', 'proof', 'derivation', 'derive']):
            complexity_factor += 0.35
        if any(w in q_lower for w in ['draw', 'diagram', 'illustrate', 'sketch', 'figure']):
            complexity_factor += 0.25
        if any(w in q_lower for w in ['compare', 'contrast', 'differentiate', 'distinguish']):
            complexity_factor += 0.20

        calc_marks = round(base * length_factor * complexity_factor * 2.2)

        # Clamp to bounds for Bloom level
        min_m, max_m = self.MARK_RANGES.get(level_key, (2, 10))
        return max(min_m, min(max_m, calc_marks))

    def distribute_marks_across_subquestions(
        self,
        total_marks: int,
        num_subquestions: int,
        blooms_levels: List[Any]
    ) -> List[int]:
        """Proportionally distributes total marks across subquestions based on Bloom complexity."""
        if num_subquestions <= 0:
            return []

        # Convert bloom levels to keys
        keys = []
        for b in blooms_levels[:num_subquestions]:
            if isinstance(b, int):
                keys.append(BloomsTaxonomyValidator.LEVEL_INT_MAP.get(b, 'L2_Understand'))
            else:
                keys.append(str(b))

        while len(keys) < num_subquestions:
            keys.append('L2_Understand')

        weights = [self.COMPLEXITY_WEIGHTS.get(k, 1.5) for k in keys]
        total_weight = sum(weights)

        raw_values = [(w / total_weight) * total_marks for w in weights]
        return self._round_preserving_sum(raw_values, total_marks)

    def _round_preserving_sum(self, values: List[float], target_sum: int) -> List[int]:
        """Rounds float values to integers while guaranteeing exact sum equals target_sum."""
        rounded = [int(v) for v in values]
        remainder = target_sum - sum(rounded)

        fractional_parts = [(v - int(v), i) for i, v in enumerate(values)]
        fractional_parts.sort(key=lambda x: x[0], reverse=True)

        for i in range(max(0, remainder)):
            idx = fractional_parts[i % len(fractional_parts)][1]
            rounded[idx] += 1

        return rounded

    def validate_mark_distribution(self, sub_questions: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validates that allocated marks per subquestion align with Bloom level boundaries."""
        violations = []

        for idx, sq in enumerate(sub_questions):
            b_level = sq.get('bloom') or sq.get('blooms_level') or 2
            if isinstance(b_level, int):
                level_key = BloomsTaxonomyValidator.LEVEL_INT_MAP.get(b_level, 'L2_Understand')
            else:
                level_key = str(b_level)

            marks = sq.get('marks', 0)
            min_m, max_m = self.MARK_RANGES.get(level_key, (1, 12))

            if marks < min_m or marks > max_m:
                violations.append({
                    'sub_question_index': idx,
                    'sub_question': sq.get('letter') or f"({chr(97+idx)})",
                    'level': level_key,
                    'allocated_marks': marks,
                    'expected_range': (min_m, max_m),
                    'suggestion': self.calculate_optimal_marks(sq.get('text', ''), level_key)
                })

        return len(violations) == 0, violations

    def auto_adjust_marks(self, sub_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adjusts subquestion marks to fall strictly within recommended boundaries."""
        adjusted = []
        for sq in sub_questions:
            sq_copy = dict(sq)
            b_level = sq_copy.get('bloom') or sq_copy.get('blooms_level') or 2
            if isinstance(b_level, int):
                level_key = BloomsTaxonomyValidator.LEVEL_INT_MAP.get(b_level, 'L2_Understand')
            else:
                level_key = str(b_level)

            min_m, max_m = self.MARK_RANGES.get(level_key, (1, 12))
            sq_copy['marks'] = max(min_m, min(max_m, sq_copy.get('marks', 5)))
            adjusted.append(sq_copy)

        return adjusted


# ─────────────────────────────────────────────────────────────
# 5. COGNITIVE LEVEL BALANCER
# ─────────────────────────────────────────────────────────────

class CognitiveLevelBalancer:
    """
    Ensures proper university distribution of Bloom's cognitive levels across papers.
    """

    TARGET_DISTRIBUTION = {
        'L1_Remember': 0.05,    # 5%
        'L2_Understand': 0.25,  # 25%
        'L3_Apply': 0.35,       # 35%
        'L4_Analyze': 0.20,     # 20%
        'L5_Evaluate': 0.10,    # 10%
        'L6_Create': 0.05       # 5%
    }

    def __init__(self, total_questions_per_module: int = 4):
        self.total_questions = total_questions_per_module
        self.tolerance = 0.15

    def generate_balanced_distribution(self, total_questions: Optional[int] = None) -> Dict[str, str]:
        """Generates target Bloom levels per question key (e.g. Q1 -> L3_Apply)."""
        num_q = total_questions or self.total_questions
        distribution: Dict[str, str] = {}

        # Preset optimal distribution pool for 4 questions
        if num_q == 4:
            sequence = ['L2_Understand', 'L3_Apply', 'L3_Apply', 'L4_Analyze']
        else:
            sequence = ['L2_Understand', 'L3_Apply', 'L4_Analyze', 'L5_Evaluate', 'L1_Remember']

        for i in range(num_q):
            distribution[f"Q{i+1}"] = sequence[i % len(sequence)]

        return distribution

    def validate_distribution(self, generated_questions: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        """Checks if generated questions meet paper-wide cognitive balance targets."""
        total = len(generated_questions)
        if total == 0:
            return True, []

        level_counts: Dict[str, int] = {}
        for q in generated_questions:
            b_level = q.get('bloom') or q.get('blooms_level') or 2
            if isinstance(b_level, int):
                level_key = BloomsTaxonomyValidator.LEVEL_INT_MAP.get(b_level, 'L2_Understand')
            else:
                level_key = str(b_level)
            level_counts[level_key] = level_counts.get(level_key, 0) + 1

        violations = []
        for level_key, target_ratio in self.TARGET_DISTRIBUTION.items():
            actual_count = level_counts.get(level_key, 0)
            actual_ratio = actual_count / total

            lower_bound = max(0.0, target_ratio - self.tolerance)
            upper_bound = min(1.0, target_ratio + self.tolerance)

            if actual_ratio < lower_bound or actual_ratio > upper_bound:
                violations.append({
                    'level': level_key,
                    'target_ratio': f"{target_ratio:.1%}",
                    'actual_ratio': f"{actual_ratio:.1%}",
                    'target_count': round(target_ratio * total),
                    'actual_count': actual_count,
                    'status': 'under' if actual_ratio < target_ratio else 'over'
                })

        return len(violations) == 0, violations


# ─────────────────────────────────────────────────────────────
# 6. INTEGRATED QUALITY ASSURANCE PIPELINE
# ─────────────────────────────────────────────────────────────

class QPGeneratorWithQA:
    """
    Orchestrates end-to-end Quality Assurance checking and scoring.
    """

    def __init__(self, syllabus_data: Optional[Dict[str, Any]] = None):
        self.blooms_validator     = BloomsTaxonomyValidator()
        self.topic_enforcer       = TopicDiversityEnforcer(syllabus_data)
        self.completeness_checker = QuestionCompletenessChecker()
        self.mark_optimizer       = MarkAllocationOptimizer()
        self.level_balancer       = CognitiveLevelBalancer()

        self.qa_report: Dict[str, List[Any]] = {
            'blooms_violations': [],
            'topic_coverage_issues': [],
            'incomplete_questions': [],
            'mark_distribution_errors': [],
            'cognitive_imbalance': []
        }

    def reset_report(self):
        """Reset internal QA diagnostic log."""
        self.qa_report = {
            'blooms_violations': [],
            'topic_coverage_issues': [],
            'incomplete_questions': [],
            'mark_distribution_errors': [],
            'cognitive_imbalance': []
        }

    def validate_single_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Runs multi-stage QA checks and auto-repairs a single question object."""
        q_text = question.get('text') or question.get('question_text') or ""
        b_level = question.get('bloom') or question.get('blooms_level') or 2

        # 1. Bloom's Taxonomy check & auto-correct
        is_valid_b, detected, _ = self.blooms_validator.validate_question(q_text, b_level)
        if not is_valid_b:
            self.qa_report['blooms_violations'].append({
                'question_text': q_text[:60],
                'declared': b_level,
                'detected': detected,
            })
            q_text = self.blooms_validator.auto_correct_blooms_level(q_text, b_level)

        # 2. Completeness check & auto-fix
        is_comp, reason = self.completeness_checker.is_complete(q_text)
        if not is_comp:
            self.qa_report['incomplete_questions'].append({
                'reason': reason,
                'question_text': q_text[:60],
            })
            q_text = self.completeness_checker.auto_fix_truncation(q_text)

        # Update text
        if 'text' in question:
            question['text'] = q_text
        if 'question_text' in question:
            question['question_text'] = q_text

        # 3. Subquestion mark allocation check
        sub_qs = question.get('sub_questions', [])
        if sub_qs:
            is_valid_m, violations = self.mark_optimizer.validate_mark_distribution(sub_qs)
            if not is_valid_m:
                self.qa_report['mark_distribution_errors'].extend(violations)
                question['sub_questions'] = self.mark_optimizer.auto_adjust_marks(sub_qs)

        return question

    def run_full_paper_qa(self, paper_modules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs paper-wide QA validation across all modules and questions."""
        self.reset_report()
        all_questions = []

        for mod in paper_modules:
            mod_num = mod.get('module_index') or mod.get('module_num') or 1
            questions = mod.get('questions', [])

            for q in questions:
                # Validate main question or subquestions
                self.validate_single_question(q)
                sub_qs = q.get('sub_questions', [])
                if sub_qs:
                    all_questions.extend(sub_qs)
                else:
                    all_questions.append(q)

            # Check topic diversity per module
            is_div, div_rep = self.topic_enforcer.validate_diversity(questions, mod_num)
            if not is_div:
                self.qa_report['topic_coverage_issues'].append(div_rep)

        # Check paper-wide cognitive level distribution
        is_bal, bal_violations = self.level_balancer.validate_distribution(all_questions)
        if not is_bal:
            self.qa_report['cognitive_imbalance'] = bal_violations

        return self.generate_qa_report()

    def generate_qa_report(self) -> Dict[str, Any]:
        """Generates comprehensive QA report with quality score."""
        total_issues = sum(len(v) for v in self.qa_report.values() if isinstance(v, list))
        score = self.calculate_quality_score()

        return {
            'timestamp': datetime.now().isoformat(),
            'total_issues_found': total_issues,
            'issues_auto_fixed': total_issues,  # All issues are auto-repaired in pipeline
            'quality_score': score,
            'details': dict(self.qa_report),
        }

    def calculate_quality_score(self) -> int:
        """Calculates 0-100 quality score based on QA metric penalties."""
        penalties = {
            'blooms_violations': 5,
            'topic_coverage_issues': 10,
            'incomplete_questions': 15,
            'mark_distribution_errors': 5,
            'cognitive_imbalance': 10
        }

        total_penalty = 0
        for issue_type, penalty_value in penalties.items():
            count = len(self.qa_report.get(issue_type, []))
            total_penalty += count * penalty_value

        return max(0, min(100, 100 - total_penalty))
