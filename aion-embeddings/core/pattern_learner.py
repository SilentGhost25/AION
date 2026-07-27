# core/pattern_learner.py

import re
import json
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from storage.database import get_connection

# Stop words to ignore when extracting concepts
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "about", "as", "into", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "so", "yet", "both",
    "it", "its", "this", "that", "these", "those", "what", "which",
    "who", "whom", "how", "why", "when", "where", "each", "every",
    "all", "any", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "also", "between", "own"
}


class PatternLearner:
    """
    Learns question formation patterns from parsed questions.
    Extracts templates, discovers common structures, and tracks
    what types of questions appear for which topics.
    """

    def __init__(self):
        self.templates = Counter()
        self.bloom_distribution = defaultdict(Counter)   # subject → bloom → count
        self.type_distribution = defaultdict(Counter)    # subject → type → count
        self.topic_question_map = defaultdict(list)      # topic → [questions]
        self.mark_distribution = defaultdict(Counter)    # subject → marks → count

    def learn_from_questions(self, questions: List[Dict]):
        """Process a batch of parsed questions and learn patterns."""
        for q in questions:
            subject = q.get("subject", "general")
            text = q["question_text"]
            bloom = q.get("bloom_level", "understand")
            qtype = q.get("question_type", "descriptive")
            marks = q.get("marks")

            # 1. Extract template
            template = self._extract_template(text)
            if template:
                self.templates[template] += 1

            # 2. Track distributions
            self.bloom_distribution[subject][bloom] += 1
            self.type_distribution[subject][qtype] += 1
            if marks:
                self.mark_distribution[subject][marks] += 1

            # 3. Map topics to questions
            concepts = self._extract_concepts(text)
            for concept in concepts:
                self.topic_question_map[concept].append({
                    "question": text,
                    "bloom": bloom,
                    "type": qtype,
                    "marks": marks,
                    "subject": subject
                })

        # Save patterns to database
        self._save_patterns()

    def _extract_template(self, question_text: str) -> str:
        """
        Convert a specific question into a generic template.
        
        "Explain the concept of virtual memory with a diagram"
        →
        "Explain the concept of {CONCEPT} with a {DETAIL}"
        """
        text = question_text.strip()

        # Replace quoted terms
        text = re.sub(r'"([^"]+)"', '{TERM}', text)
        text = re.sub(r"'([^']+)'", '{TERM}', text)

        # Replace technical terms (capitalized multi-word phrases)
        text = re.sub(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', '{CONCEPT}', text)

        # Replace single capitalized words that aren't sentence starters
        words = text.split()
        result = []
        for i, word in enumerate(words):
            clean = re.sub(r'[^a-zA-Z]', '', word)
            if (
                i > 0
                and clean
                and clean[0].isupper()
                and clean.lower() not in STOP_WORDS
                and len(clean) > 3
                and word != '{CONCEPT}'
                and word != '{TERM}'
            ):
                result.append('{CONCEPT}')
            else:
                result.append(word)
        text = ' '.join(result)

        # Replace numbers
        text = re.sub(r'\b\d+\b', '{NUM}', text)

        # Deduplicate consecutive placeholders
        text = re.sub(r'(\{CONCEPT\}\s*)+', '{CONCEPT} ', text)

        return text.strip()

    def _extract_concepts(self, question_text: str) -> List[str]:
        """Extract key concepts/topics from a question."""
        # Remove common question prefixes
        cleaned = re.sub(
            r'^(?:explain|define|describe|discuss|what is|list|state|write)\s+',
            '', question_text, flags=re.IGNORECASE
        ).strip()

        # Extract multi-word capitalized phrases
        concepts = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', cleaned)

        # Also extract terms after "of", "about", "between"
        context_patterns = [
            r'(?:of|about|regarding|between)\s+([a-zA-Z\s]+?)(?:\.|,|\?|$)',
        ]
        for pat in context_patterns:
            matches = re.findall(pat, cleaned, re.IGNORECASE)
            for m in matches:
                words = m.strip().split()
                # Take first 3 words as concept
                concept = ' '.join(words[:3]).strip()
                if len(concept) > 3 and concept.lower() not in STOP_WORDS:
                    concepts.append(concept)

        # Deduplicate and clean
        seen = set()
        unique = []
        for c in concepts:
            c_lower = c.lower().strip()
            if c_lower not in seen and c_lower not in STOP_WORDS and len(c_lower) > 2:
                seen.add(c_lower)
                unique.append(c)

        return unique

    def _save_patterns(self):
        """Save learned patterns to database."""
        with get_connection() as conn:
            for template, count in self.templates.most_common(500):
                # Find an example question for this template
                conn.execute("""
                    INSERT INTO question_patterns
                    (pattern_template, frequency, question_type, bloom_level, subject)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT DO UPDATE SET frequency = frequency + ?
                """, (template, count, None, None, None, count))

    def get_statistics(self) -> Dict:
        """Return current learning statistics."""
        with get_connection() as conn:
            total_questions = conn.execute(
                "SELECT COUNT(*) FROM questions"
            ).fetchone()[0]

            total_patterns = conn.execute(
                "SELECT COUNT(*) FROM question_patterns"
            ).fetchone()[0]

            top_patterns = conn.execute(
                "SELECT pattern_template, frequency FROM question_patterns ORDER BY frequency DESC LIMIT 10"
            ).fetchall()

            bloom_dist = conn.execute(
                "SELECT bloom_level, COUNT(*) as cnt FROM questions GROUP BY bloom_level"
            ).fetchall()

            type_dist = conn.execute(
                "SELECT question_type, COUNT(*) as cnt FROM questions GROUP BY question_type"
            ).fetchall()

        return {
            "total_questions": total_questions,
            "total_patterns": total_patterns,
            "top_patterns": [
                {"template": r["pattern_template"], "frequency": r["frequency"]}
                for r in top_patterns
            ],
            "bloom_distribution": {
                r["bloom_level"]: r["cnt"] for r in bloom_dist
            },
            "type_distribution": {
                r["question_type"]: r["cnt"] for r in type_dist
            }
        }

    def generate_training_pairs(self, questions: List[Dict]) -> List[Dict]:
        """
        Generate training pairs from questions for embedding model.
        Multiple strategies:
        1. Similar questions (same topic) → positive pairs
        2. Same bloom level questions → weak positive
        3. Different subject questions → negatives
        4. Question ↔ extracted concept → positive
        """
        pairs = []
        by_subject = defaultdict(list)
        by_bloom = defaultdict(list)
        by_topic = defaultdict(list)

        for q in questions:
            by_subject[q.get("subject", "general")].append(q)
            by_bloom[q.get("bloom_level", "understand")].append(q)
            concepts = self._extract_concepts(q["question_text"])
            for c in concepts:
                by_topic[c.lower()].append(q)

        # Strategy 1: Same-topic questions are positives
        for topic, topic_questions in by_topic.items():
            if len(topic_questions) < 2:
                continue
            for i in range(len(topic_questions)):
                for j in range(i + 1, min(i + 3, len(topic_questions))):
                    pairs.append({
                        "anchor": topic_questions[i]["question_text"],
                        "positive": topic_questions[j]["question_text"],
                        "pair_type": "question_similarity",
                        "subject": topic_questions[i].get("subject", "general")
                    })

        # Strategy 2: Question ↔ concept definition
        for q in questions:
            concepts = self._extract_concepts(q["question_text"])
            for concept in concepts[:2]:
                pairs.append({
                    "anchor": f"What is {concept}?",
                    "positive": q["question_text"],
                    "pair_type": "question_topic",
                    "subject": q.get("subject", "general")
                })

        # Strategy 3: Cross-subject negatives (for triplet training)
        subjects = list(by_subject.keys())
        if len(subjects) >= 2:
            import random
            for q in questions[:100]:
                q_subject = q.get("subject", "general")
                other_subjects = [s for s in subjects if s != q_subject]
                if other_subjects:
                    neg_subject = random.choice(other_subjects)
                    neg_q = random.choice(by_subject[neg_subject])
                    # Find a positive from same subject
                    same_subject_qs = [
                        sq for sq in by_subject[q_subject]
                        if sq["question_text"] != q["question_text"]
                    ]
                    if same_subject_qs:
                        pos_q = random.choice(same_subject_qs)
                        pairs.append({
                            "anchor": q["question_text"],
                            "positive": pos_q["question_text"],
                            "negative": neg_q["question_text"],
                            "pair_type": "cross_subject_triplet",
                            "subject": q_subject
                        })

        return pairs
