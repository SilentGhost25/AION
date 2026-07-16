# AION-Trainer/server/prompt/vtu_reference_library.py
"""
VTU Reference Question Library.

A hand-curated set of real VTU question patterns grouped by topic area
and question type. These are injected into prompts as style examples —
they show the model what "a real VTU question looks like" rather than
relying on it to infer style from general pretraining.

This is a static library at first. As real previous-papers are
extracted by PYQParser, `DynamicReferenceLibrary` automatically
extends it with subject-specific examples.

Important: these are STYLE examples, not answer templates. The model
is explicitly told not to copy them.
"""

from typing import Dict, List, Tuple

# Key: (topic_keyword, bloom_level, question_type)
# Value: list of reference question strings
STATIC_LIBRARY: Dict[Tuple[str, str, str], List[str]] = {

    # ---- Search Algorithms -------------------------------------------
    ("search", "L2", "explanation"): [
        "Explain the working of BFS with a suitable example.",
        "Describe the Depth-First Search algorithm and trace it on a sample graph.",
    ],
    ("search", "L3", "algorithm"): [
        "Illustrate the A* search algorithm with a worked example. "
        "Show the f(n), g(n), and h(n) values at each step.",
        "Apply Dijkstra's algorithm to find the shortest path in the given graph. "
        "Show all steps.",
    ],
    ("search", "L4", "comparison"): [
        "Compare BFS and DFS with respect to time complexity, space complexity, "
        "completeness, and optimality.",
        "Differentiate between informed and uninformed search strategies "
        "with suitable examples.",
    ],

    # ---- Trees and Graphs -------------------------------------------
    ("tree", "L2", "explanation"): [
        "Explain the structure of a Binary Search Tree with a suitable diagram.",
        "Describe AVL trees and explain how rotations maintain balance.",
    ],
    ("tree", "L3", "algorithm"): [
        "Construct a Binary Search Tree for the given set of values "
        "and trace the insertion process.",
        "Illustrate the in-order, pre-order, and post-order traversals "
        "of the given binary tree.",
    ],
    ("tree", "L4", "comparison"): [
        "Compare Binary Search Trees and AVL Trees with respect to "
        "insertion, deletion, and search complexity.",
        "Differentiate between B-Tree and B+ Tree with a suitable example.",
    ],

    # ---- Sorting and Algorithms -------------------------------------
    ("sort", "L3", "algorithm"): [
        "Trace the Quick Sort algorithm on the array [5, 3, 8, 1, 4] "
        "and show each partition step.",
        "Apply Merge Sort on the given array and show the merge steps clearly.",
    ],
    ("sort", "L4", "comparison"): [
        "Compare Quick Sort and Merge Sort with respect to best, average, "
        "and worst-case time complexity.",
        "Analyze the stability and space complexity of Heap Sort, "
        "Merge Sort, and Quick Sort.",
    ],

    # ---- AI / Machine Learning -------------------------------------
    ("neural", "L2", "explanation"): [
        "Explain the architecture of an Artificial Neural Network "
        "with a neat diagram.",
        "Describe the back-propagation algorithm and explain how "
        "weights are updated.",
    ],
    ("neural", "L3", "algorithm"): [
        "Illustrate the working of a single-layer perceptron with a "
        "suitable example showing weight updates.",
        "Apply the gradient descent algorithm to minimize the given "
        "loss function. Show each iteration step.",
    ],

    # ---- Databases --------------------------------------------------
    ("database", "L2", "explanation"): [
        "Explain the concept of normalization and discuss the need "
        "for normal forms in database design.",
        "Describe the ACID properties of a transaction with a "
        "suitable example.",
    ],
    ("database", "L4", "comparison"): [
        "Compare SQL and NoSQL databases with respect to schema, "
        "scalability, and use cases.",
        "Differentiate between clustered and non-clustered indexes "
        "with a suitable example.",
    ],

    # ---- Operating Systems -----------------------------------------
    ("process", "L2", "explanation"): [
        "Explain the various states of a process with a neat state "
        "transition diagram.",
        "Describe deadlock and explain the four necessary conditions "
        "for deadlock to occur.",
    ],
    ("process", "L3", "algorithm"): [
        "Apply the Banker's Algorithm to determine if the system is "
        "in a safe state for the given resource allocation matrix.",
        "Illustrate the Round Robin scheduling algorithm for the "
        "given set of processes with a Gantt chart.",
    ],

    # ---- Computer Networks -----------------------------------------
    ("network", "L2", "explanation"): [
        "Explain the TCP/IP protocol stack with a neat diagram "
        "showing the function of each layer.",
        "Describe the working of the sliding window protocol "
        "with a suitable example.",
    ],
    ("network", "L4", "comparison"): [
        "Compare TCP and UDP with respect to reliability, connection "
        "establishment, and use cases.",
        "Differentiate between circuit switching and packet switching "
        "networks.",
    ],

    # ---- Generic VTU patterns (fallback) ----------------------------
    ("", "L1", "definition"): [
        "Define the term {} with a suitable example.",
        "State and explain the significance of {}.",
    ],
    ("", "L2", "explanation"): [
        "Explain {} with a neat diagram and a suitable example.",
        "Describe the working of {} and list its advantages and limitations.",
    ],
    ("", "L3", "algorithm"): [
        "Illustrate the {} algorithm with a worked example showing each step.",
        "Apply {} to solve the following problem: [problem statement].",
    ],
    ("", "L4", "comparison"): [
        "Compare {} and {} with respect to time complexity, space "
        "complexity, and applicability.",
        "Differentiate between {} and {} with a suitable example.",
    ],
    ("", "L5", "explanation"): [
        "Evaluate the advantages and limitations of {} for real-world applications.",
        "Justify the choice of {} over alternative approaches for the given scenario.",
    ],
    ("", "L6", "explanation"): [
        "Design an efficient {} for the given problem statement and justify your design.",
        "Propose an improvement to the existing {} and explain the expected benefits.",
    ],
}


class VTUReferenceLibrary:
    """
    Retrieves style-matched reference questions for a given context.
    Falls back gracefully through: exact match -> bloom match -> generic.
    """

    def __init__(self):
        self._library = dict(STATIC_LIBRARY)

    def get_references(
        self,
        topic: str,
        bloom_level: str,
        question_type: str,
        max_references: int = 3,
    ) -> List[str]:
        """
        Return reference questions matched to the given context.
        Topic matching is keyword-based so it works without any
        embeddings or external dependencies.
        """
        topic_lower = topic.lower()

        # Pass 1: exact keyword + bloom + type
        for (kw, bl, qt), questions in self._library.items():
            if kw and kw in topic_lower and bl == bloom_level and qt == question_type:
                return questions[:max_references]

        # Pass 2: keyword + bloom only
        for (kw, bl, qt), questions in self._library.items():
            if kw and kw in topic_lower and bl == bloom_level:
                return questions[:max_references]

        # Pass 3: bloom + type generic (empty keyword = generic fallback)
        generic_key = ("", bloom_level, question_type)
        if generic_key in self._library:
            refs = self._library[generic_key]
            # Substitute the topic name into generic templates where possible
            return [
                r.replace("{}", topic)[:200] for r in refs[:max_references]
            ]

        # Pass 4: bloom-only generic
        for (kw, bl, qt), questions in self._library.items():
            if kw == "" and bl == bloom_level:
                return [
                    q.replace("{}", topic)[:200] for q in questions[:max_references]
                ]

        return []

    def add(self, topic_keyword: str, bloom_level: str, question_type: str,
             questions: List[str]):
        """Add reference questions from PYQParser output at runtime."""
        key = (topic_keyword.lower(), bloom_level, question_type)
        existing = self._library.get(key, [])
        for q in questions:
            if q not in existing:
                existing.append(q)
        self._library[key] = existing


class DynamicReferenceLibrary(VTUReferenceLibrary):
    """
    Extended reference library that is populated at runtime from the
    PYQParser output for a specific subject, giving the model
    subject-specific style examples rather than just generic VTU patterns.
    """

    def populate_from_pyq_records(
        self,
        pyq_records: List[Dict],
        topic_extractor=None,
    ) -> int:
        """
        Ingest PYQParser question records into the library.
        Returns the number of questions added.

        topic_extractor: optional callable(question_text) -> str keyword.
        Defaults to using the first 'real' word of the question as keyword.
        """
        from server.pyq_extractor import classify_bloom
        from builders.examiner_style import ExaminerStyleExtractor

        extractor = ExaminerStyleExtractor()
        added = 0

        for record in pyq_records:
            text = record.get("text", "")
            bloom = record.get("bloom") or classify_bloom(text)
            verb = extractor._detect_verb(text)
            qtype = self._verb_to_qtype(verb)
            keyword = (
                topic_extractor(text) if topic_extractor
                else self._extract_keyword(text)
            )
            self.add(keyword, bloom, qtype, [text])
            added += 1

        return added

    def _verb_to_qtype(self, verb: str) -> str:
        mapping = {
            "define": "definition", "explain": "explanation",
            "describe": "explanation", "discuss": "explanation",
            "trace": "algorithm", "apply": "algorithm",
            "illustrate": "algorithm", "implement": "algorithm",
            "compare": "comparison", "differentiate": "comparison",
            "design": "explanation", "evaluate": "explanation",
            "justify": "explanation", "analyze": "explanation",
        }
        return mapping.get(verb.lower(), "explanation")

    def _extract_keyword(self, text: str) -> str:
        """Heuristic: first non-verb, non-article word of length >= 4."""
        skip = {
            "explain", "describe", "compare", "define", "discuss",
            "trace", "apply", "design", "derive", "prove", "with",
            "and", "the", "for", "its", "that", "this", "using",
        }
        for word in text.lower().split():
            word_clean = word.strip(".,?")
            if len(word_clean) >= 4 and word_clean not in skip:
                return word_clean
        return ""
