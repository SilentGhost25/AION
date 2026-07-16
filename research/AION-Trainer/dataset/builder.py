"""
Dataset Builder — converts Knowledge Objects into training samples.

One Knowledge Object → 40+ training samples.

Output format:
{
    "knowledge": "...",
    "answer_graph": "...",
    "bloom": "L3",
    "marks": 10,
    "question_style": "Explain",
    "question": "...",
    "subject": "BAI401"
}
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

logger = logging.getLogger("aion.dataset")


@dataclass
class TrainingSample:
    """A single training sample."""
    knowledge: str = ""
    answer_graph: str = ""
    bloom: str = "L2"
    marks: int = 10
    question_style: str = "Explain"
    question: str = ""
    expected_answer: str = ""
    subject: str = ""
    module: int = 0
    difficulty: str = "medium"
    sample_type: str = "knowledge_to_question"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetBuilder:
    """
    Builds training dataset from Knowledge Objects.

    One concept can produce 40+ training examples:
        Knowledge → Question
        Knowledge → Expected Answer
        Knowledge → Bloom
        Knowledge → Question Style
        Knowledge → Diagram Requirement
        Knowledge → Marks
        Definition → Concept
        Concept → Definition
        Algorithm → Question
        Diagram → Question
        ...
    """

    BLOOM_LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6"]
    QUESTION_STYLES = [
        "Define", "Explain", "Describe", "Compare", "Differentiate",
        "Trace", "Construct", "Analyze", "Evaluate", "Design",
        "Illustrate", "Solve", "List", "State", "Summarize",
        "Demonstrate", "Justify", "Critique", "Propose", "Develop",
        "Apply",
    ]

    def __init__(self, output_dir: str = "dataset/", subject_code: str = ""):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.subject_code = subject_code

    def build(self, knowledge_objects: List) -> str:
        """Build complete dataset from Knowledge Objects."""
        logger.info("Building training dataset...")

        # Group objects by topic
        topics = self._group_by_topic(knowledge_objects)
        logger.info(f"Found {len(topics)} topics")

        all_samples = []
        for topic_name, objects in topics.items():
            samples = self._build_samples_for_topic(topic_name, objects)
            all_samples.extend(samples)

        # Save dataset
        dataset_path = self._save_dataset(all_samples)

        logger.info(f"Dataset built: {len(all_samples)} samples from {len(topics)} topics")
        return dataset_path

    def _group_by_topic(self, objects: List) -> Dict[str, List]:
        """Group Knowledge Objects by topic/heading."""
        topics = defaultdict(list)
        current_topic = "General"

        for obj in objects:
            if obj.kind == "heading":
                current_topic = obj.content
            else:
                topics[current_topic].append(obj)

        return dict(topics)

    def _build_samples_for_topic(self, topic: str, objects: List) -> List[TrainingSample]:
        """Build all training samples for one topic."""
        samples = []

        # Combine all content for this topic
        combined_content = " ".join([obj.content for obj in objects if obj.kind == "text"])
        definitions = [obj.content for obj in objects if "is" in obj.content.lower() or "defined" in obj.content.lower()]
        algorithms = [obj.content for obj in objects if obj.kind == "algorithm"]
        has_diagram = any(obj.kind == "image" for obj in objects)

        # 1. Knowledge → Question (multiple bloom levels)
        for bloom in self.BLOOM_LEVELS:
            style = self._bloom_to_style(bloom)
            question = self._generate_question_text(topic, style, bloom)
            samples.append(TrainingSample(
                knowledge=combined_content[:1000],
                bloom=bloom,
                marks=self._bloom_to_marks(bloom),
                question_style=style,
                question=question,
                subject=self.subject_code,
                sample_type="knowledge_to_question",
            ))

        # 2. Knowledge → Expected Answer
        samples.append(TrainingSample(
            knowledge=combined_content[:1000],
            answer_graph=combined_content[:500],
            bloom="L2",
            marks=10,
            question_style="Explain",
            question=f"Explain {topic}.",
            expected_answer=combined_content[:500],
            subject=self.subject_code,
            sample_type="knowledge_to_answer",
        ))

        # 3. Knowledge → Bloom Level Classification
        for bloom in self.BLOOM_LEVELS:
            samples.append(TrainingSample(
                knowledge=combined_content[:500],
                bloom=bloom,
                question=f"Explain {topic}.",
                subject=self.subject_code,
                sample_type="knowledge_to_bloom",
            ))

        # 4. Knowledge → Question Style
        for style in self.QUESTION_STYLES[:10]:  # Limit to 10
            samples.append(TrainingSample(
                knowledge=combined_content[:500],
                question_style=style,
                question=f"{style} {topic}.",
                subject=self.subject_code,
                sample_type="knowledge_to_style",
            ))

        # 5. Knowledge → Diagram Requirement
        samples.append(TrainingSample(
            knowledge=combined_content[:500],
            bloom="L2",
            question=f"Explain {topic}.",
            expected_answer="Yes, a diagram is required." if has_diagram else "No diagram needed.",
            subject=self.subject_code,
            sample_type="knowledge_to_diagram",
        ))

        # 6. Knowledge → Marks Allocation
        for marks in [5, 10, 15]:
            samples.append(TrainingSample(
                knowledge=combined_content[:500],
                marks=marks,
                question=f"Explain {topic}.",
                subject=self.subject_code,
                sample_type="knowledge_to_marks",
            ))

        # 7. Definition → Concept
        for definition in definitions[:3]:
            samples.append(TrainingSample(
                knowledge=definition,
                question=definition,
                expected_answer=topic,
                subject=self.subject_code,
                sample_type="definition_to_concept",
            ))

        # 8. Concept → Definition
        if definitions:
            samples.append(TrainingSample(
                knowledge=topic,
                question=f"Define {topic}.",
                expected_answer=definitions[0],
                subject=self.subject_code,
                sample_type="concept_to_definition",
            ))

        # 9. Algorithm → Question
        for algo in algorithms[:3]:
            samples.append(TrainingSample(
                knowledge=algo,
                bloom="L3",
                marks=10,
                question_style="Trace",
                question=f"Trace the {topic} algorithm with an example.",
                subject=self.subject_code,
                sample_type="algorithm_to_question",
            ))

        # 10. Diagram → Question
        if has_diagram:
            samples.append(TrainingSample(
                knowledge=combined_content[:500],
                bloom="L3",
                marks=10,
                question_style="Illustrate",
                question=f"Draw and explain the diagram for {topic}.",
                subject=self.subject_code,
                sample_type="diagram_to_question",
            ))

        return samples

    def _bloom_to_style(self, bloom: str) -> str:
        """Map Bloom level to question style."""
        mapping = {
            "L1": "Define",
            "L2": "Explain",
            "L3": "Apply",
            "L4": "Compare",
            "L5": "Evaluate",
            "L6": "Design",
        }
        return mapping.get(bloom, "Explain")

    def _bloom_to_marks(self, bloom: str) -> int:
        """Map Bloom level to typical marks."""
        mapping = {
            "L1": 2,
            "L2": 5,
            "L3": 10,
            "L4": 10,
            "L5": 15,
            "L6": 15,
        }
        return mapping.get(bloom, 10)

    def _generate_question_text(self, topic: str, style: str, bloom: str) -> str:
        """Generate a question text."""
        templates = {
            "Define": f"Define {topic}.",
            "Explain": f"Explain {topic} with a suitable example.",
            "Describe": f"Describe the working of {topic}.",
            "Compare": f"Compare {topic} with alternative approaches.",
            "Differentiate": f"Differentiate between {topic} and related concepts.",
            "Trace": f"Trace the {topic} algorithm for a given input.",
            "Construct": f"Construct {topic} for the given problem.",
            "Analyze": f"Analyze the complexity of {topic}.",
            "Evaluate": f"Evaluate the advantages and limitations of {topic}.",
            "Design": f"Design a solution using {topic}.",
            "Illustrate": f"Illustrate {topic} with a neat diagram.",
            "Solve": f"Solve the given problem using {topic}.",
            "List": f"List the key properties of {topic}.",
            "State": f"State and explain {topic}.",
            "Summarize": f"Summarize the applications of {topic}.",
            "Demonstrate": f"Demonstrate {topic} with an example.",
            "Justify": f"Justify the use of {topic}.",
            "Critique": f"Critique the limitations of {topic}.",
            "Propose": f"Propose an improvement to {topic}.",
            "Develop": f"Develop an algorithm based on {topic}.",
            "Apply": f"Apply {topic} to solve the given problem.",
        }
        return templates.get(style, f"Explain {topic}.")

    def _save_dataset(self, samples: List[TrainingSample]) -> str:
        """Save dataset to disk."""
        subject_dir = self.output_dir / self.subject_code
        subject_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSONL
        dataset_file = subject_dir / "train.jsonl"
        with open(dataset_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

        # Save metadata
        metadata = {
            "subject_code": self.subject_code,
            "total_samples": len(samples),
            "sample_types": defaultdict(int),
            "bloom_distribution": defaultdict(int),
            "marks_distribution": defaultdict(int),
        }
        for sample in samples:
            metadata["sample_types"][sample.sample_type] += 1
            metadata["bloom_distribution"][sample.bloom] += 1
            metadata["marks_distribution"][str(sample.marks)] += 1

        metadata_file = subject_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        return str(dataset_file)
