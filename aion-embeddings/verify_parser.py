from core.question_parser import QuestionParser
from core.pattern_learner import PatternLearner
import pprint

parser = QuestionParser()
questions = parser.parse_file("test_paper.txt", subject="AI")

print(f"Parsed {len(questions)} questions:")
for q in questions:
    print(f"Q: {q['question_text']} | Marks: {q['marks']} | Bloom: {q['bloom_level']} | Type: {q['question_type']}")

learner = PatternLearner()
learner.learn_from_questions(questions)

print("\nLearned Patterns:")
stats = learner.get_statistics()
for p in stats["top_patterns"]:
    print(p)

pairs = learner.generate_training_pairs(questions)
print(f"\nGenerated {len(pairs)} pairs")
for i, p in enumerate(pairs):
    print(f"Pair {i}: Anchor='{p['anchor']}' | Pos='{p['positive']}'")
