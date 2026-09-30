"""Unit tests for rule-based SPO triple extraction."""

import pytest

from core.knowledge.triple_extractor import (
    RuleBasedTripleExtractor,
    RawTriple,
)


@pytest.fixture
def extractor():
    return RuleBasedTripleExtractor()


# -----------------------------------------------------------------------------
# is_a
# -----------------------------------------------------------------------------


def test_is_a_positive(extractor):
    triples = extractor.extract("Docker is a containerization platform.")
    assert len(triples) == 1
    t = triples[0]
    assert t.subject == "Docker"
    assert t.predicate == "is_a"
    assert t.object == "containerization platform"


def test_is_an_positive(extractor):
    triples = extractor.extract("A hypervisor is an abstract layer.")
    assert len(triples) == 1
    assert triples[0].subject == "hypervisor"
    assert triples[0].predicate == "is_a"
    assert triples[0].object == "abstract layer"


# -----------------------------------------------------------------------------
# consists_of
# -----------------------------------------------------------------------------


def test_consists_of(extractor):
    triples = extractor.extract(
        "Cloud architecture consists of a front end and a back end."
    )
    assert any(t.predicate == "consists_of" for t in triples)
    t = next(t for t in triples if t.predicate == "consists_of")
    assert t.subject == "Cloud architecture"
    assert "front end" in t.object


# -----------------------------------------------------------------------------
# used_for
# -----------------------------------------------------------------------------


def test_is_used_for(extractor):
    triples = extractor.extract(
        "Cloud storage is used for remote data persistence."
    )
    assert len(triples) == 1
    assert triples[0].predicate == "used_for"
    assert triples[0].subject == "Cloud storage"
    assert triples[0].object == "remote data persistence"


def test_is_used_to(extractor):
    triples = extractor.extract("Docker is used to deploy applications.")
    assert len(triples) == 1
    assert triples[0].predicate == "used_for"


# -----------------------------------------------------------------------------
# requires / enables
# -----------------------------------------------------------------------------


def test_requires(extractor):
    triples = extractor.extract("Kubernetes requires a container runtime.")
    assert len(triples) == 1
    assert triples[0].predicate == "requires"
    assert triples[0].subject == "Kubernetes"
    assert triples[0].object == "container runtime"


def test_enables(extractor):
    triples = extractor.extract("Encryption enables secure communication.")
    assert len(triples) == 1
    assert triples[0].predicate == "enables"


# -----------------------------------------------------------------------------
# Multiple sentences
# -----------------------------------------------------------------------------


def test_multiple_sentences(extractor):
    text = (
        "Docker is a containerization platform. "
        "Kubernetes requires a container runtime. "
        "Encryption enables secure communication."
    )
    triples = extractor.extract(text)
    assert len(triples) == 3
    predicates = {t.predicate for t in triples}
    assert predicates == {"is_a", "requires", "enables"}


# -----------------------------------------------------------------------------
# Noise rejection
# -----------------------------------------------------------------------------


def test_empty_input(extractor):
    assert extractor.extract("") == []
    assert extractor.extract("   ") == []


def test_question_rejected(extractor):
    assert extractor.extract("What is Docker?") == []


def test_too_short_rejected(extractor):
    assert extractor.extract("Docker is.") == []


def test_too_long_rejected(extractor):
    long_sentence = " ".join(["word"] * 70) + " is a thing."
    assert extractor.extract(long_sentence) == []


def test_non_alpha_rejected(extractor):
    # "12345 is 67890." — no alphabetic content
    assert extractor.extract("12345 is 67890.") == []


# -----------------------------------------------------------------------------
# Deduplication
# -----------------------------------------------------------------------------


def test_deduplication_case_insensitive(extractor):
    text = (
        "Docker is a containerization platform. "
        "Docker is a containerization platform."
    )
    triples = extractor.extract(text)
    assert len(triples) == 1


# -----------------------------------------------------------------------------
# Pattern precedence
# -----------------------------------------------------------------------------


def test_used_for_wins_over_is_a(extractor):
    # "X is used for Y" must not be matched by the `is a` pattern
    triples = extractor.extract("Cloud is used for storage.")
    assert len(triples) == 1
    assert triples[0].predicate == "used_for"


# -----------------------------------------------------------------------------
# Confidence
# -----------------------------------------------------------------------------


def test_confidence_in_valid_range(extractor):
    triples = extractor.extract("Docker is a containerization platform.")
    for t in triples:
        assert 0.0 <= t.confidence <= 1.0


def test_min_confidence_filters():
    strict = RuleBasedTripleExtractor(min_confidence=0.95)
    triples = strict.extract("X has Y.")  # "has" is 0.65 base
    assert triples == []


def test_invalid_min_confidence_rejected():
    with pytest.raises(ValueError):
        RuleBasedTripleExtractor(min_confidence=1.5)
