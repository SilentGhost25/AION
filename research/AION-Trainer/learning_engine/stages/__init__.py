# learning_engine/stages/__init__.py

"""

Concept Understanding Stages — the progression every concept must pass

through before AION is allowed to generate questions about it.



    Discovered   -> the concept exists in the knowledge graph

    Recognised   -> AION can identify it from a definition fragment

    Understood   -> AION can paraphrase it correctly

    Connected    -> AION knows its relationships to other concepts

    Explainable  -> AION can write a coherent explanation

    Answerable   -> AION can produce a correct expected answer

    Questionable -> AION can generate valid exam questions

    ExaminerLevel-> AION matches professor-level question style



Gate rule: no stage can be skipped.

The LearningOrchestrator checks the stage before invoking

any downstream operation.

"""



from enum import IntEnum





class ConceptStage(IntEnum):

    DISCOVERED    = 0

    RECOGNISED    = 1

    UNDERSTOOD    = 2

    CONNECTED     = 3

    EXPLAINABLE   = 4

    ANSWERABLE    = 5

    QUESTIONABLE  = 6

    EXAMINER_LEVEL = 7



    def label(self) -> str:

        return self.name.replace("_", " ").title()



    def next(self) -> "ConceptStage":

        nxt = self.value + 1

        return ConceptStage(min(nxt, ConceptStage.EXAMINER_LEVEL))



    def can_generate_questions(self) -> bool:

        return self >= ConceptStage.QUESTIONABLE



    def can_generate_answers(self) -> bool:

        return self >= ConceptStage.ANSWERABLE
