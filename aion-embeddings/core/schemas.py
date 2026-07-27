from pydantic import BaseModel, Field, validator, model_validator
from typing import List, Literal, Optional
from enum import Enum

class BloomLevel(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"
    L6 = "L6"

class QuestionPart(BaseModel):
    """A single part of a question (a, b, c, etc.)"""
    part_letter: str = Field(..., regex="^[a-z]$", description="Part identifier: a, b, c, etc.")
    text: str = Field(..., min_length=20, max_length=500, description="Question text")
    marks: int = Field(..., ge=2, le=20, description="Marks for this part")
    bloom_level: BloomLevel = Field(..., description="Bloom's Taxonomy level")

    @validator('text')
    def text_quality(cls, v):
        # Reject trivially short or obviously AI-hallucinated text
        if v.count("?") == 0 and "define" not in v.lower() and "explain" not in v.lower():
            raise ValueError("Question must be a proper question (contain '?' or action verb)")
        return v

class VTUQuestion(BaseModel):
    """A VTU question that can have multiple parts (a, b, c) with OR choices."""
    question_number: int = Field(..., ge=1, le=10, description="Question number")
    parts: List[QuestionPart] = Field(..., min_items=1, max_items=3, description="Question parts (a, b, c, etc.)")
    internal_choice: bool = Field(..., description="Does this question have internal choice (OR)?")
    
    @validator('parts')
    def validate_parts_sequence(cls, v):
        """Ensure parts are in order (a, b, c) with no gaps"""
        expected = [chr(ord('a') + i) for i in range(len(v))]
        actual = [p.part_letter for p in v]
        if actual != expected:
            raise ValueError(f"Parts must be sequential (a, b, c...). Got {actual}")
        return v
    
    @validator('parts')
    def validate_parts_marks(cls, v, values):
        """If internal_choice, all parts should have equal marks"""
        if values.get('internal_choice') and len(v) > 1:
            marks = [p.marks for p in v]
            if len(set(marks)) > 1:
                raise ValueError(f"Internal choice parts must have equal marks. Got {marks}")
        return v

class VTUModule(BaseModel):
    """A VTU Module containing questions"""
    module_number: int = Field(..., ge=1, le=5, description="Module 1-5")
    questions: List[VTUQuestion] = Field(..., min_items=1, max_items=3, description="Questions in this module")
    total_marks_in_module: int = Field(..., ge=20, le=20, description="Must total exactly 20 marks per module")
    
    @validator('total_marks_in_module')
    def validate_module_marks(cls, v, values):
        """Ensure total marks = sum of all question part marks"""
        if 'questions' in values:
            questions = values['questions']
            actual_total = sum(part.marks for q in questions for part in q.parts)
            if actual_total != v:
                raise ValueError(f"Module marks must sum to {v}, got {actual_total}")
        return v

class VTUQuestionPaper(BaseModel):
    """Complete 100-mark CBCS question paper"""
    subject_code: str = Field(..., regex="^[0-9]{2}[A-Z]{2}[0-9]{2}$", description="Format: 18CS51")
    subject_name: str = Field(..., min_length=5, max_length=100)
    exam_type: Literal["CIE", "SEE"] = Field(..., description="Internal or End Semester")
    duration_minutes: int = Field(..., ge=120, le=180)
    modules: List[VTUModule] = Field(..., min_length=5, max_length=5, description="Exactly 5 modules")
    
    @model_validator(mode="after")
    def validate_marks(self):
        """Ensure exactly 100 marks total and 20 per module."""
        total = 0
        for module in self.modules:
            # We assume internal choice means we take the MAX marks among questions in a module to calculate paper total
            # For VTU, one full question (or its parts) from a module must be answered for 20 marks.
            # But the requirement here is a paper where the choices sum to what? 
            # VTU CBCS modules generally have 2 questions of 20 marks each (with internal choice).
            # The previous validation assumed `total_marks_in_module` is exactly 20, but that means the whole module is 20 marks worth of questions?
            # Actually, typically there are two questions of 20 marks each, so 40 marks printed per module, but the student answers 20.
            # Let's enforce that the printed questions represent exactly what the student answers or what is assigned.
            # We'll rely on the earlier `total_marks_in_module` for now and just sum it up.
            
            # Let's just sum all the `total_marks_in_module`
            total += module.total_marks_in_module
        
        if total != 100:
            raise ValueError(f"Total marks {total}, expected 100")
        
        return self

    @validator('modules')
    def validate_bloom_distribution(cls, v):
        """Ensure balanced Bloom's Taxonomy distribution"""
        all_blooms = []
        for mod in v:
            for q in mod.questions:
                for part in q.parts:
                    all_blooms.append(part.bloom_level)
        
        bloom_counts = {b: all_blooms.count(b) for b in BloomLevel}
        
        l2_l3 = bloom_counts.get(BloomLevel.L2, 0) + bloom_counts.get(BloomLevel.L3, 0)
        if len(all_blooms) > 0 and l2_l3 < len(all_blooms) * 0.6:
            raise ValueError(f"At least 60% of questions should be L2-L3. Got {l2_l3}/{len(all_blooms)}")
        
        return v
