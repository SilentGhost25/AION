"""
Rasch 1PL Item Response Theory difficulty calibration.

Estimates the difficulty parameter b_i for a question by simulating
student responses at multiple ability levels and fitting the model:

    P(correct | θ, b) = 1 / (1 + exp(-(θ - b)))

This is the 1-parameter logistic (1PL) / Rasch model. The difficulty
parameter b is on the same scale as the ability parameter θ; higher b
means the question is harder.

Estimation is via Newton-Raphson on the log-likelihood. The log-likelihood
for Rasch is strictly concave in b, so Newton-Raphson converges to the
unique MLE from any starting point (subject to boundary handling).

Boundary cases:
    - All responses correct → MLE at -∞; clamped to difficulty_bounds[0].
    - All responses incorrect → MLE at +∞; clamped to difficulty_bounds[1].
    - The standard error is reported as +∞ in these cases.

Determinism:
    - Given a fixed rng seed and a deterministic student_sim_fn, the
      output is deterministic.
    - Given a stochastic student_sim_fn, the outcome distribution is
      deterministic but individual runs vary.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


StudentSimFn = Callable[[str, float], float]
"""
Signature: (question_text, ability_theta) -> probability in [0, 1].

The calibrator treats the return as P(correct) for a student at that
ability level. Values outside [0, 1] are clamped.
"""


# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------


@dataclass
class DifficultyEstimate:
    """
    Result of calibrating one question.

    difficulty      : Rasch b parameter, in difficulty_bounds (default [-3, 3]).
    standard_error  : Standard error of the MLE. +inf for all-correct/all-wrong.
    n_responses     : Number of simulated responses used in the fit.
    converged       : Whether Newton-Raphson converged within tolerance.
    notes           : Free-form diagnostic. Empty for normal cases.
    """
    difficulty: float
    standard_error: float
    n_responses: int
    converged: bool
    notes: str = ""


# -----------------------------------------------------------------------------
# Calibrator
# -----------------------------------------------------------------------------


class IRTCalibrator:
    """
    Rasch 1PL difficulty calibrator.

    Parameters
    ----------
    student_sim_fn : StudentSimFn
        Injected simulator. Returns P(correct) for a given question and ability.
    ability_levels : List[float] | None
        Ability values θ at which to simulate. Default (-2, -1, 0, 1, 2).
    samples_per_level : int
        Number of sampled responses per ability level. Default 3.
        Total responses = len(ability_levels) * samples_per_level.
    difficulty_bounds : Tuple[float, float]
        Clamp for the estimated b. Default (-3.0, 3.0).
    max_iterations : int
        Newton-Raphson iteration cap. Default 50.
    tolerance : float
        Convergence threshold on |Δb|. Default 1e-6.
    rng : random.Random | None
        Random number generator. If None, a fresh Random() is used.
        Pass a seeded instance for reproducible sampling.
    clock : Callable[[], float] | None
        Optional time source for telemetry. Not used in the fit itself.
    """

    DEFAULT_ABILITY_LEVELS = (-2.0, -1.0, 0.0, 1.0, 2.0)

    def __init__(
        self,
        student_sim_fn: StudentSimFn,
        ability_levels: Optional[List[float]] = None,
        samples_per_level: int = 3,
        difficulty_bounds: Tuple[float, float] = (-3.0, 3.0),
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        rng: Optional[random.Random] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if student_sim_fn is None:
            raise ValueError("student_sim_fn is required")
        if samples_per_level < 1:
            raise ValueError("samples_per_level must be >= 1")
        if len(difficulty_bounds) != 2 or difficulty_bounds[0] >= difficulty_bounds[1]:
            raise ValueError("difficulty_bounds must be (low, high) with low < high")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")

        self._sim = student_sim_fn
        self._ability_levels = list(ability_levels or self.DEFAULT_ABILITY_LEVELS)
        self._samples_per_level = samples_per_level
        self._bounds = difficulty_bounds
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._rng = rng or random.Random()
        self._clock = clock or time.monotonic

    # ------------------------------------------------------------- Public API

    def calibrate(self, question_text: str) -> DifficultyEstimate:
        """
        Estimate the difficulty of a single question.

        Simulates N responses at each ability level, samples binary
        outcomes, and fits the Rasch model.
        """
        responses = self._simulate_responses(question_text)
        if not responses:
            raise ValueError("No responses generated (empty ability levels?)")
        return self._fit(responses)

    def calibrate_batch(self, questions: List[str]) -> List[DifficultyEstimate]:
        """Calibrate a list of questions sequentially."""
        return [self.calibrate(q) for q in questions]

    # ------------------------------------------------------------- Simulation

    def _simulate_responses(self, question_text: str) -> List[Tuple[float, int]]:
        """
        Return a list of (ability_theta, response_0_or_1) pairs.

        Each response is sampled from P(correct | ability) using the
        injected RNG. Same question + same RNG state → same outcome.
        """
        responses: List[Tuple[float, int]] = []
        for theta in self._ability_levels:
            for _ in range(self._samples_per_level):
                try:
                    p = float(self._sim(question_text, theta))
                except Exception as e:
                    # Simulator crash on one sample should not abort the fit.
                    # Skip the sample; the caller sees n_responses lower than expected.
                    print(f"[IRT] simulator raised at θ={theta}: {e}", flush=True)
                    continue

                # Clamp to [0, 1] — simulators should stay in range, but
                # tolerate numerical drift.
                if math.isnan(p):
                    continue
                p = max(0.0, min(1.0, p))

                # Sample binary outcome
                outcome = 1 if self._rng.random() < p else 0
                responses.append((theta, outcome))

        return responses

    # ------------------------------------------------------------- Fit

    def _fit(self, responses: List[Tuple[float, int]]) -> DifficultyEstimate:
        """
        Fit Rasch b via Newton-Raphson.

        Log-likelihood (up to constants):
            L(b) = Σ [y_i * (θ_i - b) - log(1 + exp(θ_i - b))]

        Score:
            L'(b) = Σ [sigmoid(θ_i - b) - y_i]

        Hessian:
            L''(b) = -Σ [sigmoid(θ_i - b) * (1 - sigmoid(θ_i - b))]
        """
        n = len(responses)
        if n == 0:
            return DifficultyEstimate(
                difficulty=0.0,
                standard_error=float("inf"),
                n_responses=0,
                converged=False,
                notes="no_responses",
            )

        # Boundary cases — all correct or all incorrect.
        total_correct = sum(y for _, y in responses)
        if total_correct == n:
            return DifficultyEstimate(
                difficulty=self._bounds[0],
                standard_error=float("inf"),
                n_responses=n,
                converged=True,
                notes="all_correct_boundary",
            )
        if total_correct == 0:
            return DifficultyEstimate(
                difficulty=self._bounds[1],
                standard_error=float("inf"),
                n_responses=n,
                converged=True,
                notes="all_incorrect_boundary",
            )

        # Newton-Raphson on b.
        b = 0.0  # start at 0
        converged = False
        for iteration in range(self._max_iterations):
            score, hessian = self._score_and_hessian(b, responses)
            if abs(hessian) < 1e-12:
                # Degenerate curvature — stop. Fit is unreliable.
                break

            step = score / hessian
            b_new = b - step

            # Clamp to bounds during iteration
            if b_new < self._bounds[0]:
                b_new = self._bounds[0]
            elif b_new > self._bounds[1]:
                b_new = self._bounds[1]

            if abs(b_new - b) < self._tolerance:
                b = b_new
                converged = True
                break
            b = b_new
        else:
            converged = False  # exhausted iterations

        # Standard error = 1 / sqrt(-Hessian at MLE)
        _, hessian_at_mle = self._score_and_hessian(b, responses)
        if hessian_at_mle >= 0 or math.isnan(hessian_at_mle):
            se = float("inf")
        else:
            se = 1.0 / math.sqrt(-hessian_at_mle)

        return DifficultyEstimate(
            difficulty=round(b, 6),
            standard_error=round(se, 6) if math.isfinite(se) else se,
            n_responses=n,
            converged=converged,
            notes="" if converged else "max_iterations_exhausted",
        )

    def _score_and_hessian(
        self,
        b: float,
        responses: List[Tuple[float, int]],
    ) -> Tuple[float, float]:
        """
        Return (score, hessian) at the given b.

        Both are sums over all responses.
        """
        score = 0.0
        hessian = 0.0
        for theta, y in responses:
            p = self._sigmoid(theta - b)
            score += p - y
            hessian -= p * (1.0 - p)
        return score, hessian

    @staticmethod
    def _sigmoid(z: float) -> float:
        """
        Numerically stable sigmoid. Avoids overflow in exp(z) for large |z|.
        """
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)
