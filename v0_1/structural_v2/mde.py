"""
AION Structural Architecture v2 — Mark Distribution Engine (MDE)
=================================================================
Deterministic calculation of sub-question mark distributions under
BALANCED, PRIMARY_HEAVY, PROGRESSIVE, and CUSTOM policies.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
from .contracts import DistributionPolicy


class DistributionError(Exception):
    """Raised when mark distribution preconditions or invariants are violated."""
    pass


class MarkDistributionEngine:
    """Computes deterministic mark distributions for sub-questions."""

    @classmethod
    def compute(
        cls,
        total: int,
        n: int,
        policy: DistributionPolicy = DistributionPolicy.BALANCED,
        custom: Optional[List[int]] = None,
    ) -> Tuple[int, ...]:
        """
        Compute sub-question mark distribution tuple D of length n summing to total.
        All slots must get >= 1 mark.
        """
        if n < 1:
            raise DistributionError(f"Sub-question count n must be >= 1, got {n}")
        if total < n:
            raise DistributionError(f"Total marks {total} must be >= sub-question count {n}")

        if policy == DistributionPolicy.BALANCED:
            base = total // n
            remainder = total % n
            D = [base] * n
            for i in range(remainder):
                D[i] += 1

        elif policy == DistributionPolicy.PRIMARY_HEAVY:
            weight_table = {1: 1.0, 2: 0.6, 3: 0.5, 4: 0.4}
            ratio = weight_table.get(n, 1.0 / n)
            primary = max(1, math.ceil(total * ratio))

            if n > 1:
                if primary >= total - (n - 1):
                    primary = total - (n - 1)
                remaining = total - primary
                rest = list(cls.compute(remaining, n - 1, DistributionPolicy.BALANCED))
                D = [primary] + rest
            else:
                D = [total]

        elif policy == DistributionPolicy.PROGRESSIVE:
            triangle = n * (n + 1) / 2
            raw = [total * (n - i) / triangle for i in range(n)]
            D = [math.floor(x) for x in raw]

            deficit = total - sum(D)
            for i in range(deficit):
                D[i] += 1

            # Safety: enforce minimum 2 marks for non-L1 positions (i >= 1)
            for i in range(1, n):
                if D[i] < 2:
                    borrow_from = max(range(i), key=lambda k: D[k])
                    if D[borrow_from] - 1 >= D[i] + 1:
                        D[borrow_from] -= 1
                        D[i] += 1

        elif policy == DistributionPolicy.CUSTOM:
            if custom is None:
                raise DistributionError("Custom policy requires 'custom' parameter")
            if len(custom) != n:
                raise DistributionError(f"Custom distribution length {len(custom)} != n {n}")
            if sum(custom) != total:
                raise DistributionError(f"Custom distribution sum {sum(custom)} != total {total}")
            if not all(x >= 1 for x in custom):
                raise DistributionError("All custom marks must be >= 1")
            D = list(custom)
        else:
            raise DistributionError(f"Unknown policy {policy}")

        # Universal Invariants Check
        if sum(D) != total:
            raise DistributionError(f"Distribution sum {sum(D)} != total {total}")
        if len(D) != n:
            raise DistributionError(f"Distribution length {len(D)} != n {n}")
        if not all(x >= 1 for x in D):
            raise DistributionError(f"All slots in distribution must be >= 1, got {D}")

        if policy != DistributionPolicy.CUSTOM:
            D = sorted(D, reverse=True)

        return tuple(D)
