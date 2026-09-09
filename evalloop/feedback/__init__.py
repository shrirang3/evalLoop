"""Compile verified failures into training data.

Nothing here invents a target. A failure whose correct answer is unknown is
dropped and counted, and that count is the useful output (`plan/000` P4).
"""

from evalloop.feedback.compile import SELECTION_SOURCE, Dataset, DatasetRow, build_dpo

__all__ = ["SELECTION_SOURCE", "Dataset", "DatasetRow", "build_dpo"]
