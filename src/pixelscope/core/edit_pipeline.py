"""Non-destructive image edit history with section-scoped resets.

Operations are replayed from the immutable source image, so editing an earlier
step never destroys changes made by later tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .processing import run_tool


@dataclass
class EditStep:
    section: str
    operation: str
    parameters: dict = field(default_factory=dict)


class EditPipeline:
    def __init__(self, source: np.ndarray) -> None:
        self.source = source.copy()
        self.steps: list[EditStep] = []
        self.result = source.copy()
        self.active_index: int | None = None

    def leave_tool(self) -> None:
        """Finish the active live preview without clearing the history."""
        self.active_index = None

    def apply(self, section: str, operation: str, **parameters) -> np.ndarray:
        """Apply immediately. Live slider updates replace only their own step."""
        if operation == "original":
            return self.reset_all()

        if operation == "adjust":
            # The three main sliders are one section-wide adjustment block.
            # Re-entering Adjustments modifies that block, replaying all later
            # steps as well rather than accidentally double-applying contrast.
            existing = next(
                (i for i, step in enumerate(self.steps)
                 if step.section == "Adjustments" and step.operation == "adjust"),
                None,
            )
            if existing is not None:
                self.steps[existing].parameters = parameters
                self.active_index = existing
                return self._render()

        index = self.active_index
        if (
            index is not None
            and index < len(self.steps)
            and self.steps[index].section == section
            and self.steps[index].operation == operation
        ):
            self.steps[index].parameters = parameters
        else:
            self.steps.append(EditStep(section, operation, parameters))
            self.active_index = len(self.steps) - 1

        return self._render()

    def reset_section(self, section: str) -> np.ndarray:
        self.steps = [step for step in self.steps if step.section != section]
        self.active_index = None
        return self._render()

    def reset_all(self) -> np.ndarray:
        self.steps.clear()
        self.active_index = None
        return self._render()

    def _render(self) -> np.ndarray:
        result = self.source.copy()
        for step in self.steps:
            result = run_tool(result, step.operation, **step.parameters)
        self.result = result
        return result
