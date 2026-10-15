"""Non-destructive edit stack.

Each tool can be independently enabled, disabled, or tuned. Its step is
uniquely identified by (section, operation), so navigating between tools
never applies the same effect twice by accident. Steps are replayed in
the order in which tools were first enabled.
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
        self.result = self.source.copy()
        self.steps: list[EditStep] = []
        self.active_index: int | None = None

    def leave_tool(self) -> None:
        # No visual side effect. Retained for existing client code.
        self.active_index = None

    def is_enabled(self, section: str, operation: str) -> bool:
        return any(s.section == section and s.operation == operation
                   for s in self.steps)

    def apply(self, section: str, operation: str, **parameters) -> np.ndarray:
        if operation == "original":
            return self.reset_all()
        for step in self.steps:
            if step.section == section and step.operation == operation:
                step.parameters = parameters
                return self._render()

        self.steps.append(EditStep(section, operation, parameters))
        return self._render()

    def remove(self, section: str, operation: str) -> np.ndarray:
        self.steps = [
            step for step in self.steps
            if not (step.section == section and step.operation == operation)
        ]
        self.active_index = None
        return self._render()

    def reset_section(self, section: str) -> np.ndarray:
        self.steps = [s for s in self.steps if s.section != section]
        self.active_index = None
        return self._render()

    def reset_all(self) -> np.ndarray:
        self.steps.clear()
        self.active_index = None
        return self._render()

    def _render(self) -> np.ndarray:
        current = self.source.copy()
        for step in self.steps:
            current = run_tool(current, step.operation, **step.parameters)
        self.result = current
        return current
