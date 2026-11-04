"""Response-adaptive technique selection for injection scans."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Iterable, List, Optional, Set

from .schema import (
    InjectionFamily,
    Objective,
    ResponseOutcome,
    SUCCESS_OUTCOMES,
)

class AdaptiveAction(str, Enum):
    TRY_PAYLOAD = "try_payload"
    TRY_MUTATIONS = "try_mutations"
    FOLLOWUP = "followup"
    STOP = "stop"

_REFUSAL_JUMPS = (
    "persona_override_dan",
    "fictional_no_sysprompt",
    "hypothetical_frame",
    "developer_impersonation",
    "cot_framing",
    "translation_wrapper",
)

_IGNORE_JUMPS = (
    "instruction_reset",
    "priority_claim",
    "delimiter_xml",
    "debug_mode_trigger",
    "markdown_header_frame",
    "task_substitution",
)

_PARTIAL_JUMPS = (
    "priority_claim",
    "debug_mode_trigger",
    "yaml_frame",
    "fragment_reassembly",
    "json_field_split",
)

_ESCALATION_OBJECTIVES = (
    Objective.REVEAL_SYSPROMPT,
    Objective.ROLE_TAKEOVER,
    Objective.IGNORE_POLICY,
    Objective.CALL_TOOL,
)

@dataclass
class AdaptiveDecision:
    action: AdaptiveAction
    payload: Optional[Dict] = None
    reason: str = ""

@dataclass
class ResponseAdaptivePolicy:
    """Budgeted search over a payload catalog conditioned on oracle outcomes."""

    payloads: List[Dict]
    full_matrix: bool = False
    max_steps: int = 24

    _by_technique: Dict[str, Dict] = field(default_factory=dict, init=False)
    _by_objective: Dict[Objective, List[Dict]] = field(default_factory=dict, init=False)
    _queue: Deque[str] = field(default_factory=deque, init=False)
    _tried: Set[str] = field(default_factory=set, init=False)
    _steps: int = field(default=0, init=False)
    _last_technique: Optional[str] = field(default=None, init=False)
    _pending_mutations_for: Optional[str] = field(default=None, init=False)
    _history: List[Dict] = field(default_factory=list, init=False)
    _seeded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        by_obj: Dict[Objective, List[Dict]] = defaultdict(list)
        for p in self.payloads:
            tech = p.get("technique")
            if not tech:
                continue
            self._by_technique[tech] = p
            obj = p.get("objective")
            if isinstance(obj, Objective):
                by_obj[obj].append(p)
        self._by_objective = dict(by_obj)
        self._seed_queue()

    def _seed_queue(self) -> None:
        """Seed with low-complexity direct echo_token techniques first."""
        direct_echo = [
            p for p in self.payloads
            if p.get("family") == InjectionFamily.DIRECT
            and p.get("objective") == Objective.ECHO_TOKEN
        ]
        direct_echo.sort(key=lambda p: (p.get("complexity", 5), p.get("technique", "")))
        for p in direct_echo:
            self._enqueue(p["technique"], front=False)

        others = [
            p for p in self.payloads
            if p.get("technique") not in self._queue
        ]
        others.sort(
            key=lambda p: (
                0 if p.get("family") == InjectionFamily.DIRECT else 1,
                p.get("complexity", 5),
                p.get("technique", ""),
            )
        )
        for p in others:
            self._enqueue(p["technique"], front=False)
        self._seeded = True

    def _enqueue(self, technique: str, *, front: bool = False) -> None:
        if not technique or technique in self._tried or technique not in self._by_technique:
            return
        if technique in self._queue:
            if not front:
                return
            try:
                self._queue.remove(technique)
            except ValueError:
                return
        if front:
            self._queue.appendleft(technique)
        else:
            self._queue.append(technique)

    def _enqueue_jumps(self, names: Iterable[str]) -> None:
        for name in reversed(list(names)):
            self._enqueue(name, front=True)

    def next(self) -> AdaptiveDecision:
        if self._pending_mutations_for:
            tech = self._pending_mutations_for
            self._pending_mutations_for = None
            pdef = self._by_technique.get(tech)
            if pdef:
                return AdaptiveDecision(
                    AdaptiveAction.TRY_MUTATIONS, pdef, "partial -> mutate same technique"
                )

        if self._steps >= self.max_steps and not self.full_matrix:
            return AdaptiveDecision(AdaptiveAction.STOP, reason="max adaptive steps")

        while self._queue:
            tech = self._queue.popleft()
            if tech in self._tried:
                continue
            pdef = self._by_technique.get(tech)
            if not pdef:
                continue
            self._steps += 1
            self._last_technique = tech
            return AdaptiveDecision(
                AdaptiveAction.TRY_PAYLOAD, pdef, "queued technique"
            )

        return AdaptiveDecision(AdaptiveAction.STOP, reason="payload queue exhausted")

    def record(
        self,
        technique: str,
        outcome: ResponseOutcome,
        *,
        mutation_label: str = "base",
    ) -> Optional[AdaptiveDecision]:
        """Update policy after an oracle result. May return an immediate follow-up action."""
        self._tried.add(technique)
        self._history.append({
            "technique": technique,
            "outcome": outcome.value if isinstance(outcome, ResponseOutcome) else str(outcome),
            "mutation": mutation_label,
        })

        if outcome in SUCCESS_OUTCOMES:
            if self.full_matrix:
                self._enqueue_escalation(technique)
                return None
            return AdaptiveDecision(AdaptiveAction.STOP, reason="success fail-early")

        if outcome == ResponseOutcome.REFUSED:
            self._enqueue_jumps(_REFUSAL_JUMPS)
            if mutation_label == "base":
                return AdaptiveDecision(
                    AdaptiveAction.FOLLOWUP,
                    self._by_technique.get(technique),
                    "refused -> followup then jailbreak jumps",
                )
            return None

        if outcome == ResponseOutcome.PARTIAL:
            self._enqueue_jumps(_PARTIAL_JUMPS)
            if mutation_label == "base":
                self._pending_mutations_for = technique
                return AdaptiveDecision(
                    AdaptiveAction.TRY_MUTATIONS,
                    self._by_technique.get(technique),
                    "partial -> mutations on same technique",
                )
            return None

        if outcome in (ResponseOutcome.IGNORED, ResponseOutcome.HALLUCINATED, ResponseOutcome.ERROR):
            self._enqueue_jumps(_IGNORE_JUMPS)
            return None

        return None

    def _enqueue_escalation(self, technique: str) -> None:
        current = self._by_technique.get(technique) or {}
        cur_obj = current.get("objective")
        for obj in _ESCALATION_OBJECTIVES:
            if obj == cur_obj:
                continue
            for p in self._by_objective.get(obj, []):
                self._enqueue(p["technique"], front=True)
                break

    @property
    def history(self) -> List[Dict]:
        return list(self._history)

    @property
    def tried(self) -> Set[str]:
        return set(self._tried)
