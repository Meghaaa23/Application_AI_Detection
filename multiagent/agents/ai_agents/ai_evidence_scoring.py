from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

class EvidenceClass(str, Enum):
    BEHAVIORAL = "behavioral"
    LIVE_SURFACE = "live_surface"

    ROUTE_CONFIRMED = "route_confirmed"
    PASSIVE_STRUCTURAL = "passive_structural"
    PASSIVE_HINT = "passive_hint"

    PASSIVE = "passive_hint"
    BRAND = "brand"

class Detection(str, Enum):
    CONFIRMED = "confirmed"

    ROUTE_CONFIRMED = "route_confirmed"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    NOT_DETECTED = "not_detected"

_CLASS_RANK: Dict[EvidenceClass, int] = {
    EvidenceClass.BEHAVIORAL: 6,
    EvidenceClass.LIVE_SURFACE: 5,
    EvidenceClass.ROUTE_CONFIRMED: 4,
    EvidenceClass.PASSIVE_STRUCTURAL: 3,
    EvidenceClass.PASSIVE_HINT: 2,
    EvidenceClass.BRAND: 1,
}

_CLASS_RULES: List[Tuple[re.Pattern, EvidenceClass]] = [
    (re.compile(r"^behavior_model_reported$"), EvidenceClass.PASSIVE_STRUCTURAL),
    (re.compile(r"^behavior_llm_response$"), EvidenceClass.PASSIVE_HINT),
    (re.compile(r"^behavior_"), EvidenceClass.BEHAVIORAL),
    (re.compile(r"^error_"), EvidenceClass.ROUTE_CONFIRMED),
    (re.compile(r"_endpoint_200$"), EvidenceClass.LIVE_SURFACE),
    (re.compile(r"_endpoint_(?:401_403|405)$"), EvidenceClass.ROUTE_CONFIRMED),
    (re.compile(r"_ai_(?:endpoint|field)$"), EvidenceClass.ROUTE_CONFIRMED),
    (re.compile(
        r"^(?:sse_streaming_header|ai_response_header)$"
    ), EvidenceClass.LIVE_SURFACE),
    (re.compile(r"^sse_chunked_transfer$"), EvidenceClass.PASSIVE_HINT),
    (re.compile(r"^framework_header$"), EvidenceClass.BRAND),
    (re.compile(r"^framework_identified$"), EvidenceClass.BRAND),
    (re.compile(r"^framework_cookie$"), EvidenceClass.PASSIVE_HINT),
    (re.compile(r"^(?:openapi|graphql)_schema$"), EvidenceClass.PASSIVE_STRUCTURAL),
    (re.compile(r"^behavior_chatbot_ui_"), EvidenceClass.BEHAVIORAL),
    (re.compile(r"^chatbot_ui_live_surface$"), EvidenceClass.LIVE_SURFACE),
    (re.compile(r"^chatbot_ui_structural$"), EvidenceClass.PASSIVE_STRUCTURAL),
    (re.compile(r"^(?:js_|sse_in_js|websocket_|robots_|sitemap_|csp_|chatbot_ui_)"),
     EvidenceClass.PASSIVE_HINT),
    (re.compile(r"^(?:meta_ai_tag|dom_assistant_role|html_ai_form|"
                r"favicon_known_framework|page_title_ai|html_brand_signal)$"),
     EvidenceClass.BRAND),
]

_CONFIDENCE: Dict[Detection, str] = {
    Detection.CONFIRMED: "high",
    Detection.ROUTE_CONFIRMED: "medium",
    Detection.PROBABLE: "medium",
    Detection.POSSIBLE: "low",
    Detection.NOT_DETECTED: "none",
}

PROBABLE_MIN_HINT = 4
PROBABLE_MIN_PASSIVE = PROBABLE_MIN_HINT

def classify_evidence(key: str) -> EvidenceClass:
    for pattern, cls in _CLASS_RULES:
        if pattern.search(key):
            return cls
    return EvidenceClass.BRAND

def evidence_weight(key: str) -> int:
    return _CLASS_RANK[classify_evidence(key)]

@dataclass
class EvidenceGroup:
    behavioral: List[str] = field(default_factory=list)
    live_surface: List[str] = field(default_factory=list)
    route_confirmed: List[str] = field(default_factory=list)
    passive_structural: List[str] = field(default_factory=list)
    passive_hint: List[str] = field(default_factory=list)
    brand: List[str] = field(default_factory=list)

    @property
    def passive(self) -> List[str]:
        """Legacy combined view (hints only — structural has its own bucket)."""
        return self.passive_hint

    def bucket(self, cls: EvidenceClass) -> List[str]:
        return {
            EvidenceClass.BEHAVIORAL: self.behavioral,
            EvidenceClass.LIVE_SURFACE: self.live_surface,
            EvidenceClass.ROUTE_CONFIRMED: self.route_confirmed,
            EvidenceClass.PASSIVE_STRUCTURAL: self.passive_structural,
            EvidenceClass.PASSIVE_HINT: self.passive_hint,
            EvidenceClass.PASSIVE: self.passive_hint,
            EvidenceClass.BRAND: self.brand,
        }[cls]

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            EvidenceClass.BEHAVIORAL.value: self.behavioral,
            EvidenceClass.LIVE_SURFACE.value: self.live_surface,
            EvidenceClass.ROUTE_CONFIRMED.value: self.route_confirmed,
            EvidenceClass.PASSIVE_STRUCTURAL.value: self.passive_structural,
            EvidenceClass.PASSIVE_HINT.value: self.passive_hint,
            "passive": self.passive_hint,
            EvidenceClass.BRAND.value: self.brand,
        }

def group_evidence(keys: Iterable[str]) -> EvidenceGroup:
    group = EvidenceGroup()
    for key in dict.fromkeys(k for k in keys if k):
        group.bucket(classify_evidence(key)).append(key)
    return group

@dataclass
class DetectionResult:
    detection: Detection
    confidence: str
    is_ai: bool
    has_behavioral: bool
    has_live_surface: bool

    has_route_confirmed: bool
    brand_only: bool
    classes: Dict[str, List[str]]

    def to_dict(self) -> Dict:
        return {
            "detection": self.detection.value,
            "confidence": self.confidence,
            "is_ai": self.is_ai,
            "has_behavioral": self.has_behavioral,
            "has_live_surface": self.has_live_surface,
            "has_route_confirmed": self.has_route_confirmed,
            "brand_only": self.brand_only,
            "classes": self.classes,
        }

def score_evidence(keys: Iterable[str]) -> DetectionResult:
    group = group_evidence(keys)
    has_behavioral = bool(group.behavioral)

    has_live_surface = bool(group.behavioral or group.live_surface)
    has_route_confirmed = bool(group.route_confirmed)
    any_passive = bool(group.passive_structural or group.passive_hint)
    structural = bool(
        group.behavioral or group.live_surface or has_route_confirmed or any_passive
    )

    if has_live_surface:
        detection = Detection.CONFIRMED
    elif has_route_confirmed:
        detection = Detection.ROUTE_CONFIRMED
    elif (
        len(group.passive_structural) >= 1
        or len(group.passive_hint) >= PROBABLE_MIN_HINT
    ):
        detection = Detection.PROBABLE
    elif any_passive or group.brand:
        detection = Detection.POSSIBLE
    else:
        detection = Detection.NOT_DETECTED

    return DetectionResult(
        detection=detection,
        confidence=_CONFIDENCE[detection],

        is_ai=has_live_surface,
        has_behavioral=has_behavioral,
        has_live_surface=has_live_surface,
        has_route_confirmed=has_route_confirmed,
        brand_only=bool(group.brand) and not structural,
        classes=group.as_dict(),
    )

def has_live_surface_evidence(keys: Iterable[str]) -> bool:
    group = group_evidence(keys)
    return bool(group.behavioral or group.live_surface)

def has_route_confirmed_evidence(keys: Iterable[str]) -> bool:
    """True when a route/handler was confirmed (structured 4xx, schema-"""
    return bool(group_evidence(keys).route_confirmed)

def has_behavioral_evidence(keys: Iterable[str]) -> bool:
    return bool(group_evidence(keys).behavioral)

def has_brand_only_evidence(keys: Iterable[str]) -> bool:
    group = group_evidence(keys)
    return bool(group.brand) and not (
        group.behavioral
        or group.live_surface
        or group.route_confirmed
        or group.passive_structural
        or group.passive_hint
    )

_AUTH_REASON_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bbann(?:ed|ing)\b", re.I),                              "account_banned"),
    (re.compile(r"\bsuspend(?:ed|ing)?\b", re.I),                          "account_suspended"),
    (re.compile(r"\bdisabled\b", re.I),                                    "account_disabled"),
    (re.compile(r"\block(?:ed|ed[\s-]?out)\b", re.I),                      "account_locked"),
    (re.compile(r"\b(?:not\s+verified|unverified|verify\s+your\s+email)\b", re.I),
                                                                             "email_not_verified"),
    (re.compile(r"\bexpired\b", re.I),                                     "token_expired"),
    (re.compile(r"\binvalid\s+(?:api[ _-]?key|token|credentials?)\b", re.I),
                                                                             "invalid_credentials"),
    (re.compile(r"\bmissing\s+(?:api[ _-]?key|token|credentials?|authorization)\b", re.I),
                                                                             "missing_credentials"),
    (re.compile(r"\binsufficient\s+(?:permission|scope|privilege)s?\b", re.I),
                                                                             "insufficient_permissions"),
    (re.compile(r"\brate[ _-]?limit", re.I),                               "rate_limited"),
    (re.compile(r"\bip\s+(?:blocked|banned|blacklisted)\b", re.I),         "ip_blocked"),
]

_AUTH_BODY_MESSAGE_KEYS = (
    "message", "error", "detail", "error_description", "reason", "msg", "title",
)

def extract_auth_reason(
    status_code: int, body: str = "", headers: Optional[Dict] = None,
) -> Tuple[str, str]:
    """Classify a 401/403 response into (auth_status, reason)."""
    if status_code == 401:
        auth_status = "authentication_required"
    elif status_code == 403:
        auth_status = "access_forbidden"
    else:
        return "", ""

    body = body or ""
    stripped = body.strip()
    message = ""
    if stripped[:1] in "{[":
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                for key in _AUTH_BODY_MESSAGE_KEYS:
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        message = val.strip()
                        break
        except Exception:
            pass
    if not message:
        message = body[:200]

    www_auth = (headers or {}).get("www-authenticate", "") if headers else ""

    for candidate in (message, www_auth):
        if not candidate:
            continue
        for pattern, reason in _AUTH_REASON_PATTERNS:
            if pattern.search(candidate):
                return auth_status, reason

    return (
        auth_status,
        "invalid_or_missing_credentials" if status_code == 401 else "forbidden",
    )
