
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional, Tuple
from urllib.parse import urlparse

_MULTI_PART_SUFFIXES = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk",
    "com.au", "net.au", "org.au",
    "co.nz", "org.nz",
    "co.jp", "ne.jp", "or.jp",
    "com.br", "com.mx", "com.ar",
    "co.in", "com.sg", "com.hk",
    "co.za", "com.tr",
})

class OwnershipClass(str, Enum):
    """Whose product/org the surface belongs to relative to the scan target."""
    SAME_SITE = "same_site"
    VENDOR_WIDGET = "vendor_widget"
    EXTERNAL = "external"

class Reachability(str, Enum):
    """Capability fact — independent of authorization."""
    SAME_ORIGIN = "same_origin"
    CROSS_ORIGIN_FRAME = "cross_origin_frame"
    UNKNOWN = "unknown"

class InjectionPolicy(str, Enum):
    """Authorization decision — overridable, not welded to origin."""
    INJECT = "inject"
    DETECT_ONLY = "detect_only"
    SKIP_INJECT = "skip_inject"

@dataclass
class ScopeDecision:
    ownership: str
    reachability: str
    injection_policy: str
    reason: str
    target_etld1: str = ""
    surface_etld1: str = ""
    provider: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def scope_label(self) -> str:
        """Back-compat SurfaceScope-ish label for fingerprint consumers."""
        if self.ownership == OwnershipClass.SAME_SITE.value:
            return "app_owned"
        if self.ownership == OwnershipClass.VENDOR_WIDGET.value:
            return "third_party"
        return "unknown"

@dataclass
class ScopePolicy:
    """Runtime authorization overrides (CLI / scanner config)."""
    include_thirdparty: bool = False
    authorize_external: bool = False
    authorize_support_queue: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

class SurfaceKind(str, Enum):
    """What the iframe/product actually is — independent of vendor brand."""
    CHAT = "chat"
    SUPPORT_QUEUE = "support_queue"
    FEEDBACK = "feedback"
    UNKNOWN = "unknown"

_FEEDBACK_URL_RE = re.compile(
    r"feedback-web|feedback[_-]?fetcher|/survey|/nps\b|/csat\b|rating-form",
    re.I,
)
_SUPPORT_QUEUE_URL_RE = re.compile(
    r"conversations-visitor|/threads/utk|"
    r"widget\.intercom|js\.driftt\.com|zendesk\.com|zopim|"
    r"cdn\.livechatinc|tawk\.to|tidio\.co|static\.olark|"
    r"wchat\.freshchat|usemessages\.com",
    re.I,
)

def classify_surface_kind(url: str, marker: str = "") -> str:
    """Classify iframe/product by URL shape — provider brand alone is not enough."""
    blob = f"{url or ''} {marker or ''}"
    if _FEEDBACK_URL_RE.search(blob):
        return SurfaceKind.FEEDBACK.value
    if _SUPPORT_QUEUE_URL_RE.search(blob):
        return SurfaceKind.SUPPORT_QUEUE.value
    if re.search(r"/chat|/assistant|/copilot|/conversation", blob, re.I):
        return SurfaceKind.CHAT.value
    return SurfaceKind.UNKNOWN.value

def side_effect_blocks_inject(
    surface_kind: str,
    policy: Optional[ScopePolicy] = None,
) -> Tuple[bool, str]:
    """Second gate after ownership: does submit fire a real-world action?"""
    policy = policy or ScopePolicy()
    if surface_kind == SurfaceKind.FEEDBACK.value:
        return True, "feedback_not_chat"
    if surface_kind == SurfaceKind.SUPPORT_QUEUE.value:
        if policy.authorize_support_queue:
            return False, ""
        return True, "support_queue_side_effect"
    return False, ""

def hostname_of(url_or_host: str) -> str:
    raw = (url_or_host or "").strip().lower()
    if not raw:
        return ""
    if "://" not in raw:
        host = raw.split("/")[0]
    else:
        host = urlparse(raw).hostname or ""
    if host.startswith("www."):
        pass
    return host

def registrable_domain(url_or_host: str) -> str:
    """Approximate eTLD+1 (registrable domain) via a small public-suffix set."""
    host = hostname_of(url_or_host)
    if not host or host in ("localhost",) or host.replace(".", "").isdigit():
        return host
    host = host.rstrip(".")
    if ":" in host and not host.replace(":", "").replace(".", "").isalnum():
        return host
    labels = host.split(".")
    if len(labels) < 2:
        return host
    last2 = ".".join(labels[-2:])
    if last2 in _MULTI_PART_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last2

def same_origin(a: str, b: str) -> bool:
    try:
        pa, pb = urlparse(a), urlparse(b)
        if not pb.netloc:
            return True
        return (pa.scheme, pa.hostname) == (pb.scheme, pb.hostname)
    except Exception:
        return False

def same_registrable_domain(target_url: str, surface_url: str) -> bool:
    ta = registrable_domain(target_url)
    sa = registrable_domain(surface_url)
    return bool(ta and sa and ta == sa)

def classify_reachability(target_url: str, surface_url: str = "") -> str:
    """Capability fact. Cross-origin iframes are still Playwright-reachable."""
    if not surface_url or surface_url in ("about:blank",):
        return Reachability.SAME_ORIGIN.value
    if same_origin(target_url, surface_url):
        return Reachability.SAME_ORIGIN.value
    return Reachability.CROSS_ORIGIN_FRAME.value

def classify_injection_scope(
    target_url: str,
    surface_url: str = "",
    *,
    provider: str = "",
    is_vendor_marker: bool = False,
    policy: Optional[ScopePolicy] = None,
) -> ScopeDecision:
    """Map ownership + authorization policy (not a capability gate)."""
    policy = policy or ScopePolicy()
    target_etld1 = registrable_domain(target_url)
    surface_etld1 = registrable_domain(surface_url) if surface_url else target_etld1
    reach = classify_reachability(target_url, surface_url or target_url)

    if not surface_url or same_registrable_domain(target_url, surface_url):
        return ScopeDecision(
            ownership=OwnershipClass.SAME_SITE.value,
            reachability=reach,
            injection_policy=InjectionPolicy.INJECT.value,
            reason="same_registrable_domain",
            target_etld1=target_etld1,
            surface_etld1=surface_etld1 or target_etld1,
            provider=provider,
        )

    vendor = bool(provider) or is_vendor_marker
    if vendor:
        if policy.include_thirdparty:
            return ScopeDecision(
                ownership=OwnershipClass.VENDOR_WIDGET.value,
                reachability=reach,
                injection_policy=InjectionPolicy.INJECT.value,
                reason="vendor_widget_explicitly_authorized",
                target_etld1=target_etld1,
                surface_etld1=surface_etld1,
                provider=provider,
            )
        return ScopeDecision(
            ownership=OwnershipClass.VENDOR_WIDGET.value,
            reachability=reach,
            injection_policy=InjectionPolicy.SKIP_INJECT.value,
            reason="vendor_widget_default_skip",
            target_etld1=target_etld1,
            surface_etld1=surface_etld1,
            provider=provider,
        )

    if policy.authorize_external:
        return ScopeDecision(
            ownership=OwnershipClass.EXTERNAL.value,
            reachability=reach,
            injection_policy=InjectionPolicy.INJECT.value,
            reason="external_surface_explicitly_authorized",
            target_etld1=target_etld1,
            surface_etld1=surface_etld1,
            provider=provider,
        )
    return ScopeDecision(
        ownership=OwnershipClass.EXTERNAL.value,
        reachability=reach,
        injection_policy=InjectionPolicy.DETECT_ONLY.value,
        reason="external_surface_detect_only",
        target_etld1=target_etld1,
        surface_etld1=surface_etld1,
        provider=provider,
    )

def match_vendor_marker(
    text: str,
    markers: Tuple[Tuple[str, str], ...],
) -> Tuple[str, str]:
    """Return (marker, provider_name) if text matches a vendor marker."""
    low = (text or "").lower()
    for marker, name in markers:
        if marker in low:
            return marker, name
    return "", ""
