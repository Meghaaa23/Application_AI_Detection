"""
ssrf_guard.py  —  centralized SSRF egress protection for the web scanner.

F1 remediation. The scanner legitimately fetches caller-supplied target URLs,
so we cannot simply block all requests. Instead we enforce a single policy at
the point where a hostname is turned into an IP address and a socket is opened:

  * Public scanning  — only globally routable HTTP/HTTPS destinations are
    allowed. Loopback, RFC1918, link-local (incl. 169.254.169.254 cloud
    metadata), IPv6 ULA/loopback/link-local, unspecified, CGNAT and other
    non-global ranges are rejected.
  * Internal scanning — a non-global destination is permitted ONLY when it is
    explicitly listed in the operator-configured allowlist
    (SSRF_INTERNAL_ALLOWLIST). There is deliberately no "allow all private"
    switch.

Enforcement happens in two places that together cover every network sink:

  1. Python HTTP clients (urllib / httpx / requests): ``socket.getaddrinfo`` is
     wrapped once at import. It is INERT until a scan activates a policy via
     ``guarded_egress(...)``. Because validation runs on the *resolved* address
     at connect time, it defeats DNS rebinding, alternate IP representations,
     and public->private redirects (each hop re-resolves).
  2. Playwright/Chromium (separate process, own DNS): ``install_browser_guard``
     attaches a ``context.route`` handler that validates every navigation and
     redirect against the active policy.

The active policy is stored in a thread-visible module global (not a ContextVar)
so it also applies to work dispatched to executor threads via run_in_executor.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import threading
from contextlib import contextmanager
from typing import List, Optional, Set
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Keep a handle to the real resolver BEFORE we patch it, so our own lookups and
# the patched function never recurse.
_ORIG_GETADDRINFO = socket.getaddrinfo


class SSRFBlocked(OSError):
    """Raised when a destination violates the egress policy.

    Subclasses OSError so that HTTP clients which catch connection errors treat
    a blocked destination like an unreachable host instead of crashing.
    """


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class EgressPolicy:
    def __init__(
        self,
        allow_internal: bool,
        internal_hosts: Optional[Set[str]] = None,
        internal_networks: Optional[list] = None,
        infra_hosts: Optional[Set[str]] = None,
    ) -> None:
        self.allow_internal = allow_internal
        self.internal_hosts = {h.strip(".").lower() for h in (internal_hosts or set())}
        self.internal_networks = list(internal_networks or [])
        # Hosts the application itself must reach during a scan (e.g. a local
        # LLM backend). Allowed regardless of allow_internal, matched by host.
        self.infra_hosts = {h.strip(".").lower() for h in (infra_hosts or set())}

    def ip_allowed(self, host: Optional[str], ip_str: str) -> bool:
        """Is a connection to ``ip_str`` (resolved from ``host``) permitted?"""
        try:
            ip = _normalize_ip(ip_str)
        except ValueError:
            return False

        # Publicly routable addresses are always allowed for public scanning.
        if ip.is_global:
            return True

        host_l = (host or "").strip(".").lower()

        # App's own backend endpoints (e.g. local LLM) stay reachable.
        if host_l and host_l in self.infra_hosts:
            return True

        if not self.allow_internal:
            return False

        # Internal scanning: destination must be explicitly authorized.
        if host_l and host_l in self.internal_hosts:
            return True
        for net in self.internal_networks:
            if ip.version == net.version and ip in net:
                return True
        return False

    def describe(self) -> str:
        return (
            f"EgressPolicy(allow_internal={self.allow_internal}, "
            f"hosts={sorted(self.internal_hosts)}, "
            f"networks={[str(n) for n in self.internal_networks]}, "
            f"infra={sorted(self.infra_hosts)})"
        )


def load_policy_from_env() -> EgressPolicy:
    """Build the policy from configuration.

    SSRF_INTERNAL_ALLOWLIST: comma-separated list of hostnames and/or CIDR
    networks that authorized internal/testbed scanning may reach. An empty or
    unset value means public-only (no internal destinations allowed).
    """
    raw = os.getenv("SSRF_INTERNAL_ALLOWLIST", "") or ""
    hosts: Set[str] = set()
    networks: list = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        net = _try_network(token)
        if net is not None:
            networks.append(net)
        else:
            # Allow "host" or "host:port" — port is ignored for matching.
            host_only = token.split("/")[0]
            if host_only.startswith("[") and "]" in host_only:
                host_only = host_only[1:host_only.index("]")]
            elif host_only.count(":") == 1:
                host_only = host_only.split(":")[0]
            hosts.add(host_only.strip(".").lower())

    allow_internal = bool(hosts or networks)

    # Narrowly permit the app's own local LLM backend during scans so that
    # running in local-provider mode does not break scanning. Azure (the
    # default) is public and needs no allowance.
    infra: Set[str] = set()
    if os.getenv("ACTIVE_MODEL_PROVIDER", "azure").lower() == "local":
        llama = os.getenv("LLAMA_BASE_URL", "http://localhost:8080")
        h = urlsplit(llama if "://" in llama else "http://" + llama).hostname
        if h:
            infra.add(h.strip(".").lower())

    return EgressPolicy(
        allow_internal=allow_internal,
        internal_hosts=hosts,
        internal_networks=networks,
        infra_hosts=infra,
    )


def _try_network(token: str):
    try:
        if "/" in token:
            return ipaddress.ip_network(token, strict=False)
        # A bare IP is treated as a /32 or /128.
        ip = ipaddress.ip_address(token)
        return ipaddress.ip_network(f"{ip}/{ip.max_prefixlen}", strict=False)
    except ValueError:
        return None


def _normalize_ip(ip_str: str) -> ipaddress._BaseAddress:
    ip = ipaddress.ip_address(ip_str)
    # Unwrap IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) so it is classified as v4.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


# ---------------------------------------------------------------------------
# URL parsing + resolution
# ---------------------------------------------------------------------------
class ParsedTarget:
    def __init__(self, url: str, scheme: str, host: str, port: Optional[int], ips: List[str]):
        self.url = url
        self.scheme = scheme
        self.host = host
        self.port = port
        self.ips = ips


def parse_target(url: str) -> ParsedTarget:
    """Robustly parse a URL, rejecting malformed/ambiguous/unsafe forms.

    Does NOT resolve DNS. Raises SSRFBlocked on anything that is not a clean
    http(s) URL with a bare hostname and no userinfo.
    """
    if not isinstance(url, str) or not url.strip():
        raise SSRFBlocked("empty or non-string URL")
    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        raise SSRFBlocked(f"malformed URL: {exc}") from exc

    scheme = (parts.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise SSRFBlocked(f"scheme {scheme!r} not allowed (only http/https)")

    # Reject userinfo tricks like http://expected@attacker/ or http://a:b@host/.
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise SSRFBlocked("userinfo in URL is not allowed")

    host = parts.hostname  # already lowercased, IPv6 brackets stripped
    if not host:
        raise SSRFBlocked("missing host")
    # Reject obviously malformed hostnames.
    if any(c.isspace() for c in host) or "\x00" in host:
        raise SSRFBlocked("invalid characters in host")

    try:
        port = parts.port  # raises ValueError on an out-of-range/invalid port
    except ValueError as exc:
        raise SSRFBlocked(f"invalid port: {exc}") from exc

    return ParsedTarget(url=url, scheme=scheme, host=host, port=port, ips=[])


def resolve_host(host: str) -> List[str]:
    """Resolve a host to every A/AAAA address using the real resolver."""
    try:
        infos = _ORIG_GETADDRINFO(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SSRFBlocked(f"could not resolve host {host!r}: {exc}") from exc
    ips = []
    seen = set()
    for info in infos:
        ip = info[4][0]
        if ip not in seen:
            seen.add(ip)
            ips.append(ip)
    if not ips:
        raise SSRFBlocked(f"host {host!r} resolved to no addresses")
    return ips


def validate_url(url: str, policy: EgressPolicy) -> ParsedTarget:
    """Parse, resolve, and authorize a URL. Raises SSRFBlocked if disallowed.

    The decision is made on the *resolved* addresses, not the hostname string,
    so alternate IP representations and names that resolve to private space are
    caught. Blocks if ANY resolved address is disallowed (handles mixed/rebind
    answer sets).
    """
    target = parse_target(url)
    target.ips = resolve_host(target.host)
    for ip in target.ips:
        if not policy.ip_allowed(target.host, ip):
            raise SSRFBlocked(
                f"destination {target.host!r} -> {ip} is not permitted by the "
                f"egress policy ({policy.describe()})"
            )
    return target


# ---------------------------------------------------------------------------
# Active policy (thread-visible global) + getaddrinfo enforcement
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_policy_stack: List[EgressPolicy] = []


def current_policy() -> Optional[EgressPolicy]:
    with _lock:
        return _policy_stack[-1] if _policy_stack else None


@contextmanager
def guarded_egress(policy: EgressPolicy):
    """Activate connect-time egress enforcement for the duration of a scan."""
    with _lock:
        _policy_stack.append(policy)
    try:
        yield policy
    finally:
        with _lock:
            if _policy_stack:
                _policy_stack.pop()


def _patched_getaddrinfo(host, *args, **kwargs):
    results = _ORIG_GETADDRINFO(host, *args, **kwargs)
    policy = current_policy()
    if policy is None:
        return results  # inert outside a guarded scan
    for info in results:
        ip = info[4][0]
        if not policy.ip_allowed(host, ip):
            raise SSRFBlocked(
                f"SSRF guard blocked connection: {host!r} -> {ip} "
                f"is not permitted by the active egress policy"
            )
    return results


def install() -> None:
    """Install the socket-level guard once (idempotent)."""
    if getattr(socket.getaddrinfo, "_ssrf_guarded", False):
        return
    _patched_getaddrinfo._ssrf_guarded = True  # type: ignore[attr-defined]
    socket.getaddrinfo = _patched_getaddrinfo  # type: ignore[assignment]


# Install on import so every network sink is covered as soon as the scanner
# package is loaded. It stays inert until a scan activates a policy.
install()


# ---------------------------------------------------------------------------
# Browser (Playwright) enforcement
# ---------------------------------------------------------------------------
async def install_browser_guard(context) -> None:
    """Attach a request/redirect validator to a Playwright browser context.

    No-op when no scan policy is active. Chromium resolves DNS itself, so a
    narrow DNS-rebinding race remains between this check and the browser's own
    connection; the Python-layer guard has no such gap.
    """
    policy = current_policy()
    if policy is None:
        return

    async def _route_handler(route):
        try:
            validate_url(route.request.url, policy)
        except SSRFBlocked:
            try:
                await route.abort()
            except Exception:
                pass
            return
        except Exception:
            # Parsing/resolution error: fail closed for the browser path too.
            try:
                await route.abort()
            except Exception:
                pass
            return
        try:
            await route.continue_()
        except Exception:
            pass

    try:
        await context.route("**/*", _route_handler)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not install browser SSRF guard: %s", exc)
