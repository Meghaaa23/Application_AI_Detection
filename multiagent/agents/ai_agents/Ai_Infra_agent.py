import os
import re
import asyncio
import logging
import ipaddress
import json
import httpx
from dataclasses import dataclass, field
from typing import Optional, Tuple
from datetime import datetime, timezone
from urllib.parse import urlparse

try:
    from .ai_evidence_scoring import extract_auth_reason
except ImportError:
    try:
        from multiagent.agents.ai_agents.ai_evidence_scoring import extract_auth_reason
    except ImportError:
        from ai_evidence_scoring import extract_auth_reason

logger = logging.getLogger(__name__)

@dataclass
class AIInfraThreat:
    attack_name:     str
    attack_plane:    str
    stride_category: str
    severity:        str
    description:     str
    evidence:        str
    status:          str = "Surface"
    basis:           str = "observed"
    verified:        str = "not_attempted"
    cve_reference:   str = ""
    atlas_technique: str = ""
    mitigation:      str = ""

@dataclass
class AIInfraScanResult:
    target_url:      str
    scan_time:       str
    ai_stack:        dict
    reachability:    str = "unknown"
    threats:         list = field(default_factory=list)
    build_threats:   list = field(default_factory=list)
    deploy_threats:  list = field(default_factory=list)
    runtime_threats: list = field(default_factory=list)
    orch_threats:    list = field(default_factory=list)
    summary:         str = ""
    error:           str = ""
    catch_all_detected: bool = False
    auth_protected_endpoints: list = field(default_factory=list)
    auth_protected_details: list = field(default_factory=list)
    exposed_endpoints: list = field(default_factory=list)

RAY_SERVE_PATHS = ["/-/routes", "/api/serve/routes", "/-/healthz",
                   "/api/serve/deployments", "/api/jobs/", "/dashboard"]
OLLAMA_PATHS = ["/api/tags", "/api/generate", "/api/chat", "/api/show",
                "/api/pull", "/api/delete", "/api/copy"]
VLLM_PATHS = ["/v1/models", "/v1/completions", "/v1/chat/completions", "/health", "/metrics"]
TGI_PATHS = ["/info", "/generate", "/generate_stream", "/metrics", "/health"]
TORCHSERVE_PATHS = ["/ping", "/models", "/api/description", "/metrics",
                    "/predictions", "/management/api/v1/models"]
TRITON_PATHS = ["/v2/models", "/v2/health/ready", "/v2/health/live",
                "/v2/repository/models", "/metrics"]
LANGSERVE_PATHS = ["/chat/playground", "/invoke", "/batch", "/stream", "/docs"]
KUBERNETES_PATHS = ["/api/v1", "/apis", "/healthz", "/readyz", "/metrics",
                    "/api/v1/namespaces", "/api/v1/pods"]
UNIVERSAL_PATHS = ["/metrics", "/health", "/healthz", "/readyz", "/docs",
                   "/openapi.json", "/v1/models", "/v1/chat/completions",
                   "/api/tags", "/api/generate"]
MANAGEMENT_PATHS = ["/api/pull", "/api/delete", "/api/copy", "/api/serve/deployments",
                    "/api/jobs/", "/management/api/v1/models", "/v2/repository/models"]

WRITE_ENDPOINTS = {
    "/api/pull":                 ("model pull",        "CRITICAL", "AML.T0010"),
    "/api/delete":               ("model delete",      "CRITICAL", "AML.T0029"),
    "/api/copy":                 ("model copy",        "HIGH",     "AML.T0010"),
    "/management/api/v1/models": ("model register",    "CRITICAL", "AML.T0010"),
    "/api/serve/deployments":    ("deployment config", "CRITICAL", "AML.T0010"),
    "/api/jobs/":                ("code execution",    "CRITICAL", "AML.T0010"),
    "/v2/repository/models":     ("model register",    "CRITICAL", "AML.T0010"),
}

INFO_ENDPOINTS = {
    "/api/show":     ("model metadata",   "MEDIUM", "AML.T0044", "model_meta"),
    "/api/tags":     ("model list",       "MEDIUM", "AML.T0044", "model_list"),
    "/v1/models":    ("model list",       "MEDIUM", "AML.T0044", "openai_list"),
    "/v2/models":    ("model list",       "MEDIUM", "AML.T0044", "triton_list"),
    "/info":         ("service metadata", "MEDIUM", "AML.T0044", "generic"),
    "/openapi.json": ("API schema",       "LOW",    "AML.T0044", "openapi"),
}

GENERIC_PATHS = {"/docs", "/health", "/healthz", "/readyz", "/metrics",
                 "/openapi.json", "/info", "/ping"}

AI_BODY_SIGNALS = [
    "model", "embedding", "inference", "completion", "generate",
    "choices", "messages", "prompt", "generated_text", "chatflows", "flows",
]

CANARY_PATH = "/__scanner_canary_nonexistent_check_xyz789_do_not_create_this_route"
INFERENCE_PATHS = ["/v1/chat/completions", "/v1/completions", "/api/generate",
                   "/api/chat", "/generate", "/api/v1/prediction", "/api/ask",
                   "/api/v1/run", "/run/predict", "/api/chat-messages",
                   "/api/v1/openai/chat/completions"]
TRAINING_PATHS = ["/train", "/fine-tune", "/finetune", "/training", "/api/train"]

INFERENCE_PATH_RE = re.compile(
    r"/(?:v\d+/)?(?:chat/)?completions(?:/|$)"
    r"|/api/(?:chat|generate|ask|messages|prediction|predict)(?:/|$)"
    r"|/run(?:/|$)|/invoke(?:/$)",
    re.I,
)

OPENAI_CHAT_PROBE = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 1,
    "stream": False,
}

HIGH_IMPACT_CAPS = {"code_execution", "shell", "python", "email_send", "tool_calling", "mcp"}
MED_IMPACT_CAPS  = {"memory", "multi_agent", "web_search", "browser", "retrieval", "rag"}
_SEV_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

SKIP_STATUS = {404, 405, 501, 502, 503, 0}

def _normalize_caps(capabilities) -> set:
    return {str(c).lower() for c in (capabilities or [])}

def _reachability(target_url: str) -> str:
    host = urlparse(target_url).hostname or ""
    if host == "localhost" or host.endswith(".localhost"):
        return "loopback"
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback:
            return "loopback"
        if ip.is_private or ip.is_link_local:
            return "private"
        return "public"
    except ValueError:
        return "public"

def _cap_severity(severity: str, reachability: str) -> str:
    if reachability in ("loopback", "private") and severity in _SEV_ORDER:
        return _SEV_ORDER[max(0, _SEV_ORDER.index(severity) - 1)]
    return severity

def _inference_severity(reachability: str) -> str:
    return "CRITICAL" if reachability == "public" else "MEDIUM"

def _injection_surface_severity(capabilities) -> str:
    caps = _normalize_caps(capabilities)
    if caps & HIGH_IMPACT_CAPS:
        return "CRITICAL"
    if caps & MED_IMPACT_CAPS:
        return "HIGH"
    return "MEDIUM"

def _extraction_surface_severity(capabilities) -> str:
    caps = _normalize_caps(capabilities)
    if caps & HIGH_IMPACT_CAPS:
        return "HIGH"
    return "MEDIUM"

def _json_body(body: str):
    if not body:
        return None
    try:
        return json.loads(body)
    except Exception:
        return None

def _looks_html(ct: str, body: str) -> bool:
    b = (body or "").strip().lower()
    return "html" in (ct or "").lower() or b.startswith(("<!doctype", "<html"))

def _has_data(status: int, body: str) -> bool:
    if status != 200:
        return False
    b = (body or "").strip().lower()
    return b not in ("", "{}", "[]", "null")

def _is_model_list(body: str) -> bool:
    d = _json_body(body)
    return isinstance(d, dict) and isinstance(d.get("models"), list) and len(d["models"]) > 0

def _is_openai_list(body: str) -> bool:
    d = _json_body(body)
    return isinstance(d, dict) and d.get("object") == "list" and isinstance(d.get("data"), list)

def _is_triton_list(body: str) -> bool:
    d = _json_body(body)
    return isinstance(d, (list, dict))

def _is_k8s_body(body: str) -> bool:
    d = _json_body(body)
    if not isinstance(d, dict):
        return False
    if "kind" in d and "apiVersion" in d:
        return True
    kind = d.get("kind")
    if kind in ("APIVersions", "APIGroupList", "NamespaceList", "PodList", "Status"):
        return True
    return (
        str(d.get("apiVersion", "")).startswith("v1")
        and isinstance(d.get("items"), list)
        and kind is not None
    )

def _is_ai_json(body: str) -> bool:
    d = _json_body(body)
    if not isinstance(d, (dict, list)):
        return False
    low = body.lower()
    return any(s in low for s in AI_BODY_SIGNALS) or (
        isinstance(d, dict) and any(k in d for k in
                                    ("choices", "response", "message", "models", "data", "generated_text")))

_INFO_VALIDATORS = {
    "model_list":  _is_model_list,
    "openai_list": _is_openai_list,
    "triton_list": _is_triton_list,
    "model_meta":  lambda b: _json_body(b) is not None,
    "generic":     lambda b: _json_body(b) is not None,
    "openapi":     lambda b: _json_body(b) is not None,
}

class AIInfraAgent:

    def __init__(self, openai_client=None, timeout: int = 15, llm_model: str = None,
                 auth_headers: Optional[dict] = None):
        self.client    = openai_client
        self.timeout   = timeout
        self.llm_model = llm_model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "MultiAgentLLM")
        self.auth_headers = auth_headers or {}

    async def scan(self, target_url: str, ai_stack: Optional[dict] = None) -> AIInfraScanResult:
        target_url   = target_url.rstrip("/")
        reachability = _reachability(target_url)
        result = AIInfraScanResult(
            target_url=target_url,
            scan_time=datetime.now(timezone.utc).isoformat(),
            ai_stack=ai_stack or {},
            reachability=reachability,
        )
        logger.info("[AIInfraAgent] Starting AI infra threat scan -> %s (reachability=%s)",
                    target_url, reachability)
        try:
            for ev in (ai_stack or {}).get("evidence", []):
                if "HTTP 401" in ev or "HTTP 403" in ev:
                    for ep in (ai_stack or {}).get("ml_endpoints", []):
                        if urlparse(ep).path in ev and ep not in result.auth_protected_endpoints:
                            result.auth_protected_endpoints.append(ep)
            for sig in (ai_stack or {}).get("auth_required_paths", []):
                ep = sig.get("endpoint", "") if isinstance(sig, dict) else ""
                if ep and ep not in result.auth_protected_endpoints:
                    result.auth_protected_endpoints.append(ep)
                if ep and isinstance(sig, dict):
                    result.auth_protected_details.append({
                        "url": ep,
                        "status": sig.get("status_code"),
                        "auth_status": sig.get("auth_status", ""),
                        "reason": sig.get("reason", ""),
                    })

            capabilities = (ai_stack or {}).get("capabilities", [])

            exposed, auth_hits, auth_details, catch_all = await self._probe_ai_endpoints(
                target_url, ai_stack)
            result.catch_all_detected = catch_all
            result.exposed_endpoints = [
                (meta or {}).get("url") or (
                    key if str(key).startswith("http") else f"{target_url}{key}"
                )
                for key, meta in exposed.items()
                if (meta or {}).get("verdict") == "exposed"
                or (meta or {}).get("status") == 200
            ]
            for ep in auth_hits:
                if ep not in result.auth_protected_endpoints:
                    result.auth_protected_endpoints.append(ep)
            existing_urls = {d.get("url") for d in result.auth_protected_details}
            for d in auth_details:
                if d.get("url") not in existing_urls:
                    result.auth_protected_details.append(d)
                    existing_urls.add(d.get("url"))

            result.build_threats   = await self._analyze_build_plane(target_url, ai_stack, exposed, reachability)
            result.deploy_threats  = await self._analyze_deployment_plane(target_url, exposed, reachability)
            result.runtime_threats = await self._analyze_runtime_plane(target_url, exposed, capabilities, reachability)
            result.orch_threats    = await self._analyze_orchestration_plane(target_url, exposed, reachability)

            all_threats = (result.build_threats + result.deploy_threats +
                           result.runtime_threats + result.orch_threats)
            result.threats = all_threats
            result.summary = self._synthesize_report(
                target_url, ai_stack, all_threats, catch_all,
                result.auth_protected_endpoints, capabilities, reachability,
                auth_protected_details=result.auth_protected_details)
        except Exception as e:
            logger.error("[AIInfraAgent] Scan error: %s", e, exc_info=True)
            result.error = str(e)
        return result

    async def _detect_catch_all(self, base_url: str) -> bool:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            async with httpx.AsyncClient(timeout=self.timeout, verify=False) as http:
                r = await http.get(f"{base_url}{CANARY_PATH}", headers=headers)
                if r.status_code == 200:
                    logger.warning(
                        "[AIInfraAgent] Catch-all/SPA-fallback detected at %s "
                        "(nonexistent path returned 200). Switching to content-based verification.",
                        base_url)
                    return True
        except Exception:
            pass
        return False

    def _select_paths(self, confirmed_paths: list, ai_stack: Optional[dict] = None,
                      catch_all: bool = False) -> list:
        paths_to_probe = []

        if any(p in confirmed_paths for p in ["/api/tags", "/api/generate", "/api/chat"]):
            paths_to_probe += OLLAMA_PATHS
        if any(p in confirmed_paths for p in ["/v1/models", "/v1/chat/completions"]):
            paths_to_probe += VLLM_PATHS + LANGSERVE_PATHS
        if any(p in confirmed_paths for p in ["/-/routes", "/-/healthz", "/api/serve/deployments"]):
            paths_to_probe += RAY_SERVE_PATHS
        if any(p in confirmed_paths for p in ["/generate", "/generate_stream", "/info"]):
            paths_to_probe += TGI_PATHS
        if any(p in confirmed_paths for p in ["/ping", "/predictions", "/models"]):
            paths_to_probe += TORCHSERVE_PATHS
        if any(p in confirmed_paths for p in ["/v2/models", "/v2/health/ready"]):
            paths_to_probe += TRITON_PATHS

        paths_to_probe += confirmed_paths

        if not catch_all:
            for p in MANAGEMENT_PATHS:
                if p in confirmed_paths:
                    paths_to_probe.append(p)
            if any(p.startswith("/api/v1/namespaces") or p == "/apis" for p in confirmed_paths):
                paths_to_probe += KUBERNETES_PATHS
        else:
            for p in MANAGEMENT_PATHS:
                if p in confirmed_paths:
                    paths_to_probe.append(p)

        return list(dict.fromkeys(p for p in paths_to_probe if p))

    def _allowed_hosts(self, base_url: str, ai_stack: Optional[dict]) -> set:
        """Hosts infra may probe: page origin + fingerprint JS allowlist only."""
        hosts = set()
        page_host = urlparse(base_url).netloc
        if page_host:
            hosts.add(page_host)
        stack = ai_stack or {}
        for h in stack.get("allowed_api_hosts") or []:
            if isinstance(h, str) and h.strip():
                host = urlparse(h if "://" in h else f"https://{h}").netloc or h.strip()
                if host:
                    hosts.add(host)
        return hosts

    async def _probe_ai_endpoints(self, base_url: str, ai_stack: Optional[dict]) -> Tuple[dict, list, bool]:
        is_catch_all = await self._detect_catch_all(base_url)
        if not is_catch_all and (ai_stack or {}).get("spa_catch_all"):
            is_catch_all = True

        confirmed       = list((ai_stack or {}).get("ml_endpoints") or [])
        for sig in (ai_stack or {}).get("auth_required_paths") or []:
            if isinstance(sig, dict) and sig.get("endpoint"):
                confirmed.append(sig["endpoint"])

        page_host = urlparse(base_url).netloc
        allowed = self._allowed_hosts(base_url, ai_stack)

        same_origin_paths: list = []
        cross_targets: list = []
        for e in confirmed:
            if not e:
                continue
            if e.startswith("http://") or e.startswith("https://"):
                parsed = urlparse(e)
                path = parsed.path or "/"
                origin = f"{parsed.scheme}://{parsed.netloc}"
                if parsed.netloc == page_host:
                    same_origin_paths.append(path)
                elif parsed.netloc in allowed:
                    cross_targets.append((origin, path, f"{origin}{path}"))
                else:
                    logger.info(
                        "[AIInfraAgent] skip cross-origin not allowlisted: %s", e)
            else:
                same_origin_paths.append(e if e.startswith("/") else "/" + e)

        paths_to_probe = self._select_paths(
            list(dict.fromkeys(same_origin_paths)), ai_stack, catch_all=is_catch_all)

        seen_cross = set()
        cross_unique = []
        for origin, path, full in cross_targets:
            if full in seen_cross:
                continue
            seen_cross.add(full)
            cross_unique.append((origin, path, full))

        exposed: dict = {}
        auth_hits: list = []
        auth_details: list = []
        work = [
            (base_url, p, base_url + p, is_catch_all, True) for p in paths_to_probe
        ] + [
            (origin, path, full, False, False)
            for origin, path, full in cross_unique
        ]

        async with httpx.AsyncClient(timeout=self.timeout, verify=False, follow_redirects=True) as http:
            tasks   = [self._probe_path(http, b, p) for b, p, _, _, _ in work]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for (_b, path, full_url, use_spa, same_origin), res in zip(work, results):
            if isinstance(res, Exception):
                continue
            status, body, ct = res
            verdict = self._classify(
                status, body, ct, path, bool(same_origin_paths or cross_unique),
                catch_all=use_spa)
            if verdict == "auth":
                if full_url not in auth_hits:
                    auth_hits.append(full_url)
                    auth_status, reason = extract_auth_reason(status, body)
                    auth_details.append({
                        "url": full_url, "status": status,
                        "auth_status": auth_status, "reason": reason,
                    })
                    logger.info(
                        "[AIInfraAgent] AUTH-PROTECTED -> %s [%d] %s (%s)",
                        full_url, status, auth_status, reason)
                continue
            if verdict == "skip":
                continue
            if status in SKIP_STATUS or status in (401, 403):
                continue
            key = path if same_origin else full_url
            exposed[key] = {
                "status": status, "body": body, "verdict": verdict,
                "path": path, "url": full_url,
            }
            tag = "content-verified" if use_spa else "reached"
            logger.info("[AIInfraAgent] EXPOSED (%s) -> %s [%d]", tag, full_url, status)

        return exposed, auth_hits, auth_details, is_catch_all

    def _classify(self, status: int, body: str, ct: str, path: str,
                  have_confirmed: bool, catch_all: bool = False) -> str:
        if _looks_html(ct, body):
            return "skip"
        if status in (401, 403):
            return "auth"
        if status in SKIP_STATUS:
            return "skip"
        if status == 200:
            parsed = _json_body(body)
            b = (body or "").strip().lower()
            is_inference = bool(INFERENCE_PATH_RE.search(path or ""))
            soft_404 = (
                b in ("", "{}", "[]", "null")
                or (isinstance(parsed, dict) and parsed.get("statusCode") in (404, 405))
            )
            if catch_all or path in GENERIC_PATHS:
                ai_shaped = _is_ai_json(body) or any(
                    s in (body or "").lower() for s in AI_BODY_SIGNALS
                )
                if not ai_shaped and not (is_inference and parsed is not None and not soft_404):
                    if not (is_inference and "json" in (ct or "").lower() and not soft_404):
                        return "skip"
            if parsed is None and b in ("", "null"):
                if is_inference and "json" in (ct or "").lower():
                    return "exposed"
                return "skip"
            if parsed is None and not any(s in b for s in AI_BODY_SIGNALS):
                if is_inference and not _looks_html(ct, body):
                    return "exposed"
                return "skip"
            return "exposed"
        if status in (400, 422):
            if catch_all and not (_json_body(body) or any(
                s in (body or "").lower() for s in AI_BODY_SIGNALS
            )):
                return "skip"
            return "reached"
        return "skip"

    async def _probe_path(self, http: httpx.AsyncClient, base: str, path: str) -> Tuple[int, str, str]:
        headers = {"User-Agent": "Mozilla/5.0", **self.auth_headers}
        try:
            if "delete" in path:
                r = await http.request("DELETE", f"{base}{path}", headers=headers)
                return r.status_code, (r.text[:400] if r.text else ""), r.headers.get("content-type", "")
            if any(seg in path for seg in ["/chat", "/generate", "/completions"]):
                payload = OPENAI_CHAT_PROBE if "chat" in path or "completions" in path else {}
                r = await http.post(f"{base}{path}", json=payload, headers=headers)
                return r.status_code, (r.text[:400] if r.text else ""), r.headers.get("content-type", "")
            r = await http.get(f"{base}{path}", headers=headers)
            if r.status_code == 405:
                r = await http.post(f"{base}{path}", json={}, headers=headers)
            return r.status_code, (r.text[:400] if r.text else ""), r.headers.get("content-type", "")
        except Exception:
            return 0, "", ""

    async def _analyze_build_plane(self, url, ai_stack, exposed, reachability):
        threats = []
        confirmed = ai_stack.get("ml_endpoints", []) if ai_stack else []
        confirmed_paths = [urlparse(e).path for e in confirmed]

        if "/api/pull" in exposed:
            threats.append(AIInfraThreat(
                attack_name="Model Weight Substitution (contingent on unverified download)",
                attack_plane="BUILD", stride_category="Tampering",
                severity=_cap_severity("MEDIUM", reachability),
                status="Potential", basis="inferred", verified="no",
                description=("The model pull endpoint is reachable without authentication. If the pull "
                             "path does not verify weight integrity, an attacker on the download path "
                             "could substitute backdoored weights. Whether integrity verification is "
                             "performed cannot be determined by remote scanning; this is inferred."),
                evidence="/api/pull reachable without auth; weight-integrity verification not evaluated",
                atlas_technique="AML.T0010",
                mitigation="Verify model digest before loading. Restrict pull to authenticated admins.",
            ))

        training_hit = [p for p in TRAINING_PATHS if p in confirmed_paths or p in exposed]
        if training_hit:
            threats.append(AIInfraThreat(
                attack_name="Training/Fine-Tuning Endpoint Exposed",
                attack_plane="BUILD", stride_category="Tampering",
                severity=_cap_severity("CRITICAL", reachability),
                status="Potential", basis="observed", verified="not_attempted",
                description=("A training or fine-tuning endpoint is reachable. If the model is fine-tuned "
                             "on interaction data, an attacker could inject poisoned examples."),
                evidence="Training/fine-tuning endpoint reachable at: " + ", ".join(training_hit),
                atlas_technique="AML.T0020",
                mitigation="Gate fine-tuning behind auth. Validate training-data provenance.",
            ))

        return threats

    async def _analyze_deployment_plane(self, url, exposed, reachability):
        threats = []
        serve_signals = ["/-/routes", "/api/serve/deployments", "/-/healthz", "/api/jobs/"]
        detected_serve = [p for p in serve_signals if p in exposed]

        if detected_serve:
            threats.append(AIInfraThreat(
                attack_name="OOM-Triggered Deployment Restart Window",
                attack_plane="DEPLOYMENT", stride_category="Tampering",
                severity=_cap_severity("HIGH", reachability),
                status="Surface", basis="inferred", verified="no",
                description=("Fault-tolerant actor restart creates a recovery window during which "
                             "checkpoints are rehydrated from disk. An attacker with local filesystem "
                             "access could substitute checkpoint content."),
                evidence="Serving management endpoints detected at: " + ", ".join(detected_serve),
                atlas_technique="AML.T0044",
                mitigation="Restrict checkpoint directories with ACLs. Set per-replica memory limits.",
            ))

        if "/-/routes" in exposed:
            d = exposed["/-/routes"]
            if _has_data(d["status"], d.get("body", "")):
                threats.append(AIInfraThreat(
                    attack_name="AI Model Route Table Disclosure",
                    attack_plane="DEPLOYMENT", stride_category="Information Disclosure",
                    severity=_cap_severity("MEDIUM", reachability),
                    status="Confirmed", basis="observed", verified="yes",
                    description="The model serving route table returned deployed model names and routing paths.",
                    evidence=f"HTTP {d['status']} at /-/routes returned content: {d['body'][:200]}",
                    mitigation="Restrict the route table endpoint to the internal network.",
                ))
            else:
                threats.append(AIInfraThreat(
                    attack_name="Route Table Endpoint Reachable",
                    attack_plane="DEPLOYMENT", stride_category="Information Disclosure",
                    severity=_cap_severity("LOW", reachability),
                    status="Potential", basis="observed", verified="not_attempted",
                    description="The route table endpoint is reachable but returned no data on this probe.",
                    evidence=f"HTTP {d['status']} at /-/routes (no data returned on probe)",
                    mitigation="Restrict the route table endpoint to the internal network.",
                ))

        return threats

    async def _analyze_runtime_plane(self, url, exposed, capabilities, reachability):
        threats = []
        exposed_inference = []
        for p in INFERENCE_PATHS:
            if p not in exposed:
                continue
            entry = exposed[p]
            body = entry.get("body", "")
            verdict = entry.get("verdict")
            if _is_ai_json(body) or verdict == "reached":
                if entry.get("status") in SKIP_STATUS:
                    continue
                exposed_inference.append(p)
        exposed_inference = list(dict.fromkeys(exposed_inference))

        if exposed_inference:
            threats.append(AIInfraThreat(
                attack_name="Unauthenticated Inference Endpoint",
                attack_plane="RUNTIME", stride_category="Spoofing",
                severity=_inference_severity(reachability),
                status="Confirmed", basis="observed", verified="not_attempted",
                description=("Model inference endpoints respond without an authentication challenge. Risk "
                             "is deployment-dependent: expected on loopback/private hosts and severe when "
                             f"internet-exposed (reachability={reachability})."),
                evidence="Unauthenticated inference reachable at: " + ", ".join(exposed_inference),
                atlas_technique="AML.T0040",
                mitigation="Require API-key/token auth on inference endpoints if reachable by untrusted users.",
            ))
            threats.append(AIInfraThreat(
                attack_name="Prompt Injection Attack Surface",
                attack_plane="RUNTIME", stride_category="Tampering",
                severity=_injection_surface_severity(capabilities),
                status="Surface", basis="inferred", verified="no",
                description=("The reachable inference endpoint exposes a prompt-injection surface. Impact is "
                             f"scaled by detected capabilities "
                             f"({', '.join(capabilities) if capabilities else 'bare inference, no tools/memory'}). "
                             "Exploitability is confirmed only by the prompt-injection tester."),
                evidence="Direct inference endpoint reachable without auth: " + exposed_inference[0],
                atlas_technique="AML.T0051",
                mitigation="Separate system prompt from user input. Add input/output guardrails.",
            ))
            threats.append(AIInfraThreat(
                attack_name="Model Extraction Surface",
                attack_plane="RUNTIME", stride_category="Information Disclosure",
                severity=_extraction_surface_severity(capabilities),
                status="Surface", basis="inferred", verified="no",
                description="An unauthenticated inference endpoint may permit query-and-record extraction.",
                evidence="Unauthenticated inference reachable; rate limiting not evaluated (single probe)",
                atlas_technique="AML.T0044",
                mitigation="Rate-limit per client. Monitor for extraction-shaped query patterns.",
            ))
            threats.append(AIInfraThreat(
                attack_name="Resource-Exhaustion Surface (long-context / KV-cache)",
                attack_plane="RUNTIME", stride_category="Denial of Service",
                severity="MEDIUM", status="Surface", basis="inferred", verified="no",
                description="Without token or context caps, max-context or concurrent requests could exhaust memory.",
                evidence="Unauthenticated inference reachable; token/context limits not evaluated",
                atlas_technique="AML.T0029",
                mitigation="Cap max_tokens and context length. Set per-pod memory limits.",
            ))

        if "/metrics" in exposed:
            d = exposed["/metrics"]
            if _has_data(d["status"], d.get("body", "")):
                threats.append(AIInfraThreat(
                    attack_name="Metrics Endpoint Information Disclosure",
                    attack_plane="RUNTIME", stride_category="Information Disclosure",
                    severity=_cap_severity("MEDIUM", reachability),
                    status="Confirmed", basis="observed", verified="yes",
                    description="The /metrics endpoint returned operational telemetry.",
                    evidence=f"HTTP {d['status']} at /metrics returned content: {d['body'][:200]}",
                    mitigation="Restrict /metrics to the monitoring network. Require scrape auth.",
                ))

        return threats

    async def _analyze_orchestration_plane(self, url, exposed, reachability):
        threats = []

        for path, (action, severity, atlas) in WRITE_ENDPOINTS.items():
            if path not in exposed:
                continue
            d = exposed[path]
            if d.get("status") not in (200, 400, 422):
                continue
            status_note = "(reached handler, requires body) " if d["status"] in (400, 422) else ""
            threats.append(AIInfraThreat(
                attack_name=f"{action.title()} API Exposed (management endpoint)",
                attack_plane="ORCHESTRATION", stride_category="Elevation of Privilege",
                severity=_cap_severity(severity, reachability),
                status="Potential", basis="observed", verified="not_attempted",
                description=(f"The {action} management endpoint ({path}) is reachable with no auth challenge "
                             f"(request reached the handler rather than returning 401/403). This confirms the "
                             f"route exists and is unauthenticated; it does NOT confirm the {action} action "
                             f"succeeds without credentials."),
                evidence=f"HTTP {d['status']} {status_note}at {path}; {action} not verified",
                atlas_technique=atlas,
                mitigation=f"Restrict {path} to authenticated users or localhost only.",
            ))

        for path, (info_type, severity, atlas, validator_key) in INFO_ENDPOINTS.items():
            if path not in exposed or path in WRITE_ENDPOINTS:
                continue
            d = exposed[path]
            body = d.get("body", "")
            validator = _INFO_VALIDATORS.get(validator_key, lambda b: True)
            if _has_data(d["status"], body) and validator(body):
                threats.append(AIInfraThreat(
                    attack_name=f"{info_type.title()} Disclosure",
                    attack_plane="ORCHESTRATION", stride_category="Information Disclosure",
                    severity=_cap_severity(severity, reachability),
                    status="Confirmed", basis="observed", verified="yes",
                    description=f"The {path} endpoint returned {info_type} without authentication.",
                    evidence=f"HTTP {d['status']} at {path} returned content: {body[:150]}",
                    atlas_technique=atlas,
                    mitigation=f"Restrict {path} to the internal network or add authentication.",
                ))

        k8s_exposed = [p for p in KUBERNETES_PATHS
                       if p in exposed and _is_k8s_body(exposed[p].get("body", ""))]
        if k8s_exposed:
            threats.append(AIInfraThreat(
                attack_name="Kubernetes Control Plane API Reachable",
                attack_plane="ORCHESTRATION", stride_category="Elevation of Privilege",
                severity=_cap_severity("CRITICAL", reachability),
                status="Confirmed", basis="observed", verified="not_attempted",
                description=("Kubernetes API server endpoints returned Kubernetes-shaped responses "
                             "(kind/apiVersion). A reachable API server should never be publicly exposed."),
                evidence="K8s API reachable at: " + ", ".join(k8s_exposed),
                mitigation="The Kubernetes API server must not be publicly reachable. Enable RBAC.",
            ))

        return threats

    @staticmethod
    def _format_auth_detail(url: str, details_by_url: dict) -> str:
        d = details_by_url.get(url)
        if not d:
            return url
        status = d.get("status")
        auth_status = d.get("auth_status") or "auth"
        reason = d.get("reason") or "-"
        return f"{url} [{status} {auth_status}: {reason}]"

    def _synthesize_report(self, url, ai_stack, threats, catch_all_detected=False,
                           auth_protected_endpoints=None, capabilities=None,
                           reachability="unknown", auth_protected_details=None):
        auth_protected_endpoints = auth_protected_endpoints or []
        capabilities = capabilities or []
        details_by_url = {
            d.get("url"): d for d in (auth_protected_details or []) if d.get("url")
        }

        if not threats:
            base = "No AI infrastructure findings."
            if catch_all_detected:
                base += (" Target uses catch-all/SPA routing; exposure verified by response "
                         "content, not status code alone.")
            if auth_protected_endpoints:
                shown = [
                    self._format_auth_detail(ep, details_by_url)
                    for ep in auth_protected_endpoints[:5]
                ]
                base += (f" {len(auth_protected_endpoints)} AI endpoint(s) present but correctly "
                         f"require authentication: {', '.join(shown)}.")
            return base

        confirmed = [t for t in threats if t.status == "Confirmed"]
        surface   = [t for t in threats if t.status == "Surface"]
        potential = [t for t in threats if t.status == "Potential"]

        def _line(t):
            return (f"  - [{t.severity}] {t.attack_name} ({t.attack_plane}/{t.stride_category}) "
                    f"[basis={t.basis}; verified={t.verified}] - {t.evidence}")

        parts = [
            f"AI infrastructure findings for {url}  (reachability={reachability})",
            f"{len(confirmed)} confirmed, {len(surface)} attack-surface, {len(potential)} potential.",
            "Legend: basis=observed|inferred, verified=yes|no|not_attempted.",
        ]
        if capabilities:
            parts.append(f"Detected capabilities: {', '.join(capabilities)}.")
        if confirmed:
            parts.append("\nConfirmed (directly observed):")
            parts += [_line(t) for t in confirmed]
        if surface:
            parts.append("\nAttack surface (inferred from a confirmed finding; not verified):")
            parts += [_line(t) for t in surface]
        if potential:
            parts.append("\nPotential (reachable/inferred; harmful action not verified):")
            parts += [_line(t) for t in potential]
        if auth_protected_endpoints:
            shown = [
                self._format_auth_detail(ep, details_by_url)
                for ep in auth_protected_endpoints[:5]
            ]
            parts.append(f"\nSecure configuration: {len(auth_protected_endpoints)} AI endpoint(s) "
                         f"correctly require authentication: {', '.join(shown)}.")
        if catch_all_detected:
            parts.append("\nExposure verified by response content (catch-all/SPA routing detected).")

        return "\n".join(parts)

    def format_for_stride_report(self, result: AIInfraScanResult) -> dict:
        findings = []
        for t in result.threats:
            findings.append({
                "name":        t.attack_name,
                "category":    t.stride_category,
                "severity":    t.severity,
                "status":      t.status,
                "basis":       t.basis,
                "verified":    t.verified,
                "layer":       f"AI Infrastructure - {t.attack_plane}",
                "description": t.description,
                "evidence":    t.evidence,
                "mitigation":  t.mitigation,
                "references": {"atlas": t.atlas_technique, "cve": t.cve_reference,
                               "paper": "Edge GenAI on Distributed AI Infrastructure"},
            })
        confirmed = sum(1 for t in result.threats if t.status == "Confirmed")
        surface   = sum(1 for t in result.threats if t.status == "Surface")
        potential = sum(1 for t in result.threats if t.status == "Potential")
        return {
            "scan_type":      "AI_INFRASTRUCTURE",
            "target":         result.target_url,
            "scan_time":      result.scan_time,
            "reachability":   result.reachability,
            "ai_stack":       result.ai_stack,
            "threat_count":   len(result.threats),
            "confirmed_count": confirmed,
            "surface_count":   surface,
            "potential_count": potential,
            "findings":       findings,
            "summary":        result.summary,
            "catch_all_detected": result.catch_all_detected,
            "auth_protected_endpoints": result.auth_protected_endpoints,
            "auth_protected_details": result.auth_protected_details,
            "exposed_endpoints": result.exposed_endpoints,
            "planes_covered": ["BUILD", "DEPLOYMENT", "RUNTIME", "ORCHESTRATION"],
        }
