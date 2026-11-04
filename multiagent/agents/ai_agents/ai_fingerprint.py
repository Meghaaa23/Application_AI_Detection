from __future__ import annotations

import re
import json
import html as html_lib
import hashlib
import secrets
import asyncio
import logging
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    from .ai_evidence_scoring import (
        classify_evidence,
        evidence_weight,
        extract_auth_reason,
        has_live_surface_evidence,
        has_route_confirmed_evidence,
        score_evidence,
    )
except ImportError:
    try:
        from multiagent.agents.ai_agents.ai_evidence_scoring import (
            classify_evidence,
            evidence_weight,
            extract_auth_reason,
            has_live_surface_evidence,
            has_route_confirmed_evidence,
            score_evidence,
        )
    except ImportError:
        from ai_evidence_scoring import (
            classify_evidence,
            evidence_weight,
            extract_auth_reason,
            has_live_surface_evidence,
            has_route_confirmed_evidence,
            score_evidence,
        )

try:
    from .ai_protocol_signatures import (
        BEDROCK,
        ERROR_KEY_PROTOCOL,
        detect_protocol_from_headers,
        detect_protocol_from_json,
        detect_provider,
        extract_generated_text,
        is_bedrock_endpoint,
        is_gemini_endpoint,
        looks_bedrock_error,
        resolve_protocol,
    )
except ImportError:
    try:
        from multiagent.agents.ai_agents.ai_protocol_signatures import (
            BEDROCK,
            ERROR_KEY_PROTOCOL,
            detect_protocol_from_headers,
            detect_protocol_from_json,
            detect_provider,
            extract_generated_text,
            is_bedrock_endpoint,
            is_gemini_endpoint,
            looks_bedrock_error,
            resolve_protocol,
        )
    except ImportError:
        from ai_protocol_signatures import (
            BEDROCK,
            ERROR_KEY_PROTOCOL,
            detect_protocol_from_headers,
            detect_protocol_from_json,
            detect_provider,
            extract_generated_text,
            is_bedrock_endpoint,
            is_gemini_endpoint,
            looks_bedrock_error,
            resolve_protocol,
        )

logger = logging.getLogger(__name__)

MODEL_NAME_PATTERNS = [
    r'\b(gpt-?4|gpt-?3\.5|gpt-?4o)\b',
    r'\b(claude-?\d|claude-?(?:sonnet|opus|haiku))\b',
    r'\b(llama-?\d)',
    r'\b(mistral|mixtral|gemma|gemini|qwen|phi-?\d|deepseek)\b',
]

KNOWN_CAPABILITIES = {
    "chat", "rag", "tool_calling", "memory", "code_execution",
    "email_send", "web_search", "browser", "sql", "filesystem",
    "shell", "python", "image_generation", "vision", "speech",
    "audio", "embedding", "multi_agent", "mcp", "workflow",
}


AI_RESPONSE_BODY_PATTERNS = [
    r'"choices"\s*:\s*\[',
    r'"prompt_tokens"\s*:',
    r'"completion_tokens"\s*:',
    r'"total_tokens"\s*:',
    r'"finish_reason"\s*:',
    r'"embeddings"\s*:',
    r'"object"\s*:\s*"(?:chat\.completion(?:\.chunk)?|text_completion|embedding)"',
    r'"tool_calls"\s*:',
    r'"function_call"\s*:',
    r'"input_tokens"\s*:',
    r'"output_tokens"\s*:',
    r'"generated_text"\s*:',
]



GENERIC_CHAT_PROBE = json.dumps({
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 1,
    "stream": False,
}).encode()

GENERIC_PROMPT_PROBE = json.dumps({
    "prompt": "Hi",
    "input": "Hi",
    "query": "Hi",
    "text": "Hi",
    "max_tokens": 1,
}).encode()

GEMINI_PROBE = json.dumps({
    "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
}).encode()

BEDROCK_PROBE = json.dumps({
    "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
    "inferenceConfig": {"maxTokens": 1},
}).encode()

LLM_RESPONSE_SIGNALS = [
    r"\bI\s+(?:am|can|will|don'?t|cannot|understand)\b",
    r"\bAs\s+an?\s+(?:AI|assistant|language model)\b",
    r"\bHow\s+(?:can|may)\s+I\s+(?:help|assist)\b",
]

PROBE_SHAPE_BUILDERS: List[Tuple[str, "callable"]] = [
    ("openai_chat", lambda c: {"model": "gpt-3.5-turbo",
                               "messages": [{"role": "user", "content": c}]}),
    ("messages",    lambda c: {"messages": [{"role": "user", "content": c}]}),
    ("anthropic",   lambda c: {"model": "claude-3-haiku-20240307",
                               "max_tokens": 16,
                               "messages": [{"role": "user", "content": c}]}),
    ("gemini",      lambda c: {"contents": [
                                   {"role": "user", "parts": [{"text": c}]}]}),
    ("cohere",      lambda c: {"model": "command-r", "message": c}),
    ("tgi",         lambda c: {"inputs": c, "parameters": {"max_new_tokens": 16}}),
    ("bedrock",     lambda c: {"messages": [
                                   {"role": "user", "content": [{"text": c}]}],
                               "inferenceConfig": {"maxTokens": 16}}),
    ("prompt",      lambda c: {"prompt": c, "stream": False}),
    ("input",       lambda c: {"input": c}),
    ("query",       lambda c: {"query": c}),
    ("question",    lambda c: {"question": c}),
    ("text",        lambda c: {"text": c}),
    ("message",     lambda c: {"message": c}),
]
_PROBE_SHAPE_MAP = {k: b for k, b in PROBE_SHAPE_BUILDERS}

ERROR_FINGERPRINTS: List[Tuple[re.Pattern, str, str]] = [
    # Anthropic's top-level {"type":"error","error":{"type":...}} envelope must
    # be matched before the OpenAI pattern below (which its nested "error":{ ...
    # "type": } would otherwise also satisfy).
    (re.compile(r'"type"\s*:\s*"error".*?"error"\s*:\s*\{.*?"type"\s*:', re.S),
     "error_anthropic_format", "Anthropic Messages"),
    # Google Gemini error: the standard Google error envelope ONLY when it
    # references the generative-language service (generic Google errors do not).
    (re.compile(r'"error"\s*:\s*\{.*?(?:generativelanguage|models/gemini)', re.S | re.I),
     "error_gemini_format", "Google Gemini"),
    # Hugging Face TGI validation/generation error: {"error": "...",
    # "error_type": "validation"|"generation"|...}. The error_type field is
    # distinctive to TGI (generic HF loading errors carry no error_type).
    (re.compile(r'"error_type"\s*:\s*"', re.I),
     "error_tgi_format", "Hugging Face TGI"),
    (re.compile(r'"error"\s*:\s*\{.*?"type"\s*:', re.S),
     "error_openai_format", "OpenAI-compatible"),
    (re.compile(r'"messages"\s*.*?(required|missing|must be)', re.I),
     "error_missing_messages", "chat_api"),
    (re.compile(r'"model"\s*.*?(required|missing|must be)', re.I),
     "error_missing_model", "OpenAI-compatible"),
    (re.compile(r'"detail"\s*:.*?"msg"\s*:', re.S),
     "error_generic_validation", "api_validation"),
]

AI_RESPONSE_HEADERS = [
    "x-model-version", "x-model-id", "x-inference-time", "x-ai-backend",
    "x-ml-model", "x-openai-model", "openai-model", "openai-version",
    "x-inference-id", "x-model-name", "x-prediction-id",
]

_JS_INFERENCE_FEATURES: Tuple[str, ...] = (
    "useChat(",
    "useCompletion(",
    'role:"user"',
    "role: 'user'",
    'role: "user"',
    "/chat/completions",
    "/v1/completions",
    "text-generation",
    '"tool_calls"',
    "function_call",
)

SPA_CANARY_PATH = "/__scanner_canary_nonexistent_check_xyz789_do_not_create_this_route"

_JS_VENDOR_SKIP_TOKENS = (
    "mermaid", "codemirror", "monaco", "highlight", "prism",
    "katex", "mathjax", "animations", "framer", "lottie",
    "react-vtree", "query-devtools", "devtools",
    "fontawesome", "tippy", "leaflet", "mapbox", "xlsx",
    "pdfjs", "pdf.worker", "emoji-", "driver", "sortable",
    "recharts", "chart.js", "echarts", "quill", "tinymce",
)

_JS_API_PRIORITY_TOKENS = (
    "data-provider", "data-service", "api-endpoint", "endpoints",
    "http-client", "request", "messages", "message", "chat",
    "agents", "agent", "conversation", "convo",
    "openapi", "graphql", "inference", "completion",
    "provider", "service", "queries", "mutations", "hooks",
    "api.", "api-", "framework", "main.", "p_home", "p_layout",
)

_JS_LOW_PRIORITY_TOKENS = (
    "runtime", "registersw", "service-worker", "polyfill",
    "i18n", "locale", "lang-", "icons", "svg",
)

_JS_TRACKER_HOST_TOKENS = (
    "googletagmanager", "google-analytics", "googlesyndication",
    "googleadservices", "doubleclick", "facebook.net", "hotjar",
    "segment.com", "segment.io", "sentry.io", "newrelic",
    "datadoghq", "clarity.ms", "mixpanel", "amplitude",
    "fullstory", "heap-api", "intercom.io", "zendesk",
)

_STATIC_CDN_HOST_TOKENS = (
    "alicdn.com", "cloudfront.net", "unpkg.com", "jsdelivr.net",
    "cdnjs.cloudflare.com", "bootstrapcdn.com", "gstatic.com",
    "googleapis.com", "assets.vercel", "static.cloudflareinsights",
)

AI_TITLE_KEYWORDS = [
    "ai", "llm", "gpt", "chat", "assistant", "bot", "inference", "embedding",
]

_SPEC_REF_RE = re.compile(
    r"""(?P<path>/[^\s"'<>]*(?:openapi|swagger)[^\s"'<>]*\.(?:json|yaml|yml)|"""
    r"""/[^\s"'<>]*(?:api-docs|api/docs|v3/api-docs)[^\s"'<>]*)""",
    re.I,
)
_GRAPHQL_REF_RE = re.compile(
    r"""(?P<path>/[^\s"'<>]*(?:graphql|graphiql)[^\s"'<>]*)""",
    re.I,
)

def _looks_html_body(ct: str, body: str) -> bool:
    b = (body or "").strip().lower()
    return "html" in (ct or "").lower() or b.startswith(("<!doctype", "<html"))

def _looks_static_storage_error(ct: str, body: str) -> bool:
    """True for CDN/object-store errors (Aliyun OSS, S3, etc.) — not app APIs."""
    b = (body or "").strip()
    if not b:
        return False
    bl = b.lower()
    ct_l = (ct or "").lower()
    xmlish = (
        "xml" in ct_l
        or bl.startswith("<?xml")
        or bl.startswith("<error")
        or "<error>" in bl[:200]
    )
    if xmlish and any(x in bl for x in (
        "methodnotallowed", "accessdenied", "nosuchkey", "invalidaccesskey",
        "nosuchbucket", "allaccessdisabled", "<code>",
    )):
        return True
    if any(x in bl for x in (
        "oss-accelerate", "aliyuncs.com", "s3.amazonaws.com",
        "s3-website", "blob.core.windows.net",
    )):
        return True
    return False

def _looks_api_method_rejection(
    ct: str, body: str, headers: Optional[Dict] = None,
) -> bool:
    """True when HTTP 405 still looks like an application API handler."""
    if _looks_html_body(ct, body):
        return False
    if _looks_static_storage_error(ct, body):
        return False
    headers = headers or {}
    headers_low = {str(k).lower(): v for k, v in headers.items()}
    allow = str(headers_low.get("allow") or "")
    if allow and re.search(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b", allow, re.I):
        return True
    ct_l = (ct or "").lower()
    b = (body or "").strip()
    if "json" in ct_l or (b[:1] in "{["):
        return True
    if ct_l.startswith("application/") and "xml" not in ct_l:
        return True
    if headers_low.get("www-authenticate"):
        return True
    return False

def _looks_api_auth_rejection(
    ct: str, body: str, headers: Optional[Dict] = None,
) -> bool:
    """True when 401/403 looks like an application API auth gate (not CDN/HTML)."""
    if _looks_html_body(ct, body):
        return False
    if _looks_static_storage_error(ct, body):
        return False
    headers = headers or {}
    headers_low = {str(k).lower(): v for k, v in headers.items()}
    if headers_low.get("www-authenticate"):
        return True
    ct_l = (ct or "").lower()
    b = (body or "").strip()
    jsonish = "json" in ct_l or (b[:1] in "{[")
    if not jsonish:
        return False
    if not b:
        return "json" in ct_l
    bl = b.lower()
    if any(x in bl for x in (
        "auth", "unauthorized", "forbidden", "token", "login",
        "credential", "permission", "detail", "error", "message",
        "denied", "unauthenticated",
    )):
        return True
    try:
        data = json.loads(b)
    except Exception:
        return False
    return isinstance(data, (dict, list))

def _is_meaningful_ai_meta_content(content: str) -> bool:
    """True when meta content is real AI product copy, not a URL/domain hit."""
    text = html_lib.unescape((content or "").strip())
    if len(text) < 8:
        return False
    low = text.lower()
    if re.match(r"^https?://", low):
        return False
    if any(low.endswith(ext) for ext in (
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".js", ".css",
    )):
        return False
    scrubbed = re.sub(r"https?://\S+", " ", low)
    scrubbed = re.sub(r"\b[\w-]+\.ai\b", " ", scrubbed)
    if re.search(
        r"(?<![a-z0-9])(ai|llm|gpt|chatbot|assistant|inference|embedding|"
        r"language\s+model)(?![a-z0-9])",
        scrubbed,
        re.I,
    ):
        return True
    if re.search(r"\bai[-/]", scrubbed) or re.search(r"[-/]ai\b", scrubbed):
        return True
    return False

def _has_ai_protocol_structure(data) -> bool:
    """True only when the parsed JSON has structure specific to an inference"""
    if not isinstance(data, dict):
        return False

    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        c0 = choices[0]
        msg = c0.get("message")
        if ("text" in c0 or "delta" in c0 or "finish_reason" in c0
                or (isinstance(msg, dict) and "content" in msg)):
            return True

    usage = data.get("usage")
    if isinstance(usage, dict) and any(
        k in usage for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                             "input_tokens", "output_tokens")):
        return True

    emb = data.get("embedding")
    if isinstance(emb, list) and len(emb) >= 4 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in emb[:8]):
        return True

    data_field = data.get("data")
    if isinstance(data_field, list) and data_field and isinstance(data_field[0], dict):
        d0 = data_field[0]
        if isinstance(d0.get("embedding"), list):
            return True
        if data.get("object") == "list" and ("id" in d0 or "model" in d0):
            return True

    if "model" in data and ("done" in data or "created_at" in data) and (
        "response" in data or isinstance(data.get("message"), dict)
        or "eval_count" in data):
        return True

    models = data.get("models")
    if isinstance(models, list) and models and isinstance(models[0], dict) and any(
        k in models[0] for k in ("name", "model", "modified_at", "digest")):
        return True

    # Google Gemini generateContent: a text-bearing candidates[].content.parts[]
    # response with a corroborating Gemini marker. Conservative — a bare
    # "candidates" or "parts" field never qualifies (needs text + a marker).
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        c0 = candidates[0]
        content = c0.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        text_bearing = isinstance(parts, list) and any(
            isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"]
            for p in parts
        )
        um = data.get("usageMetadata")
        corroborated = (
            "finishReason" in c0
            or (isinstance(um, dict) and any(
                k in um for k in (
                    "promptTokenCount", "candidatesTokenCount", "totalTokenCount")))
            or isinstance(data.get("modelVersion"), str)
            or (isinstance(content, dict) and content.get("role") == "model")
        )
        if text_bearing and corroborated:
            return True

    # Cohere Chat/Generate: a text-bearing response with a distinctive Cohere
    # id/structure. Conservative — a bare {"text": "..."} never qualifies.
    cohere_text = isinstance(data.get("text"), str) and bool(data.get("text"))
    if cohere_text and isinstance(data.get("generation_id"), str):
        return True
    if cohere_text and isinstance(data.get("response_id"), str) and (
        isinstance(data.get("chat_history"), list)
        or isinstance(data.get("meta"), dict)
        or isinstance(data.get("finish_reason"), str)
    ):
        return True
    gens = data.get("generations")
    if (isinstance(gens, list) and gens and isinstance(gens[0], dict)
            and isinstance(gens[0].get("text"), str) and gens[0]["text"]):
        return True

    # AWS Bedrock Converse / InvokeModel (Titan, Llama) text-bearing responses.
    output = data.get("output")
    if isinstance(output, dict):
        msg = output.get("message")
        parts = msg.get("content") if isinstance(msg, dict) else None
        text_bearing = isinstance(parts, list) and any(
            isinstance(b, dict) and isinstance(b.get("text"), str) and b["text"]
            for b in parts
        )
        usage = data.get("usage")
        camel_usage = isinstance(usage, dict) and any(
            k in usage for k in ("inputTokens", "outputTokens", "totalTokens"))
        if text_bearing and ("stopReason" in data or camel_usage):
            return True
    results = data.get("results")
    if (isinstance(results, list) and results and isinstance(results[0], dict)
            and isinstance(results[0].get("outputText"), str)
            and results[0]["outputText"]
            and ("completionReason" in results[0] or "inputTextTokenCount" in data)):
        return True
    if (isinstance(data.get("generation"), str) and data["generation"]
            and ("prompt_token_count" in data or "generation_token_count" in data)):
        return True

    return bool(
        "tool_calls" in data or "function_call" in data or "generated_text" in data)

def _looks_ai_api_json(body: str) -> bool:
    text = (body or "").strip()
    if not text or text in ("{}", "[]", "null"):
        return False
    if any(re.search(p, text, re.I) for p in AI_RESPONSE_BODY_PATTERNS):
        return True
    try:
        data = json.loads(text)
    except Exception:
        return False
    return _has_ai_protocol_structure(data)

def _is_soft_404_json(body: str) -> bool:
    text = (body or "").strip()
    if text in ("", "{}", "[]", "null"):
        return True
    try:
        data = json.loads(text)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if data.get("statusCode") in (404, 405):
        return True
    code = str(data.get("code") or data.get("error") or "").lower()
    if code in ("not_found", "enoent", "404"):
        return True
    msg = str(data.get("message") or data.get("msg") or "").lower()
    return "not found" in msg and len(data) <= 3

def _looks_non_api_document(path: str) -> bool:
    """True for static/doc file shapes — never inference APIs."""
    pl = (path or "").lower().split("?")[0].split("#")[0]
    if not pl:
        return True
    if pl.endswith((
        ".txt", ".xml", ".html", ".htm", ".md", ".pdf", ".csv",
        ".css", ".js", ".map", ".ico", ".png", ".jpg", ".jpeg", ".svg", ".webp",
    )):
        return True
    return False

_INFERENCE_SEGMENTS = frozenset({
    "chat", "completion", "completions",
    "generate", "predict", "prediction",
    "ask", "message", "messages",
    "run", "invoke", "llm", "inference",
    "embedding", "embeddings",
})

def _is_inference_path(path: str) -> bool:
    """True when a discovered path's segments look like an inference handler."""
    p = (path or "").lower().split("?")[0].split("#")[0]
    if not p or _looks_non_api_document(p):
        return False
    segments = [s for s in p.strip("/").split("/") if s]
    if any(seg in _INFERENCE_SEGMENTS for seg in segments):
        return True
    # Gemini generateContent-style verb endpoints (/models/{model}:generateContent)
    # and AWS Bedrock /model/{id}/(invoke|converse…) routes are inference
    # handlers too, recognised like every other vendor's route.
    return is_gemini_endpoint(p) or is_bedrock_endpoint(p)

def _probe_post_body(
    path: str,
    category: str,
    openapi_schema: Optional[Dict] = None,
) -> bytes:
    """Build a probe body from OpenAPI request schema when available."""
    derived = _body_from_openapi(path, openapi_schema)
    if derived is not None:
        return derived
    path_l = (path or "").lower()
    if is_gemini_endpoint(path):
        return GEMINI_PROBE
    if is_bedrock_endpoint(path):
        return BEDROCK_PROBE
    if "completions" in path_l and "chat" not in path_l:
        return GENERIC_PROMPT_PROBE
    if category in ("chat", "ml", "embedding") or _is_inference_path(path):
        return GENERIC_CHAT_PROBE
    return GENERIC_PROMPT_PROBE

def _body_from_openapi(path: str, schema: Optional[Dict]) -> Optional[bytes]:
    if not schema or not isinstance(schema, dict):
        return None
    paths = schema.get("paths") or {}
    methods = paths.get(path) or paths.get(path.rstrip("/")) or {}
    if not isinstance(methods, dict):
        return None
    for method in ("post", "put", "patch", "get"):
        op = methods.get(method)
        if not isinstance(op, dict):
            continue
        props = (
            ((op.get("requestBody") or {}).get("content") or {})
            .get("application/json", {})
            .get("schema", {})
            .get("properties")
        )
        if not isinstance(props, dict) or not props:
            continue
        body: Dict = {}
        for key, prop in props.items():
            kl = key.lower()
            if kl in ("messages",):
                body[key] = [{"role": "user", "content": "Hi"}]
            elif kl in ("prompt", "input", "query", "text", "question", "input_value"):
                body[key] = "Hi"
            elif kl in ("model",):
                body[key] = "default"
            elif kl in ("max_tokens", "max_new_tokens"):
                body[key] = 1
            elif kl in ("stream",):
                body[key] = False
            elif isinstance(prop, dict) and prop.get("type") == "object":
                body[key] = {}
            elif isinstance(prop, dict) and prop.get("type") == "array":
                body[key] = []
            else:
                body[key] = "Hi" if "str" in str(prop.get("type", "string")) else 0
        if body:
            return json.dumps(body).encode()
    return None

AI_DOM_PATTERNS = [
    r'role=["\']assistant["\']',
    r'data-role=["\'](?:bot|assistant)["\']',
    r'aria-label=["\'][^"\']*(?:ai|assistant|chatbot)[^"\']*["\']',
    r'data-message-role=["\']assistant["\']',
    r'<form[^>]+action=["\'][^"\']*(?:chat|predict|ask)[^"\']*["\']',
]

ATLAS_TECHNIQUE_MAP = {
    "chat":      "AML.T0051 - LLM Prompt Injection",
    "ml":        "AML.T0043 - Craft Adversarial Data",
    "schema":    "AML.T0063 - Discover AI Model Outputs",
    "upload":    "AML.T0018.002 - Embed Malware via Model Upload",
    "embedding": "AML.T0043 - Craft Adversarial Data",
    "graphql":   "AML.T0051 - LLM Prompt Injection",
}

@dataclass
class EvidenceItem:
    key:       str
    detail:    str
    weight:    int
    source:    str
    endpoint:  str = ""
    framework: str = ""
    evidence_class: str = ""
    protocol:  str = ""
    provider:  str = ""

    @staticmethod
    def make(key: str, detail: str, source: str,
             endpoint: str = "", framework: str = "",
             protocol: str = "", provider: str = "") -> "EvidenceItem":
        return EvidenceItem(
            key=key, detail=detail,
            weight=evidence_weight(key),
            source=source, endpoint=endpoint, framework=framework,
            evidence_class=classify_evidence(key).value,
            protocol=protocol, provider=provider,
        )

@dataclass
class AuthSignal:
    endpoint:    str
    auth_type:   str
    status_code: int
    detail:      str
    auth_status: str = ""
    reason:      str = ""

ENDPOINT_TYPE_RULES: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"/(?:chat|completions|generate|messages|ask|conversation|prediction|predict)(?:/|$)", re.I),
     "inference",  "critical"),
    (re.compile(r"/(?:embed|embeddings?)(?:/|$)", re.I),                           "embedding",  "high"),
    (re.compile(r"/(?:pull|push|delete|copy|create|import)(?:/|$)", re.I),         "management", "critical"),
    (re.compile(r"/(?:models?|tags|version|health|info|config)(?:/|$)", re.I),     "metadata",   "medium"),
    (re.compile(r"/(?:tool|function|plugin|action|execute|run)(?:/|$)", re.I),     "tool",       "high"),
    (re.compile(r"/(?:memory|store|recall|history)(?:/|$)", re.I),                 "memory",     "high"),
    (re.compile(r"/(?:search|retrieve|vector|rag)(?:/|$)", re.I),                  "retrieval",  "high"),
    (re.compile(r"/(?:agents?|orchestrat|workflow|plan)(?:/|$)", re.I),            "agentic",    "high"),
    (re.compile(r"/(?:upload|file|image|audio|vision)(?:/|$)", re.I),              "multimodal", "medium"),
    (re.compile(r"/(?:graphql|query)(?:/|$)", re.I),                               "graphql",    "high"),
    (re.compile(r"/(?:chatflows|flows|components|nodes|openapi|swagger|docs|schema)(?:/|$)", re.I),
     "schema",     "low"),
]

@dataclass
class EndpointRecord:
    url:           str
    path:          str
    endpoint_type: str
    risk:          str
    auth_required: bool = False
    auth_type:     str  = ""
    status_code:   int  = 200
    verified:      bool = True

def classify_endpoint(url: str, auth_signals: List[AuthSignal] = None) -> EndpointRecord:
    path = urllib.parse.urlparse(url).path
    ep_type = "unknown"
    risk    = "low"
    for pattern, t, r in ENDPOINT_TYPE_RULES:
        if pattern.search(path):
            ep_type = t
            risk    = r
            break

    auth_required = False
    auth_type     = ""
    for sig in (auth_signals or []):
        if sig.endpoint == url:
            auth_required = True
            auth_type     = sig.auth_type
            break

    return EndpointRecord(
        url=url, path=path,
        endpoint_type=ep_type, risk=risk,
        auth_required=auth_required, auth_type=auth_type,
    )

@dataclass
class FingerprintReport:
    is_ai_powered:    bool
    confidence:       str

    framework:            str   = ""
    framework_confidence: float = 0.0
    framework_signals:    List[str] = field(default_factory=list)
    protocol:             str   = ""
    protocol_confidence:  str   = "NONE"
    provider:             str   = ""
    provider_confidence:  str   = "NONE"
    inference_server:     str   = ""
    model_reported:       str   = ""

    infra_layer: str = ""
    app_layer:   str = ""

    capabilities:  List[str] = field(default_factory=list)

    ml_endpoints:           List[str]         = field(default_factory=list)
    classified_endpoints:   List[EndpointRecord] = field(default_factory=list)
    discovered_paths:       List[str]         = field(default_factory=list)
    auth_required_paths:    List[AuthSignal]  = field(default_factory=list)
    allowed_api_hosts:      List[str]         = field(default_factory=list)

    frameworks_detected: List[str] = field(default_factory=list)
    openapi_schema:      Dict       = field(default_factory=dict)
    graphql_types:       List[str]  = field(default_factory=list)
    technologies:        List[str]  = field(default_factory=list)
    deployment_signals:  Dict       = field(default_factory=dict)

    evidence_items:   List[EvidenceItem] = field(default_factory=list)
    atlas_techniques: List[str]          = field(default_factory=list)
    page_title:       str                = ""
    spa_catch_all:    bool               = False
    app_description:  str                = ""

    has_live_surface: bool = False
    has_route_confirmed: bool = False
    ai_brand_only:    bool = False
    detection:        str  = "not_detected"
    evidence_classes: Dict = field(default_factory=dict)
    chat_components:  List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "is_ai_powered":          self.is_ai_powered,
            "detection":              self.detection,
            "confidence":             self.confidence,
            "has_live_surface":       self.has_live_surface,
            "has_route_confirmed":    self.has_route_confirmed,
            "ai_brand_only":          self.ai_brand_only,
            "evidence_classes":       self.evidence_classes,
            "framework":              self.framework,
            "framework_confidence":   round(self.framework_confidence * 100, 1),
            "framework_signals":      self.framework_signals,
            "protocol":               self.protocol,
            "protocol_confidence":    self.protocol_confidence,
            "provider":               self.provider,
            "provider_confidence":    self.provider_confidence,
            "inference_server":       self.inference_server,
            "infra_layer":            self.infra_layer,
            "app_layer":              self.app_layer,
            "model_reported":         self.model_reported,
            "capabilities":           self.capabilities,
            "ml_endpoints":           self.ml_endpoints,
            "allowed_api_hosts":      self.allowed_api_hosts,
            "classified_endpoints": [
                {"url": e.url, "path": e.path, "type": e.endpoint_type,
                 "risk": e.risk, "confirmed_live": self.has_live_surface,
                 "auth_required": e.auth_required,
                 "auth_type": e.auth_type}
                for e in self.classified_endpoints
            ],
            "discovered_paths": self.discovered_paths,
            "auth_required_paths": [
                {"endpoint": a.endpoint, "auth_type": a.auth_type,
                 "status_code": a.status_code, "detail": a.detail,
                 "auth_status": a.auth_status, "reason": a.reason}
                for a in self.auth_required_paths
            ],
            "frameworks_detected": self.frameworks_detected,
            "openapi_schema":      self.openapi_schema,
            "graphql_types":       self.graphql_types,
            "deployment_signals":  self.deployment_signals,
            "page_title":          self.page_title,
            "spa_catch_all":       self.spa_catch_all,
            "technologies":        self.technologies,
            "api_schema":          {"tools": [], "openapi": self.openapi_schema},
            "app_description":     self.app_description,
            "response_headers":    {},
            "chat_components":     self.chat_components,
            "atlas_techniques":    self.atlas_techniques,
            "evidence_items": [
                {"key": e.key, "detail": e.detail, "weight": e.weight,
                 "evidence_class": e.evidence_class,
                 "source": e.source, "endpoint": e.endpoint, "framework": e.framework,
                 "protocol": e.protocol, "provider": e.provider}
                for e in sorted(self.evidence_items, key=lambda x: x.weight, reverse=True)
            ],
        }

    def print_report(self) -> None:
        w = 62
        print(f"\n{'='*w}")
        print(f"  AI Fingerprint Report")
        print(f"{'='*w}")
        print(f"  Detection        : {self.detection.upper()}  (confidence={self.confidence})")
        print(f"  Live surface     : {self.has_live_surface}")
        if self.ai_brand_only:
            print(f"  Brand only       : True")
        if self.framework:
            pct = f"{self.framework_confidence*100:.0f}%"
            print(f"  Framework        : {self.framework}  (confidence {pct})")
            if self.framework_signals:
                print(f"  Matched signals  : {', '.join(self.framework_signals)}")
        if self.protocol:
            print(f"  Protocol         : {self.protocol}")
        if self.infra_layer and self.infra_layer != self.framework:
            print(f"  Infra layer      : {self.infra_layer}")
        if self.app_layer:
            print(f"  App layer        : {self.app_layer}")
        if self.model_reported:
            print(f"  Model            : {self.model_reported}")
        if self.capabilities:
            print(f"  Capabilities     : {', '.join(self.capabilities)}")
        if self.classified_endpoints:
            print(f"  Endpoints ({len(self.classified_endpoints)}):")
            risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}
            risk_suffix = "" if self.has_live_surface else "-IF-LIVE"
            for ep in sorted(self.classified_endpoints,
                             key=lambda e: risk_order.get(e.risk, 9)):
                auth_tag = f" [{ep.auth_type}]" if ep.auth_required else " [auth: untested]"
                risk_label = f"{ep.risk.upper()}{risk_suffix}"
                print(f"    [{risk_label:16s}] {ep.endpoint_type:12s} {ep.path}{auth_tag}")
        if self.auth_required_paths:
            print(f"  Auth-gated       : {len(self.auth_required_paths)} endpoint(s) (not tested)")
            for a in self.auth_required_paths[:5]:
                print(f"    [{a.status_code}] {a.auth_status or 'auth'}: {a.reason or '-'}  {a.endpoint}")
        if self.deployment_signals:
            print(f"  Deployment       : {self.deployment_signals}")
        print(f"\n  Evidence (top 8):")
        for ev in sorted(self.evidence_items, key=lambda x: x.weight, reverse=True)[:8]:
            print(f"    [{ev.evidence_class:12s}] [{ev.source:10s}]  {ev.detail[:60]}")
        print(f"{'='*w}\n")

class AIFingerprintAgent:

    def __init__(
        self,
        timeout: int = 8,
        max_js_bundles: int = 20,
        max_js_bundle_bytes: int = 1_500_000,
        run_behavioral_probes: bool = True,
        probe_prompt: str = "In one sentence, describe the ocean.",
        probe_model: str = "gpt-3.5-turbo",
        nondeterminism_samples: int = 3,
        nondeterminism_low: float = 0.15,
        nondeterminism_high: float = 0.98,
        auth_headers: Optional[Dict[str, str]] = None,
    ):
        self.timeout                = timeout
        self.max_js_bundles         = max_js_bundles
        self.max_js_bundle_bytes    = max_js_bundle_bytes
        self.run_behavioral_probes  = run_behavioral_probes
        self.probe_prompt           = probe_prompt
        self.probe_model            = probe_model
        self.nondeterminism_samples = nondeterminism_samples
        self.nondeterminism_low     = nondeterminism_low
        self.nondeterminism_high    = nondeterminism_high
        self.auth_headers            = auth_headers or {}

    async def fingerprint(
        self,
        target_url: str,
        discovered_endpoints: List[Dict] = None,
        page_html: str = "",
        response_headers: Dict = None,
        verbose: bool = False,
    ) -> Dict:
        report = await self._fingerprint_internal(
            target_url, discovered_endpoints or [],
            page_html, response_headers or {}, verbose)
        if verbose:
            report.print_report()
        return report.to_dict()

    async def _fingerprint_internal(
        self,
        target_url: str,
        discovered_endpoints: List[Dict],
        page_html: str,
        response_headers: Dict,
        verbose: bool,
    ) -> FingerprintReport:
        base = target_url.rstrip("/")

        if not page_html:
            loop = asyncio.get_event_loop()
            status, page_headers, page_html = await loop.run_in_executor(
                None, lambda: self._fetch(base, max_bytes=200_000))
            if not response_headers:
                response_headers = page_headers
            if not page_html:
                logger.warning(
                    "[fingerprint] empty root page body from %s (HTTP %s) — "
                    "host unreachable or returned no content",
                    base, status)

        spa_catch_all = await self._detect_spa_catch_all(base)
        if spa_catch_all and verbose:
            print("    [spa] catch-all/SPA fallback detected — using content-based probes")

        js_urls, html_evidence, extracted_links, page_title = (
            self._analyse_html(page_html, base, response_headers)
        )
        if spa_catch_all:
            html_evidence.append(EvidenceItem.make(
                "spa_catch_all",
                "SPA/catch-all routing detected (canary path returned app shell)",
                "passive"))

        parallel = await asyncio.gather(
            self._fetch_and_analyse_js(js_urls, base, verbose),
            self._fetch_robots_sitemap(base, verbose),
            self._analyse_headers_cookies(response_headers, verbose),
            return_exceptions=True,
        )
        js_r, passive_r, header_r = parallel

        all_evidence: List[EvidenceItem] = list(html_evidence)
        ml_endpoints:    List[str]  = []
        discovered_paths: List[str] = []
        all_frameworks:  List[str]  = []
        auth_signals:    List[AuthSignal] = []
        openapi_schema:  Dict       = {}
        graphql_types:   List[str]  = []
        js_api_paths:    List[str]  = []
        raw_html_js = ""
        candidates: List[str] = list(extracted_links)

        for r in [js_r, passive_r, header_r]:
            if isinstance(r, dict):
                all_evidence.extend(r.get("evidence", []))
                all_frameworks.extend(r.get("frameworks", []))

        related_bases: List[str] = []
        if isinstance(js_r, dict):
            js_api_paths = js_r.get("api_paths", [])
            raw_html_js  = js_r.get("raw_js", "")
            related_bases.extend(js_r.get("api_bases", []) or [])
            for path in js_api_paths:
                discovered_paths.append(path)
                if self._candidate_is_inference(path):
                    candidates.append(path)

        if isinstance(passive_r, dict):
            for path in passive_r.get("paths", []) or []:
                discovered_paths.append(path)
                if self._candidate_is_inference(path):
                    candidates.append(path)

        for ep in discovered_endpoints or []:
            url = ep.get("url", "") if isinstance(ep, dict) else str(ep)
            if not url:
                continue
            if "://" in url:
                candidates.append(url)
                discovered_paths.append(url)
            else:
                candidates.append(url)
                discovered_paths.append(url)

        related_bases = self._normalize_api_bases(related_bases, page_base=base)
        if related_bases:
            hop_r = await self._discover_related_bases(
                related_bases, page_base=base, verbose=verbose)
            if isinstance(hop_r, dict):
                all_evidence.extend(hop_r.get("evidence", []))
                for path in hop_r.get("api_paths", []) or []:
                    discovered_paths.append(path)
                    if self._candidate_is_inference(path):
                        candidates.append(path)
                related_bases.extend(hop_r.get("api_bases", []) or [])
                related_bases = self._normalize_api_bases(related_bases, page_base=base)

        relative_infer = [
            p for p in discovered_paths
            if p.startswith("/") and _is_inference_path(p)
        ]
        for api_base in related_bases:
            for path in relative_infer:
                candidates.append(api_base.rstrip("/") + path)

        for url in self._related_host_api_candidates(
            discovered_paths, related_bases, page_base=base,
        ):
            candidates.append(url)

        surface_blob = "\n".join([
            page_html or "",
            raw_html_js or "",
            "\n".join(discovered_paths),
        ])
        spec_refs = self._extract_spec_refs(surface_blob)
        graphql_refs = self._extract_graphql_refs(surface_blob)
        for path in discovered_paths:
            pl = path.lower()
            if any(x in pl for x in ("openapi", "swagger", "api-docs")):
                spec_refs.append(path if path.startswith("/") else urllib.parse.urlparse(path).path)
            if "graphql" in pl or "graphiql" in pl:
                graphql_refs.append(path if path.startswith("/") else urllib.parse.urlparse(path).path)

        openapi_r = await self._probe_openapi(
            base, self._normalize_candidate_paths(spec_refs), verbose)
        graphql_r = await self._probe_graphql(
            base, self._normalize_candidate_paths(graphql_refs), verbose)

        if isinstance(openapi_r, dict):
            all_evidence.extend(openapi_r.get("evidence", []))
            if openapi_r.get("found"):
                openapi_schema = openapi_r.get("schema", {})
                for ep in openapi_r.get("endpoints", []) or []:
                    path = urllib.parse.urlparse(ep).path if ep.startswith("http") else ep
                    if path:
                        candidates.append(path)
                        discovered_paths.append(path)

        if isinstance(graphql_r, dict):
            all_evidence.extend(graphql_r.get("evidence", []))
            graphql_types = graphql_r.get("types", [])
            for ep in graphql_r.get("endpoints", []) or []:
                path = urllib.parse.urlparse(ep).path if ep.startswith("http") else ep
                if path:
                    candidates.append(path)
                    discovered_paths.append(path)

        candidates = self._normalize_candidate_paths(candidates)
        if verbose:
            print(f"    [discover] {len(candidates)} candidate path(s) from app surface")
            if related_bases:
                print(
                    "    [discover] related API host(s) from JS: "
                    + ", ".join(related_bases[:6])
                )
            if not candidates:
                uniq = list(dict.fromkeys(discovered_paths))
                api_like = [
                    p for p in uniq
                    if "/api/" in p or p.startswith("/v1/")
                    or p.startswith("/v2/") or "/ollama/" in p
                    or p.startswith("http")
                ]
                sample = (api_like or uniq)[:12]
                if sample:
                    print(
                        "    [discover] non-probe paths seen: "
                        + ", ".join(sample)
                    )

        related_api_hosts = {
            (urllib.parse.urlparse(origin).netloc or "").lower()
            for origin in related_bases
            if urllib.parse.urlparse(origin).netloc
        }
        page_host = (urllib.parse.urlparse(base).netloc or "").lower()
        if candidates:
            probe_r = await self._probe_candidates(
                base, candidates, verbose, spa_catch_all=spa_catch_all,
                openapi_schema=openapi_schema,
                related_api_hosts=related_api_hosts,
                page_host=page_host,
            )
            if isinstance(probe_r, dict):
                all_evidence.extend(probe_r.get("evidence", []))
                ml_endpoints.extend(probe_r.get("endpoints", []))
                all_frameworks.extend(probe_r.get("frameworks", []))
                auth_signals.extend(probe_r.get("auth_signals", []))

        for sig in auth_signals:
            if sig.endpoint and sig.endpoint not in ml_endpoints:
                ml_endpoints.append(sig.endpoint)

        behav_r = None
        auth_urls = {a.endpoint for a in auth_signals}
        open_targets = [e for e in ml_endpoints if e not in auth_urls]
        if self.run_behavioral_probes and open_targets:
            behav_r = await self._behavioral_probes(
                open_targets[:2], verbose, openapi_schema=openapi_schema)
            if isinstance(behav_r, dict):
                all_evidence.extend(behav_r.get("evidence", []))
            nd_r = await self._nondeterminism_probe(
                open_targets[0], verbose, openapi_schema=openapi_schema)
            if isinstance(nd_r, dict):
                all_evidence.extend(nd_r.get("evidence", []))

        ml_endpoints     = list(dict.fromkeys(ml_endpoints))
        discovered_paths = list(dict.fromkeys(discovered_paths))
        all_frameworks   = list(dict.fromkeys(all_frameworks))

        seen_auth: Set[str] = set()
        deduped_auth: List[AuthSignal] = []
        for sig in auth_signals:
            if sig.endpoint in seen_auth:
                continue
            seen_auth.add(sig.endpoint)
            deduped_auth.append(sig)
        auth_signals = deduped_auth

        framework_result = self._identify_protocol(
            page_html, raw_html_js, response_headers,
            ml_endpoints + discovered_paths, all_evidence)
        all_evidence.extend(framework_result.get("evidence", []))
        detected_framework   = framework_result.get("framework", "")
        detected_protocol    = framework_result.get("protocol", "")
        detected_infra       = framework_result.get("inference_server", "")

        evidence_keys = [ev.key for ev in all_evidence]
        has_live_surface = has_live_surface_evidence(evidence_keys)

        def _ep_path(ep: str) -> str:
            return urllib.parse.urlparse(ep).path if "://" in ep else ep

        has_route_confirmed = (
            has_route_confirmed_evidence(evidence_keys)
            or any(_is_inference_path(_ep_path(ep)) for ep in ml_endpoints)
            or any(_is_inference_path(_ep_path(sig.endpoint)) for sig in auth_signals)
        )

        if detected_framework and detected_framework not in all_frameworks:
            all_frameworks.insert(0, detected_framework)
        if detected_protocol and detected_protocol not in all_frameworks:
            all_frameworks.append(detected_protocol)

        capabilities = self._infer_capabilities(
            detected_framework, all_frameworks, openapi_schema,
            ml_endpoints, graphql_types)

        seen_ev: Set[Tuple[str, str]] = set()
        deduped_evidence: List[EvidenceItem] = []
        for ev in all_evidence:
            k = (ev.key, ev.detail)
            if k in seen_ev:
                continue
            seen_ev.add(k)
            deduped_evidence.append(ev)
        all_evidence = deduped_evidence

        evidence_keys = [ev.key for ev in all_evidence]
        detection = score_evidence(evidence_keys)
        confidence = detection.confidence
        classes = detection.classes or {}
        ai_brand_only = (
            not has_live_surface
            and bool(classes.get("brand"))
            and not classes.get("behavioral")
            and not classes.get("live_surface")
            and not classes.get("passive_structural")
            and not classes.get("passive_hint")
        )
        is_ai = has_live_surface

        # Protocol / API-format identification (orthogonal to presence). Reads
        # the protocol tags on existing evidence; never alters is_ai/confidence.
        protocol_res = resolve_protocol(
            all_evidence, has_live_surface, has_route_confirmed)

        atlas: List[str] = []
        for ep in ml_endpoints:
            path = urllib.parse.urlparse(ep).path.lower()
            if any(x in path for x in ["/chat", "/completions", "/generate", "/messages",
                                       "/ask", "/prediction", "/run"]):
                atlas.append(ATLAS_TECHNIQUE_MAP["chat"])
            elif any(x in path for x in ["/embed", "/embedding"]):
                atlas.append(ATLAS_TECHNIQUE_MAP["embedding"])
            else:
                atlas.append(ATLAS_TECHNIQUE_MAP["ml"])
        if openapi_schema:
            atlas.append(ATLAS_TECHNIQUE_MAP["schema"])
        if graphql_types:
            atlas.append(ATLAS_TECHNIQUE_MAP["graphql"])
        atlas = list(dict.fromkeys(atlas))

        model_reported = ""
        if isinstance(behav_r, dict):
            model_reported = behav_r.get("model_reported", "")

        classified = [
            classify_endpoint(ep, auth_signals)
            for ep in ml_endpoints
        ]

        deployment = {}
        for ev in all_evidence:
            if ev.source == "header" and "framework" in ev.key:
                deployment["server_header"] = ev.detail
            if "docker" in ev.detail.lower() or "container" in ev.detail.lower():
                deployment["containerized"] = True
        if spa_catch_all:
            deployment["spa_catch_all"] = True
        if auth_signals:
            deployment["authentication"] = list({a.auth_type for a in auth_signals})
        else:
            deployment["authentication"] = ["untested"]

        PROTOCOL_LABELS = {"OpenAI-compatible", "chat_api"}
        infra_layer = detected_framework if detected_framework in PROTOCOL_LABELS else detected_infra
        app_layer   = ""

        meta_desc = ""
        m_desc = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
            page_html or "", re.I)
        if not m_desc:
            m_desc = re.search(
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']description["\']',
                page_html or "", re.I)
        if m_desc:
            meta_desc = m_desc.group(1)[:240]

        allowed_api_hosts = self._collect_allowed_api_hosts(
            page_base=base,
            related_bases=related_bases,
            ml_endpoints=ml_endpoints,
            auth_signals=auth_signals,
        )

        chat_components: List[Dict] = []
        try:
            from .chatbot_ui_probe import (
                detect_chatbot_from_html,
                evidence_key_for_tier,
                widgets_to_chat_components,
            )
        except ImportError:
            try:
                from multiagent.agents.ai_agents.chatbot_ui_probe import (
                    detect_chatbot_from_html,
                    evidence_key_for_tier,
                    widgets_to_chat_components,
                )
            except ImportError:
                from chatbot_ui_probe import (
                    detect_chatbot_from_html,
                    evidence_key_for_tier,
                    widgets_to_chat_components,
                )
        try:
            widgets = detect_chatbot_from_html(page_html or "", base)
            chat_components = widgets_to_chat_components(widgets)
            for w in widgets:
                key = evidence_key_for_tier(
                    getattr(w, "tier", "") or "passive_hint",
                    getattr(w, "scope", "") or "",
                )
                all_evidence.append(EvidenceItem.make(
                    key,
                    getattr(w, "detail", None)
                    or f"Chatbot UI surface ({key})",
                    "passive",
                ))
        except Exception:
            chat_components = []

        return FingerprintReport(
            is_ai_powered         = is_ai,
            confidence            = confidence,
            framework             = detected_framework,
            framework_confidence  = framework_result.get("framework_confidence", 0.0),
            framework_signals     = framework_result.get("framework_signals", []),
            protocol              = protocol_res.protocol,
            protocol_confidence   = protocol_res.confidence,
            provider              = protocol_res.provider,
            provider_confidence   = protocol_res.provider_confidence,
            inference_server      = detected_infra,
            infra_layer           = infra_layer,
            app_layer             = app_layer,
            model_reported        = model_reported,
            capabilities          = capabilities,
            ml_endpoints          = ml_endpoints,
            classified_endpoints  = classified,
            discovered_paths      = discovered_paths,
            auth_required_paths   = auth_signals,
            allowed_api_hosts     = allowed_api_hosts,
            frameworks_detected   = all_frameworks,
            openapi_schema        = openapi_schema,
            graphql_types         = graphql_types,
            technologies          = all_frameworks,
            deployment_signals    = deployment,
            evidence_items        = all_evidence,
            atlas_techniques      = atlas,
            page_title            = page_title,
            spa_catch_all         = spa_catch_all,
            app_description       = meta_desc,
            has_live_surface      = has_live_surface,
            has_route_confirmed   = has_route_confirmed,
            ai_brand_only         = ai_brand_only,
            detection             = detection.detection.value,
            evidence_classes      = detection.classes,
            chat_components       = chat_components,
        )

    def _analyse_html(
        self, html: str, base: str, headers: Dict
    ) -> Tuple[List[str], List[EvidenceItem], List[str], str]:
        """Return (js_urls, evidence, links, page_title)."""
        evidence: List[EvidenceItem] = []
        js_urls:  List[str]          = []
        links:    List[str]          = []
        page_title = ""

        if not html:
            return js_urls, evidence, links, page_title

        m_title = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
        if m_title:
            page_title = re.sub(r"\s+", " ", html_lib.unescape(m_title.group(1))).strip()
            title_low = page_title.lower()
            if any(re.search(rf"\b{re.escape(kw)}\b", title_low)
                   for kw in AI_TITLE_KEYWORDS):
                evidence.append(EvidenceItem.make(
                    "page_title_ai", f"AI page title: {page_title[:80]}", "html"))
        else:
            title_low = ""

        for m in re.finditer(r'<script[^>]+src=["\']([^"\']+)["\']', html, re.I):
            src = self._abs_url(base, m.group(1))
            if self._looks_js_asset_url(src, base, allow_cross_origin=True):
                js_urls.append(src)

        for m in re.finditer(
            r'<link[^>]+rel=["\'](?:modulepreload|preload)["\'][^>]*>',
            html, re.I,
        ):
            tag = m.group(0)
            href_m = re.search(r'href=["\']([^"\']+)["\']', tag, re.I)
            if not href_m:
                continue
            href = self._abs_url(base, href_m.group(1))
            if self._looks_js_asset_url(href, base, allow_cross_origin=True):
                js_urls.append(href)

        for m in re.finditer(
            r'<meta[^>]+content=["\']([^"\']*(?:ai|llm|gpt|chatbot|assistant|'
            r'inference|embedding|language model)[^"\']*)["\']',
            html, re.I,
        ):
            content = m.group(1)
            if not _is_meaningful_ai_meta_content(content):
                continue
            evidence.append(EvidenceItem.make(
                "meta_ai_tag", f"AI meta tag: {content[:60]}", "html"))

        for pattern in AI_DOM_PATTERNS:
            m = re.search(pattern, html, re.I)
            if m:
                ev_key = "html_ai_form" if "form" in pattern else "dom_assistant_role"
                evidence.append(EvidenceItem.make(ev_key, m.group(0)[:60], "html"))

        for m in re.finditer(r'href=["\']([^"\'#?]+)["\']', html, re.I):
            href = m.group(1)
            if href.startswith("/") and any(
                x in href.lower() for x in [
                    "/chat", "/api", "/playground", "/workspace",
                    "/dashboard", "/app", "/assistant", "/flow",
                    "/openapi", "/swagger", "/graphql", "/v1/", "/v2/",
                ]
            ):
                links.append(href)

        return (
            list(dict.fromkeys(js_urls)),
            evidence,
            list(dict.fromkeys(links)),
            page_title,
        )

    async def _detect_spa_catch_all(self, base: str) -> bool:
        loop = asyncio.get_event_loop()
        try:
            status, headers, body = await loop.run_in_executor(
                None, lambda: self._fetch(base + SPA_CANARY_PATH))
            if status == 200 and _looks_html_body(
                headers.get("Content-Type", "") or headers.get("content-type", ""),
                body,
            ):
                return True
            if status == 200 and body and len(body) > 200:
                root_status, _, root_body = await loop.run_in_executor(
                    None, lambda: self._fetch(base))
                if root_status == 200 and root_body and body[:500] == root_body[:500]:
                    return True
        except Exception:
            pass
        return False

    async def _fetch_and_analyse_js(
        self, js_urls: List[str], base: str, verbose: bool
    ) -> Dict:
        evidence:   List[EvidenceItem] = []
        api_paths:  List[str]          = []
        api_bases:  List[str]          = []
        frameworks: List[str]          = []
        raw_js_combined = ""

        loop = asyncio.get_event_loop()
        queue: List[str] = list(dict.fromkeys(js_urls))
        seen_urls: Set[str] = set()
        skipped_vendor = 0
        processed = 0
        budget = self.max_js_bundles
        extended_budget = max(self.max_js_bundles * 2, 24)

        async def fetch_js(url: str) -> Tuple[str, str]:
            try:
                _, _, body = await loop.run_in_executor(
                    None, lambda: self._fetch(url, max_bytes=self.max_js_bundle_bytes))
                return url, body
            except Exception:
                return url, ""

        def _has_inference() -> bool:
            return any(self._candidate_is_inference(p) for p in api_paths)

        def _refill_batch() -> List[str]:
            nonlocal skipped_vendor, queue
            ranked: List[Tuple[int, str]] = []
            kept: List[str] = []
            for u in queue:
                if u in seen_urls:
                    continue
                pri = self._js_chunk_priority(u)
                if pri < 0:
                    seen_urls.add(u)
                    skipped_vendor += 1
                    continue
                ranked.append((pri, u))
            ranked.sort(key=lambda t: t[0], reverse=True)
            queue = [u for _, u in ranked]
            batch: List[str] = []
            while queue and len(batch) < 4 and processed + len(batch) < budget:
                u = queue.pop(0)
                if u in seen_urls:
                    continue
                seen_urls.add(u)
                batch.append(u)
            return batch

        while True:
            if processed >= budget:
                if _has_inference() or budget >= extended_budget:
                    break
                budget = min(extended_budget, budget + 8)
            batch = _refill_batch()
            if not batch:
                break

            bundles = await asyncio.gather(
                *[fetch_js(u) for u in batch], return_exceptions=True)

            for item in bundles:
                if processed >= budget:
                    break
                if isinstance(item, Exception) or not isinstance(item, tuple):
                    continue
                url, js = item
                if not js:
                    continue
                processed += 1
                raw_js_combined += js[:50_000]

                if verbose:
                    trunc = (
                        len(js) >= self.max_js_bundle_bytes
                        and self.max_js_bundle_bytes > 0
                    )
                    if len(url) > 72:
                        label = url[:32] + "…" + url[-32:]
                    else:
                        label = url
                    flag = " truncated" if trunc else ""
                    print(f"    [js] {label} ({len(js)} bytes{flag})")

                js_low = js.lower()
                feature_hits = [p for p in _JS_INFERENCE_FEATURES if p.lower() in js_low]
                if feature_hits:
                    evidence.append(EvidenceItem.make(
                        "js_ai_sdk",
                        f"JS inference client feature(s): {feature_hits[0]}",
                        "js_bundle",
                        endpoint=url,
                    ))

                extracted = self._extract_api_paths_from_js(js)
                api_paths.extend(extracted)
                api_bases.extend(self._extract_api_bases_from_js(js))
                ai_shaped = [p for p in extracted if self._candidate_is_inference(p)]
                if ai_shaped:
                    evidence.append(EvidenceItem.make(
                        "js_api_path_extracted",
                        f"{len(ai_shaped)} inference-shaped path(s) in JS bundle",
                        "js_bundle", endpoint=url))

                for m in re.finditer(
                    r'new\s+EventSource\s*\(\s*(?:`([^`]+)`|["\']([^"\']+)["\'])',
                    js, re.I,
                ):
                    raw_es = m.group(1) if m.group(1) is not None else m.group(2)
                    for path in self._paths_from_template_expression(
                        self._unescape_js_strings(raw_es or "")
                    ):
                        if self._looks_api_candidate(path):
                            norm = self._to_path(path)
                            evidence.append(EvidenceItem.make(
                                "sse_in_js", f"EventSource({norm[:50]})",
                                "js_bundle", endpoint=url))
                            api_paths.append(norm)

                for m in re.finditer(
                    r'new\s+WebSocket\s*\(\s*(?:`([^`]+)`|["\']([^"\']+)["\'])',
                    js, re.I,
                ):
                    raw_ws = m.group(1) if m.group(1) is not None else m.group(2)
                    if any(x in (raw_ws or "") for x in (
                        "/chat", "/ws", "/stream", "/agent",
                    )):
                        evidence.append(EvidenceItem.make(
                            "websocket_ai_path",
                            f"WebSocket({(raw_ws or '')[:50]})",
                            "js_bundle", endpoint=url))

                for chunk in self._extract_js_chunk_urls(js, base, source_url=url):
                    if chunk not in seen_urls:
                        queue.append(chunk)

            if _has_inference() and processed >= min(6, self.max_js_bundles):
                break

        for p in api_paths:
            if p.startswith("http://") or p.startswith("https://"):
                if self._looks_absolute_api_url(p):
                    parsed = urllib.parse.urlparse(p)
                    api_bases.append(f"{parsed.scheme}://{parsed.netloc}")

        if verbose and processed:
            api_like = [
                p for p in dict.fromkeys(api_paths)
                if "/api/" in p or "/v1/" in p or p.startswith("http")
            ]
            infer = [p for p in api_like if self._candidate_is_inference(p)]
            print(
                f"    [js] analysed {processed} bundle(s); "
                f"{len(list(dict.fromkeys(api_paths)))} path(s) extracted "
                f"({len(api_like)} api-like, {len(infer)} inference); "
                f"skipped {skipped_vendor} vendor chunk(s)"
            )

        return {
            "evidence":  evidence,
            "api_paths": list(dict.fromkeys(api_paths)),
            "api_bases": list(dict.fromkeys(api_bases)),
            "frameworks": list(dict.fromkeys(frameworks)),
            "raw_js":    raw_js_combined,
        }

    @staticmethod
    def _js_chunk_priority(url: str) -> int:
        """Higher = fetch sooner. Negative = skip (UI vendor chrome)."""
        name = urllib.parse.urlparse(url or "").path.lower().rsplit("/", 1)[-1]
        if not name:
            return 0
        if any(tok in name for tok in _JS_VENDOR_SKIP_TOKENS):
            return -100
        score = 0
        if name.startswith("index.") or ".index." in name:
            score += 80
        for tok in _JS_API_PRIORITY_TOKENS:
            if tok in name:
                score += 50
                break
        if "vendor" in name:
            score -= 30
        for tok in _JS_LOW_PRIORITY_TOKENS:
            if tok in name:
                score -= 15
                break
        return score

    @staticmethod
    def _abs_url(base: str, src: str) -> str:
        s = (src or "").strip()
        if not s:
            return ""
        return urllib.parse.urljoin(base.rstrip("/") + "/", s)

    @staticmethod
    def _is_tracker_js_url(url: str) -> bool:
        low = (url or "").lower()
        return any(tok in low for tok in _JS_TRACKER_HOST_TOKENS)

    @staticmethod
    def _is_static_cdn_host(url_or_host: str) -> bool:
        low = (url_or_host or "").lower()
        return any(tok in low for tok in _STATIC_CDN_HOST_TOKENS)

    @classmethod
    def _looks_js_asset_url(
        cls, url: str, base: str, *, allow_cross_origin: bool = False,
        source_url: str = "",
    ) -> bool:
        if not url or not url.lower().split("?", 1)[0].endswith(".js"):
            return False
        if cls._is_tracker_js_url(url):
            return False
        u = urllib.parse.urlparse(url)
        b = urllib.parse.urlparse(base)
        s = urllib.parse.urlparse(source_url) if source_url else None
        cross = bool(u.netloc and b.netloc and u.netloc != b.netloc)
        if cross:
            if not allow_cross_origin:
                return False
            if s and s.netloc and u.netloc == s.netloc:
                pass
            elif not cls._looks_app_bundle_path(u.path):
                return False
        path = u.path.lower()
        if any(x in path for x in (
            "/assets/", "/static/", "/dist/", "/js/", "/build/",
            "/_next/", ".bundle.", "chunk", "main.", "app.", "index.",
            "/gradio_api/", "vendor", "data-provider", "framework",
            "qwenweb", "qwen-ai",
        )):
            return True
        return path.endswith(".js")

    @staticmethod
    def _looks_app_bundle_path(path: str) -> bool:
        pl = (path or "").lower()
        return any(x in pl for x in (
            "/assets/", "/static/", "/dist/", "/js/", "/build/",
            "/_next/", "chunk", "main.", "app.", "index.",
            "framework", "vendor", "qwenweb", "qwen-ai",
        ))

    def _extract_js_chunk_urls(
        self, js: str, base: str, source_url: str = "",
    ) -> List[str]:
        """Discover additional JS chunks referenced by a bundle."""
        if not js:
            return []
        text = self._unescape_js_strings(js)
        resolve_base = source_url or base
        found: List[str] = []
        patterns = (
            r'import\s*\(\s*["\']([^"\']+\.js(?:\?[^"\']*)?)["\']',
            r'["\']([^"\']*assets/[^"\']+\.js(?:\?[^"\']*)?)["\']',
            r'["\'](\.?/?assets/[^"\']+\.js(?:\?[^"\']*)?)["\']',
            r'["\'](\./(?:p_)?[a-zA-Z0-9_-]+\.js)["\']',
        )
        for pat in patterns:
            for m in re.finditer(pat, text):
                url = self._abs_url(resolve_base, m.group(1))
                if self._looks_js_asset_url(
                    url, base, allow_cross_origin=True, source_url=source_url or resolve_base,
                ):
                    found.append(url)
        return list(dict.fromkeys(found))

    @staticmethod
    def _unescape_js_strings(text: str) -> str:
        """Decode common JS string escapes so path scanners see real slashes."""
        if not text:
            return ""
        out = text.replace("\\/", "/")
        out = re.sub(
            r"\\u([0-9a-fA-F]{4})",
            lambda m: chr(int(m.group(1), 16)),
            out,
        )
        out = re.sub(
            r"\\x([0-9a-fA-F]{2})",
            lambda m: chr(int(m.group(1), 16)),
            out,
        )
        return out

    def _extract_api_paths_from_js(self, js: str) -> List[str]:
        """Discover app API routes from JS without a hardcoded endpoint catalog."""
        found: Set[str] = set()
        js = self._unescape_js_strings(js)

        def _add(raw: str) -> None:
            if not raw:
                return
            raw = raw.split("?")[0].split("#")[0].strip()
            if raw.startswith("http://") or raw.startswith("https://"):
                if self._looks_api_candidate(raw):
                    parsed = urllib.parse.urlparse(raw)
                    if not parsed.path or parsed.path == "/":
                        return
                    if self._is_static_cdn_host(parsed.netloc):
                        return
                    found.add(
                        f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
                        or raw
                    )
                return
            path = self._to_path(raw)
            if path and self._looks_api_candidate(path):
                found.add(path.rstrip("/") or path)

        for m in re.finditer(
            r'(?:fetch|axios(?:\.get|\.post|\.put|\.delete|\.patch)?|'
            r'request(?:\.get|\.post|\.put|\.delete|\.patch)?|'
            r'\$\.ajax|\$\.get|\$\.post)\s*\(\s*["\']([^"\']+)["\']',
            js, re.I,
        ):
            _add(m.group(1))

        for m in re.finditer(
            r'(?:fetch|axios(?:\.get|\.post|\.put|\.delete|\.patch)?|'
            r'request(?:\.get|\.post|\.put|\.delete|\.patch)?|'
            r'\.\s*(?:get|post|put|delete|patch))\s*\(\s*`([^`]+)`',
            js, re.I,
        ):
            for path in self._paths_from_template_expression(m.group(1)):
                _add(path)

        for m in re.finditer(
            r'\.open\s*\(\s*["\'][A-Z]+["\'],\s*["\']([^"\']+)["\']', js, re.I):
            _add(m.group(1))

        for m in re.finditer(r'["\'](/[a-zA-Z0-9_\-./]{2,120})["\']', js):
            _add(m.group(1))

        for m in re.finditer(
            r'["\'](https?://[^"\']{8,180})["\']', js, re.I,
        ):
            _add(m.group(1))

        for m in re.finditer(r'\$\{[^}]+\}(/[a-zA-Z0-9_\-./]{2,120})', js):
            frag = m.group(1)
            if frag.endswith("/"):
                frag = frag.rstrip("/")
            _add(frag)

        for m in re.finditer(r'`([^`]{2,180})`', js):
            for path in self._paths_from_template_expression(m.group(1)):
                _add(path)

        for m in re.finditer(
            r'(?<![A-Za-z0-9_])(/(?:api|v1|v2|v3|ollama|graphql|openai)'
            r'(?:/[a-zA-Z0-9_\-.]+){1,6})',
            js,
        ):
            _add(m.group(1))

        for m in re.finditer(
            r'["\'](/(?:api|v1|v2|v3)/?)["\']\s*\+\s*["\'](/?[a-zA-Z0-9_\-]+)["\']',
            js,
        ):
            left, right = m.group(1), m.group(2)
            if not right.startswith("/") and not left.endswith("/"):
                joined = left + "/" + right
            else:
                joined = left.rstrip("/") + "/" + right.lstrip("/")
            _add(joined)

        return list(found)

    def _extract_api_bases_from_js(self, js: str) -> List[str]:
        """Origins referenced in app JS that may host APIs (not static CDNs)."""
        if not js:
            return []
        text = self._unescape_js_strings(js)
        found: List[str] = []
        for m in re.finditer(r'https?://[a-zA-Z0-9._:-]+(?::\d+)?', text):
            origin = m.group(0).rstrip("/")
            host = urllib.parse.urlparse(origin).netloc.lower()
            if not host or self._is_tracker_js_url(origin):
                continue
            if self._is_static_cdn_host(host):
                continue
            if any(x in host for x in (
                "docs.", "github.io", "alibabacloud.com", "cloud.google",
                "microsoft.com", "twitter.com", "linkedin.com",
            )):
                continue
            if any(x in host for x in (
                "api.", "api-", "-api", "chat.", "chat-",
                "llm", "openai", "inference", "model", "qwen",
                "dashscope", "groq", "anthropic", "together",
                "ollama", "openrouter", "deepseek",
            )):
                found.append(origin)
        return list(dict.fromkeys(found))

    @staticmethod
    def _candidate_is_inference(candidate: str) -> bool:
        if not candidate:
            return False
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return _is_inference_path(urllib.parse.urlparse(candidate).path)
        return _is_inference_path(candidate)

    _RELATED_API_PATH_NOISE = (
        "/analytics", "/oauth", "/mediator", "/pixel", "/collect",
        "/telemetry", "/tracking", "/beacon", "/gtm", "/segment",
    )

    def _related_host_api_candidates(
        self,
        discovered_paths: List[str],
        related_bases: List[str],
        page_base: str,
        *,
        limit: int = 15,
    ) -> List[str]:
        """Absolute API URLs from JS on allowlisted related hosts (not page origin)."""
        page_host = (urllib.parse.urlparse(page_base).netloc or "").lower()
        allowed: Set[str] = set()
        for origin in related_bases or []:
            host = (urllib.parse.urlparse(origin).netloc or "").lower()
            if host and host != page_host and not self._is_static_cdn_host(host):
                allowed.add(host)
        if not allowed:
            return []

        scored: List[Tuple[int, str]] = []
        for raw in discovered_paths or []:
            if not (raw.startswith("http://") or raw.startswith("https://")):
                continue
            if not self._looks_absolute_api_url(raw):
                continue
            parsed = urllib.parse.urlparse(raw.split("#")[0])
            host = (parsed.netloc or "").lower()
            if host not in allowed:
                continue
            path = (parsed.path or "").split("?")[0]
            path_l = path.lower()
            if any(n in path_l for n in self._RELATED_API_PATH_NOISE):
                continue
            if not path or path == "/":
                continue
            score = 0
            if "/api/" in path_l or path_l.startswith("/api"):
                score += 3
            if re.search(r"/v\d+/", path_l):
                score += 2
            if _is_inference_path(path):
                score += 5
            norm = f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")
            scored.append((score, norm))

        scored.sort(key=lambda item: (-item[0], item[1]))
        out: List[str] = []
        seen: Set[str] = set()
        for _score, url in scored:
            if url in seen:
                continue
            seen.add(url)
            out.append(url)
            if len(out) >= limit:
                break
        return out

    def _normalize_api_bases(self, bases: List[str], page_base: str) -> List[str]:
        page = urllib.parse.urlparse(page_base)
        page_origin = f"{page.scheme}://{page.netloc}".rstrip("/")
        out: List[str] = []
        seen = set()
        for raw in bases or []:
            if not raw:
                continue
            p = urllib.parse.urlparse(raw if "://" in raw else "https://" + raw)
            if not p.scheme or not p.netloc:
                continue
            origin = f"{p.scheme}://{p.netloc}".rstrip("/")
            if origin in seen or origin == page_origin:
                continue
            if self._is_static_cdn_host(p.netloc) or self._is_tracker_js_url(origin):
                continue
            seen.add(origin)
            out.append(origin)
        return out[:6]

    def _collect_allowed_api_hosts(
        self,
        page_base: str,
        related_bases: List[str],
        ml_endpoints: List[str],
        auth_signals: List[AuthSignal],
    ) -> List[str]:
        """Netlocs the scanner may probe beyond the page origin (JS-derived)."""
        hosts: List[str] = []
        page_host = urllib.parse.urlparse(page_base).netloc
        if page_host:
            hosts.append(page_host)
        for origin in related_bases or []:
            h = urllib.parse.urlparse(origin).netloc
            if h and not self._is_static_cdn_host(h):
                hosts.append(h)
        for ep in list(ml_endpoints or []) + [
            s.endpoint for s in (auth_signals or []) if getattr(s, "endpoint", None)
        ]:
            if isinstance(ep, str) and ep.startswith("http"):
                h = urllib.parse.urlparse(ep).netloc
                if h and not self._is_static_cdn_host(h):
                    hosts.append(h)
        return list(dict.fromkeys(hosts))

    async def _discover_related_bases(
        self, bases: List[str], page_base: str, verbose: bool,
    ) -> Dict:
        """One-hop: fetch JS-discovered API hosts' HTML for more app routes."""
        evidence: List[EvidenceItem] = []
        api_paths: List[str] = []
        api_bases: List[str] = []
        loop = asyncio.get_event_loop()

        for origin in bases[:3]:
            try:
                status, headers, html = await loop.run_in_executor(
                    None, lambda u=origin: self._fetch(u, max_bytes=200_000))
            except Exception:
                continue
            if status == 0 or not html:
                continue
            evidence.append(EvidenceItem.make(
                "related_api_host",
                f"JS-referenced API host {origin}",
                "js_bundle", endpoint=origin))
            if verbose:
                print(f"    [discover] one-hop related host {origin}")
            js_urls, html_ev, links, _title = self._analyse_html(
                html, origin, headers or {})
            evidence.extend(html_ev)
            for link in links:
                api_paths.append(link)
            old_budget = self.max_js_bundles
            self.max_js_bundles = min(8, old_budget)
            try:
                hop_js = await self._fetch_and_analyse_js(js_urls, origin, verbose)
            finally:
                self.max_js_bundles = old_budget
            if isinstance(hop_js, dict):
                api_paths.extend(hop_js.get("api_paths", []) or [])
                api_bases.extend(hop_js.get("api_bases", []) or [])
                evidence.extend(hop_js.get("evidence", []) or [])

        return {
            "evidence": evidence,
            "api_paths": list(dict.fromkeys(api_paths)),
            "api_bases": list(dict.fromkeys(api_bases)),
        }

    @staticmethod
    def _paths_from_template_expression(raw: str) -> List[str]:
        """Pull routable path(s) or absolute API URLs out of a template body."""
        if not raw:
            return []
        out: List[str] = []
        for m in re.finditer(r'\$\{[^}]+\}(/[a-zA-Z0-9_\-./]+)', raw):
            frag = m.group(1).rstrip("/")
            if frag:
                out.append(frag)
        head = raw.split("${", 1)[0].rstrip("/")
        if head.startswith("http://") or head.startswith("https://"):
            parsed = urllib.parse.urlparse(head)
            if parsed.path and parsed.path != "/":
                out.append(f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/"))
            head = parsed.path.rstrip("/")
        if head.startswith("/") and len(head) >= 2:
            out.append(head)
        stripped = re.sub(r"\$\{[^}]*\}", "", raw).split("?")[0].strip()
        if stripped.startswith("http://") or stripped.startswith("https://"):
            parsed = urllib.parse.urlparse(stripped)
            if parsed.path and parsed.path != "/":
                out.append(
                    f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
                )
            stripped = parsed.path
        stripped = stripped.rstrip("/")
        if stripped.startswith("/") and len(stripped) >= 2:
            out.append(stripped)
        return out

    @staticmethod
    def _to_path(path: str) -> str:
        p = path.strip()
        if p.startswith("http://") or p.startswith("https://"):
            p = urllib.parse.urlparse(p).path
        if not p.startswith("/"):
            p = "/" + p.lstrip("/")
        return p.split("?")[0].split("#")[0] or "/"

    @classmethod
    def _looks_auth_account_path(cls, path: str) -> bool:
        """Auth/session/account routes are not AI live surfaces."""
        p = (path or "").lower()
        segments = [s for s in p.strip("/").split("/") if s]
        auth_segs = {
            "sso", "logout", "login", "logon", "signin", "signout", "signup",
            "oauth", "oauth2", "authorize", "token", "session", "sessions",
            "passwd", "password", "reset-password", "forgot-password",
            "csrf", "saml", "oidc", "accounts", "account",
        }
        return any(s in auth_segs for s in segments)

    @classmethod
    def _looks_absolute_api_url(cls, url: str) -> bool:
        """True for absolute URLs that look like API endpoints (not docs/SVG/CDN)."""
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            return False
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        if not host or not path or path == "/":
            return False
        if cls._is_static_cdn_host(host) or cls._is_tracker_js_url(url):
            return False
        if any(x in host for x in (
            "w3.org", "reactjs.org", "github.com", "githubusercontent",
            "wikipedia.org", "mozilla.org", "schema.org", "googleapis.com",
            "gstatic.com", "twitter.com", "x.com", "linkedin.com",
            "discord.com", "facebook.com", "youtube.com",
        )):
            return False
        if any(path.endswith(ext) for ext in (
            ".js", ".css", ".map", ".png", ".jpg", ".svg", ".html", ".md",
            ".tgz", ".dtd", ".woff", ".woff2",
        )):
            return False
        if cls._looks_auth_account_path(path):
            return False
        if re.search(
            r"/(?:api|ollama|graphql|openai|chat|completions?|"
            r"generate|messages?|embeddings?|inference|ask)(?:/|$)",
            path,
        ):
            return True
        return bool(re.search(r"/api/.*/v\d+(?:/|$)", path))

    @classmethod
    def _looks_api_candidate(cls, path: str) -> bool:
        """Heuristic: keep likely API/route strings; drop static assets."""
        p = (path or "").strip()
        if not p or len(p) > 180:
            return False
        pl = p.lower()
        if any(pl.endswith(ext) for ext in (
            ".js", ".css", ".map", ".png", ".jpg", ".jpeg", ".gif", ".svg",
            ".ico", ".woff", ".woff2", ".ttf", ".mp4", ".webp",
        )):
            return False
        if any(x in pl for x in (
            "/assets/", "/static/", "/_next/", "/dist/", "/node_modules/",
            "webpack", "chunk",
        )):
            return False
        if pl.startswith("http://") or pl.startswith("https://"):
            return cls._looks_absolute_api_url(p)
        if pl.startswith("/"):
            return True
        return False

    @staticmethod
    def _guess_probe_category(path: str) -> str:
        p = (path or "").lower()
        if "embed" in p:
            return "embedding"
        if "graphql" in p or p.endswith("/query"):
            return "graphql"
        if any(x in p for x in (
            "chat", "completion", "generate", "ask", "message", "predict",
            "prediction", "run", "invoke", "queue", "agent", "llm",
        )):
            return "chat"
        if any(x in p for x in (
            "model", "tag", "health", "openapi", "swagger", "docs", "config",
            "info", "flow", "node", "workspace", "endpoint", "convo", "schema",
        )):
            return "schema"
        return "schema"

    @staticmethod
    def _normalize_candidate_paths(paths: List[str]) -> List[str]:
        """Normalize candidates; preserve absolute URLs discovered in app JS."""
        out: List[str] = []
        seen = set()
        for raw in paths:
            if not raw:
                continue
            if raw.startswith("http://") or raw.startswith("https://"):
                parsed = urllib.parse.urlparse(raw.split("#")[0])
                path = (parsed.path or "").split("?")[0]
                if not path or path == "/":
                    continue
                if any(path.lower().endswith(ext) for ext in (
                    ".js", ".css", ".map", ".png", ".jpg", ".svg", ".ico",
                )):
                    continue
                norm = f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")
                if norm in seen:
                    continue
                seen.add(norm)
                out.append(norm)
                continue
            path = raw
            path = path if path.startswith("/") else "/" + path.lstrip("/")
            path = path.split("?")[0].split("#")[0]
            if not path or path == "/" or path in seen:
                continue
            if any(path.lower().endswith(ext) for ext in (
                ".js", ".css", ".map", ".png", ".jpg", ".svg", ".ico",
            )):
                continue
            seen.add(path)
            out.append(path)
        def _rank(p: str) -> Tuple[int, str]:
            pl = p.lower()
            if any(x in pl for x in ("chat", "completion", "generate", "predict", "ask", "run")):
                return (0, p)
            if any(x in pl for x in ("model", "flow", "agent", "message")):
                return (1, p)
            return (2, p)
        return sorted(out, key=_rank)[:80]

    @staticmethod
    def _extract_spec_refs(text: str) -> List[str]:
        found: List[str] = []
        for m in _SPEC_REF_RE.finditer(text or ""):
            found.append(m.group("path").split("?")[0])
        for m in re.finditer(
            r"""["']([^"']*(?:openapi|swagger)[^"']*\.(?:json|yaml|yml))["']""",
            text or "", re.I,
        ):
            p = m.group(1)
            if not p.startswith("/"):
                p = "/" + p.lstrip("./")
            found.append(p.split("?")[0])
        return found

    @staticmethod
    def _extract_graphql_refs(text: str) -> List[str]:
        found: List[str] = []
        for m in _GRAPHQL_REF_RE.finditer(text or ""):
            found.append(m.group("path").split("?")[0])
        return found

    async def _probe_openapi(
        self, base: str, spec_paths: List[str], verbose: bool
    ) -> Dict:
        """Fetch OpenAPI only at paths this application referenced."""
        evidence:     List[EvidenceItem] = []
        endpoints:    List[str]          = []
        capabilities: List[str]          = []
        schema:       Dict               = {}
        loop = asyncio.get_event_loop()

        if not spec_paths:
            return {
                "found": False, "evidence": evidence,
                "endpoints": endpoints, "capabilities": capabilities, "schema": schema,
            }

        for path in spec_paths:
            url = base + path
            try:
                status, _, body = await loop.run_in_executor(
                    None, lambda u=url: self._fetch(u, max_bytes=500_000))
                if status != 200 or not body:
                    continue
                spec = None
                try:
                    spec = json.loads(body)
                except json.JSONDecodeError:
                    if body.strip().startswith("openapi:"):
                        spec = self._parse_minimal_yaml_openapi(body)
                if not isinstance(spec, dict):
                    continue
                if not (spec.get("openapi") or spec.get("swagger") or spec.get("paths")):
                    continue
                schema = spec
                spec_path_map = spec.get("paths", {})
                evidence.append(EvidenceItem.make(
                    "openapi_schema",
                    f"OpenAPI spec at {path} ({len(spec_path_map)} paths)",
                    "openapi", endpoint=url))
                for ep_path, methods in spec_path_map.items():
                    endpoints.append(base + ep_path)
                    if any(x in ep_path.lower() for x in [
                        "/chat", "/completions", "/generate", "/embed",
                        "/predict", "/inference", "/model", "/messages",
                    ]):
                        evidence.append(EvidenceItem.make(
                            "openapi_ai_endpoint",
                            f"AI path in spec: {ep_path}",
                            "openapi", endpoint=base + ep_path))
                    if isinstance(methods, dict):
                        for method_def in methods.values():
                            if not isinstance(method_def, dict):
                                continue
                            props = ((method_def.get("requestBody", {}) or {})
                                     .get("content", {})
                                     .get("application/json", {})
                                     .get("schema", {})
                                     .get("properties", {}))
                            if "tools" in props or "functions" in props:
                                capabilities.append("tool_calling")
                            if "messages" in props:
                                capabilities.append("chat")
                            if "input" in props and "model" in props:
                                capabilities.append("embedding")
                if verbose:
                    print(f"    [openapi] {path}: {len(spec_path_map)} paths")
                break
            except Exception as exc:
                logger.debug(f"[OpenAPI] {url}: {exc}")

        return {
            "found": bool(schema), "evidence": evidence,
            "endpoints": list(dict.fromkeys(endpoints)),
            "capabilities": list(dict.fromkeys(capabilities)),
            "schema": schema,
        }

    @staticmethod
    def _parse_minimal_yaml_openapi(yaml_text: str) -> Optional[Dict]:
        result: Dict = {}
        paths: Dict  = {}
        for line in yaml_text.splitlines()[:30]:
            m = re.match(r'^(\w+)\s*:\s*(.+)$', line)
            if m:
                result[m.group(1)] = m.group(2).strip().strip('"\'')
        for line in yaml_text.splitlines():
            m = re.match(r'^  (/[^\s:]+)\s*:', line)
            if m:
                paths[m.group(1)] = {}
        if paths:
            result["paths"] = paths
        return result if result else None

    async def _probe_graphql(
        self, base: str, graphql_paths: List[str], verbose: bool
    ) -> Dict:
        """Probe GraphQL only at paths this application referenced."""
        evidence: List[EvidenceItem] = []
        endpoints: List[str]         = []
        types:     List[str]         = []
        loop = asyncio.get_event_loop()

        if not graphql_paths:
            return {"evidence": evidence, "endpoints": endpoints, "types": types}

        introspection_query = json.dumps({
            "query": "{ __schema { queryType { name } types { name } } }"
        }).encode()

        for path in graphql_paths:
            url = base + path
            try:
                status, resp_headers, body = await loop.run_in_executor(
                    None, lambda u=url: self._fetch(
                        u, data=introspection_query, method="POST",
                        max_bytes=50_000))
                if status not in (200, 400):
                    continue
                ct = resp_headers.get("Content-Type", "").lower()
                if "json" not in ct and status != 200:
                    continue
                try:
                    data = json.loads(body)
                except Exception:
                    continue

                schema_data = data.get("data", {})
                if schema_data and "__schema" in schema_data:
                    evidence.append(EvidenceItem.make(
                        "graphql_schema",
                        f"GraphQL introspection at {path}",
                        "graphql", endpoint=url))
                    endpoints.append(url)
                    gql_types = [
                        t.get("name", "") for t in
                        schema_data.get("__schema", {}).get("types", [])
                        if t.get("name", "")
                    ]
                    types.extend(gql_types)
                    ai_types = [t for t in gql_types if any(
                        x in t.lower() for x in
                        ["chat", "message", "conversation", "completion",
                         "embedding", "tool", "agent", "model", "stream"]
                    )]
                    if ai_types:
                        evidence.append(EvidenceItem.make(
                            "graphql_ai_field",
                            f"AI-related GraphQL types: {ai_types[:5]}",
                            "graphql", endpoint=url))
                    if verbose:
                        print(f"    [graphql] {path}: {len(gql_types)} types, "
                              f"ai={ai_types[:3]}")
                    break

                if status == 400 and "errors" in body.lower():
                    evidence.append(EvidenceItem.make(
                        "graphql_schema",
                        f"GraphQL endpoint at {path} (introspection disabled)",
                        "graphql", endpoint=url))
                    endpoints.append(url)
                    break

            except Exception:
                pass

        return {"evidence": evidence, "endpoints": endpoints, "types": types}

    async def _fetch_robots_sitemap(self, base: str, verbose: bool) -> Dict:
        evidence: List[EvidenceItem] = []
        paths:    List[str]          = []
        loop = asyncio.get_event_loop()

        try:
            _, _, body = await loop.run_in_executor(
                None, lambda: self._fetch(base + "/robots.txt"))
            if body:
                for line in body.splitlines():
                    line = line.strip()
                    if line.lower().startswith("disallow:") or line.lower().startswith("allow:"):
                        path = line.split(":", 1)[1].strip()
                        if path and path != "/" and self._looks_api_candidate(path):
                            norm = self._to_path(path)
                            paths.append(norm)
                            if _is_inference_path(norm):
                                evidence.append(EvidenceItem.make(
                                    "robots_ai_path",
                                    f"robots.txt AI-shaped path: {norm[:60]}",
                                    "passive"))
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line.split(":", 1)[1].strip()
                        try:
                            _, _, smap = await loop.run_in_executor(
                                None, lambda u=sitemap_url: self._fetch(u, max_bytes=50_000))
                            for m in re.finditer(r'<loc>([^<]+)</loc>', smap):
                                url = m.group(1).strip()
                                parsed = urllib.parse.urlparse(url)
                                if parsed.path and self._looks_api_candidate(parsed.path):
                                    norm = self._to_path(parsed.path)
                                    paths.append(norm)
                                    if _is_inference_path(norm):
                                        evidence.append(EvidenceItem.make(
                                            "sitemap_ai_path",
                                            f"sitemap AI-shaped path: {norm[:60]}",
                                            "passive"))
                        except Exception:
                            pass
        except Exception:
            pass

        return {"evidence": evidence, "paths": list(dict.fromkeys(paths))}

    def _is_related_host_api_probe(
        self,
        url: str,
        related_api_hosts: Optional[Set[str]],
        page_host: str,
    ) -> bool:
        """True when probing an absolute API URL on a JS-allowlisted related host."""
        if not related_api_hosts or not (
            url.startswith("http://") or url.startswith("https://")
        ):
            return False
        host = (urllib.parse.urlparse(url).netloc or "").lower()
        page = (page_host or "").lower()
        if not host or host == page or host not in related_api_hosts:
            return False
        return self._looks_absolute_api_url(url)

    async def _probe_candidates(
        self, base: str, candidates: List[str], verbose: bool,
        spa_catch_all: bool = False,
        openapi_schema: Optional[Dict] = None,
        related_api_hosts: Optional[Set[str]] = None,
        page_host: str = "",
    ) -> Dict:
        """Probe application-discovered paths/URLs only — no catalog spray."""
        evidence:    List[EvidenceItem] = []
        endpoints:   List[str]          = []
        frameworks:  List[str]          = []
        auth_signals: List[AuthSignal]  = []
        loop = asyncio.get_event_loop()
        related_api_hosts = related_api_hosts or set()
        page_host = page_host or (urllib.parse.urlparse(base).netloc or "")

        async def probe(candidate: str, category: str):
            if candidate.startswith("http://") or candidate.startswith("https://"):
                url = candidate
                path = urllib.parse.urlparse(candidate).path or "/"
            else:
                url = base + candidate
                path = candidate
            try:
                status, resp_headers, body = await loop.run_in_executor(
                    None, lambda: self._fetch(url))
                ct = (resp_headers.get("Content-Type")
                      or resp_headers.get("content-type") or "")
                use_spa = spa_catch_all and url.startswith(base)
                needs_post = (
                    status in (404, 405)
                    or (status == 200 and _looks_html_body(ct, body))
                    or (use_spa and status == 200 and category in (
                        "chat", "embedding", "ml", "graphql", "schema"))
                )
                if needs_post:
                    post_body = _probe_post_body(path, category, openapi_schema)
                    status, resp_headers, body = await loop.run_in_executor(
                        None,
                        lambda u=url, d=post_body: self._fetch(u, data=d, method="POST"),
                    )
                return self._evaluate_probe_response(
                    url, path, category, status, resp_headers, body,
                    auth_signals, spa_catch_all=use_spa,
                    related_api_hosts=related_api_hosts,
                    page_host=page_host,
                )
            except Exception:
                return None

        tasks = [
            probe(path, self._guess_probe_category(
                urllib.parse.urlparse(path).path if path.startswith("http") else path
            ))
            for path in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, dict) and r:
                evidence.extend(r.get("evidence", []))
                endpoints.extend(r.get("endpoints", []))
                frameworks.extend(r.get("frameworks", []))

        if verbose:
            print(f"    [probe] confirmed {len(set(endpoints))} live surface(s) "
                  f"from {len(candidates)} candidate(s)")

        return {
            "evidence":    evidence,
            "endpoints":   list(dict.fromkeys(endpoints)),
            "frameworks":  list(dict.fromkeys(frameworks)),
            "auth_signals": auth_signals,
        }

    def _evaluate_probe_response(
        self,
        url: str,
        path: str,
        category: str,
        status: int,
        headers: Dict,
        body: str,
        auth_signals: List[AuthSignal],
        spa_catch_all: bool = False,
        related_api_hosts: Optional[Set[str]] = None,
        page_host: str = "",
    ) -> Dict:
        evidence:  List[EvidenceItem] = []
        endpoints: List[str]          = []
        frameworks: List[str]         = []

        if status == 0:
            return {}

        headers_low = {k.lower(): v for k, v in headers.items()}
        ct = headers_low.get("content-type", "")
        body_l = (body or "").lower()
        related_api = self._is_related_host_api_probe(
            url, related_api_hosts, page_host,
        )
        inference = _is_inference_path(path)
        # Advisory protocol hint from distinctive response headers. Only ever
        # attached to evidence the response already produced below, so it can
        # never add a scored item or change AI-presence.
        header_det = detect_protocol_from_headers(headers_low)
        header_proto = header_det[0] if header_det else ""
        # Hosting provider (Azure / OpenAI) from host/path/headers/error body.
        # Attached only to evidence this response already produced (below), so
        # it never adds a scored item or changes AI-presence.
        prov_det = detect_provider(url=url, headers=headers_low, body=body)
        prov_id = prov_det[0] if prov_det else ""

        if _looks_html_body(ct, body):
            return {}

        if status == 200 and body:
            is_json = "json" in ct or body.strip()[:1] in "{["
            ai_shaped = is_json and _looks_ai_api_json(body)
            proto_id = ""
            if ai_shaped:
                det = detect_protocol_from_json(body)
                proto_id = (det[0] if det else "") or header_proto
            if ai_shaped:
                ev_key = ("chat_endpoint_200" if category == "chat" or inference else
                          "embedding_endpoint_200" if category == "embedding" else
                          "ml_endpoint_200")
                evidence.append(EvidenceItem.make(
                    ev_key,
                    f"AI body at {path} (HTTP 200)",
                    "probe", endpoint=url, protocol=proto_id))
                endpoints.append(url)
                if category in ("chat", "schema") or inference:
                    frameworks.append("OpenAI-compatible")

            if "text/event-stream" in ct:
                evidence.append(EvidenceItem.make(
                    "sse_streaming_header",
                    f"SSE at {path}", "probe", endpoint=url))
                endpoints.append(url)


            if (category == "schema" and ai_shaped and url not in endpoints):
                evidence.append(EvidenceItem.make(
                    "ml_endpoint_200",
                    f"AI schema/config at {path} (HTTP 200)",
                    "probe", endpoint=url, protocol=proto_id))
                endpoints.append(url)
                if any(k in body_l for k in ("choices", '"object"', "chat.completion")):
                    frameworks.append("OpenAI-compatible")

        if status in (401, 403):
            if _looks_static_storage_error(ct, body):
                return {
                    "evidence": evidence,
                    "endpoints": endpoints,
                    "frameworks": frameworks,
                }
            auth_type = self._detect_auth_type(headers_low, body_l)
            auth_status, reason = extract_auth_reason(status, body, headers_low)
            auth_signals.append(AuthSignal(
                endpoint=url, auth_type=auth_type,
                status_code=status,
                detail=f"Auth required ({auth_type}) at {path}",
                auth_status=auth_status, reason=reason))
            if inference or (
                related_api and _looks_api_auth_rejection(ct, body, headers_low)
            ):
                evidence.append(EvidenceItem.make(
                    "ml_endpoint_401_403",
                    f"{'Authentication required' if status == 401 else 'Access forbidden'} "
                    f"at {path} (HTTP {status}, {auth_type}, reason={reason})",
                    "probe", endpoint=url, protocol=header_proto))
                endpoints.append(url)

        if (
            status == 405
            and not spa_catch_all
            and (inference or related_api)
            and _looks_api_method_rejection(ct, body, headers_low)
        ):
            evidence.append(EvidenceItem.make(
                "ml_endpoint_405",
                f"Endpoint exists {path} (HTTP 405)",
                "probe", endpoint=url))
            endpoints.append(url)

        # AWS Bedrock error: an AWS service-exception marker ONLY counts as
        # Bedrock on a Bedrock Runtime path (generic AWS errors elsewhere do not).
        bedrock_err = (
            status >= 400
            and is_bedrock_endpoint(path)
            and looks_bedrock_error(body, headers_low)
        )
        if bedrock_err:
            evidence.append(EvidenceItem.make(
                "error_bedrock_format",
                f"AWS Bedrock error at {path} (HTTP {status})",
                "probe", endpoint=url, framework="AWS Bedrock", protocol=BEDROCK))
            if url not in endpoints:
                endpoints.append(url)

        if status in (400, 422) and body and not bedrock_err:
            fw = self._fingerprint_error_body(body, url, evidence)
            ai_error_keys = {
                "error_openai_format",
                "error_anthropic_format",
                "error_gemini_format",
                "error_tgi_format",
                "error_bedrock_format",
                "error_missing_messages",
                "error_missing_model",
            }
            emitted_ai_error = any(
                ev.key in ai_error_keys and ev.endpoint == url for ev in evidence
            )
            if fw and (inference or emitted_ai_error or category in ("chat", "embedding", "ml")):
                frameworks.append(fw)
                if url not in endpoints:
                    endpoints.append(url)
            elif (
                (inference or category in ("chat", "embedding", "ml"))
                and ("json" in ct or body.strip()[:1] in "{[")
                and not self._looks_auth_account_path(path)
            ):
                evidence.append(EvidenceItem.make(
                    "error_generic_validation",
                    f"API validation response at {path} (HTTP {status})",
                    "probe", endpoint=url))
                endpoints.append(url)
            elif fw:
                evidence[:] = [
                    ev for ev in evidence
                    if not (
                        ev.key == "error_generic_validation"
                        and ev.endpoint == url
                    )
                ]

        for h in AI_RESPONSE_HEADERS:
            if h in headers_low:
                evidence.append(EvidenceItem.make(
                    "ai_response_header",
                    f"{h}: {headers_low[h][:50]}",
                    "probe", endpoint=url))
                endpoints.append(url)
                break

        # Attach the hosting provider to the evidence this response produced.
        # Provider is a separate axis; it rides existing scored keys and never
        # adds one, so it cannot influence AI-presence scoring.
        if prov_id:
            for ev in evidence:
                if not ev.provider:
                    ev.provider = prov_id

        return {
            "evidence":  evidence,
            "endpoints": list(dict.fromkeys(endpoints)),
            "frameworks": frameworks,
        }

    @staticmethod
    def _detect_auth_type(headers_low: Dict, body_l: str) -> str:
        www_auth = headers_low.get("www-authenticate", "").lower()
        if "bearer" in www_auth or "jwt" in body_l or "token" in www_auth:
            return "bearer"
        if "api" in www_auth and "key" in www_auth:
            return "api_key"
        if "basic" in www_auth:
            return "basic"
        if "oauth" in www_auth or "oauth" in body_l:
            return "oauth"
        if "session" in body_l or "cookie" in body_l:
            return "cookie"
        return "unknown"

    def _fingerprint_error_body(
        self, body: str, url: str, evidence: List[EvidenceItem]
    ) -> Optional[str]:
        for pattern, ev_key, framework in ERROR_FINGERPRINTS:
            if pattern.search(body):
                evidence.append(EvidenceItem.make(
                    ev_key,
                    f"{framework} error at {urllib.parse.urlparse(url).path}",
                    "probe", endpoint=url, framework=framework,
                    protocol=ERROR_KEY_PROTOCOL.get(ev_key, "")))
                return framework
        return None

    async def _behavioral_probes(
        self, candidate_endpoints: List[str], verbose: bool,
        openapi_schema: Optional[Dict] = None,
    ) -> Dict:
        """Shape-adaptive behavioral confirmation."""
        evidence:      List[EvidenceItem] = []
        model_reported: str              = ""
        loop = asyncio.get_event_loop()

        def _extract(body: str) -> str:
            text = extract_generated_text(body)
            return text if text else (body or "")

        async def _post(endpoint: str, body_bytes: bytes):
            return await loop.run_in_executor(
                None, lambda u=endpoint, d=body_bytes: self._fetch(
                    u, data=d, method="POST"))

        for endpoint in candidate_endpoints[:2]:
            path = urllib.parse.urlparse(endpoint).path

            hello_bodies: List[Tuple[str, bytes]] = []
            try:
                derived = _probe_post_body(
                    path, self._guess_probe_category(path), openapi_schema)
                if derived:
                    hello_bodies.append(("derived", derived))
            except Exception:
                pass
            for kind, builder in PROBE_SHAPE_BUILDERS:
                hello_bodies.append((kind, json.dumps(builder("Hello")).encode()))

            working_kind: Optional[str] = None
            for kind, body_bytes in hello_bodies:
                try:
                    status, resp_headers, body = await _post(endpoint, body_bytes)
                except Exception as e:
                    logger.debug(f"Behavioral probe failed on {endpoint}: {e}")
                    continue
                if status not in (200, 201):
                    continue
                text = _extract(body)
                if not text:
                    continue
                working_kind = kind

                if any(re.search(sig, text, re.I) for sig in LLM_RESPONSE_SIGNALS):
                    det = detect_protocol_from_json(body)
                    evidence.append(EvidenceItem.make(
                        "behavior_llm_response",
                        f"LLM-shaped response to 'Hello' at {path} "
                        f"(request shape: {kind})",
                        "behavior", endpoint=endpoint,
                        protocol=(det[0] if det else "")))
                    if verbose:
                        print(f"    [behavior] LLM response at {endpoint} [{kind}]")

                ct = (resp_headers.get("Content-Type", "")
                      or resp_headers.get("content-type", "")).lower()
                if "event-stream" in ct:
                    evidence.append(EvidenceItem.make(
                        "behavior_streaming",
                        f"Streaming response at {path}",
                        "behavior", endpoint=endpoint))
                break

            if working_kind is None:
                continue

            builder = _PROBE_SHAPE_MAP.get(
                working_kind, _PROBE_SHAPE_MAP["openai_chat"])

            def _payload(content: str, _builder=builder) -> bytes:
                shaped = _builder(content)
                if isinstance(shaped, dict):
                    shaped.pop("model", None)
                return json.dumps(shaped).encode()

            canary = f"AIPROBE{secrets.token_hex(3).upper()}"
            instr = (f"Repeat the following token exactly and output nothing "
                     f"else: {canary}")
            try:
                status, _h, body = await _post(endpoint, _payload(instr))
                if status in (200, 201):
                    reply = _extract(body)
                    low = reply.lower()
                    if (canary in reply
                            and "repeat the following token" not in low
                            and instr.lower() not in low):
                        evidence.append(EvidenceItem.make(
                            "behavior_instruction_followed",
                            f"Endpoint followed a natural-language instruction "
                            f"(returned canary) at {path}",
                            "behavior", endpoint=endpoint))
                        if verbose:
                            print(f"    [behavior] instruction followed at {endpoint}")
            except Exception as e:
                logger.debug(f"Instruction probe failed on {endpoint}: {e}")

            identity_q = "What model or system are you? Answer in one short sentence."
            try:
                status, _resp_headers, body = await _post(endpoint, _payload(identity_q))
                if status in (200, 201):
                    text = _extract(body)
                    if identity_q.lower() not in text.lower():
                        for pattern in MODEL_NAME_PATTERNS:
                            m = re.search(pattern, text, re.I)
                            if m:
                                model_reported = m.group(0)
                                evidence.append(EvidenceItem.make(
                                    "behavior_model_reported",
                                    f"Model self-report: {model_reported}",
                                    "behavior", endpoint=endpoint))
                                if verbose:
                                    print(f"    [behavior] model={model_reported}")
                                break
            except Exception as e:
                logger.debug(f"Model-query probe failed on {endpoint}: {e}")

        return {"evidence": evidence, "model_reported": model_reported}

    async def _nondeterminism_probe(self, endpoint: str, verbose: bool,
                                    openapi_schema: Optional[Dict] = None) -> Dict:
        evidence: List[EvidenceItem] = []
        loop = asyncio.get_event_loop()
        path = urllib.parse.urlparse(endpoint).path

        candidate_bodies: List[bytes] = []
        try:
            derived = _probe_post_body(
                path, self._guess_probe_category(path), openapi_schema)
            if derived:
                candidate_bodies.append(derived)
        except Exception:
            pass
        candidate_bodies.append(json.dumps({
            "model": self.probe_model,
            "messages": [{"role": "user", "content": self.probe_prompt}],
        }).encode())
        for _kind, builder in PROBE_SHAPE_BUILDERS:
            candidate_bodies.append(json.dumps(builder(self.probe_prompt)).encode())

        payload: Optional[bytes] = None
        for body_bytes in candidate_bodies:
            try:
                status, _, body = await loop.run_in_executor(
                    None,
                    lambda ep=endpoint, data=body_bytes: self._fetch(
                        ep, data=data, method="POST"),
                )
            except Exception:
                continue
            if status in (200, 201) and body and self._extract_text(body):
                payload = body_bytes
                break
        if payload is None:
            return {"evidence": evidence}

        responses: List[str] = []
        for _ in range(max(2, self.nondeterminism_samples)):
            try:
                status, _, body = await loop.run_in_executor(
                    None,
                    lambda ep=endpoint, data=payload: self._fetch(
                        ep, data=data, method="POST"),
                )
                if status in (200, 201) and body:
                    text = self._extract_text(body)
                    if text:
                        responses.append(text)
            except Exception:
                continue

        if len(responses) < 2:
            return {"evidence": evidence}

        divergences = [
            self._token_divergence(responses[i], responses[j])
            for i in range(len(responses))
            for j in range(i + 1, len(responses))
        ]
        mean_div = sum(divergences) / len(divergences)
        if self.nondeterminism_low < mean_div < self.nondeterminism_high:
            evidence.append(EvidenceItem.make(
                "behavior_nondeterministic",
                f"Non-deterministic output over {len(responses)} samples "
                f"(mean divergence {mean_div:.2f})",
                "behavior", endpoint=endpoint))
            if verbose:
                print(f"    [behavior] non-determinism at {endpoint} div={mean_div:.2f}")
        return {"evidence": evidence}

    @staticmethod
    def _token_divergence(a: str, b: str) -> float:
        ta = set(re.findall(r"\w+", a.lower()))
        tb = set(re.findall(r"\w+", b.lower()))
        if not ta and not tb:
            return 0.0
        if not ta or not tb:
            return 1.0
        return 1.0 - len(ta & tb) / len(ta | tb)

    @staticmethod
    def _extract_text(body: str) -> str:
        text = extract_generated_text(body)
        return text if text else body

    async def _analyse_headers_cookies(
        self, headers: Dict, verbose: bool
    ) -> Dict:
        evidence:   List[EvidenceItem] = []
        frameworks: List[str]          = []
        headers_low = {k.lower(): v for k, v in headers.items()}

        for h in AI_RESPONSE_HEADERS:
            if h in headers_low:
                evidence.append(EvidenceItem.make(
                    "ai_response_header",
                    f"{h}: {headers_low[h][:50]}", "header"))
                break

        return {"evidence": evidence, "frameworks": list(dict.fromkeys(frameworks))}

    def _identify_protocol(
        self,
        html: str,
        js: str,
        headers: Dict,
        known_endpoints: List[str],
        existing_evidence: List[EvidenceItem],
    ) -> Dict:
        """Label protocol from live evidence shapes — not path catalogs."""
        evidence: List[EvidenceItem] = []
        keys = {ev.key for ev in existing_evidence}
        signals: List[str] = []
        framework = ""
        protocol = ""

        if keys & {
            "chat_endpoint_200", "error_openai_format", "error_missing_messages",
            "error_missing_model", "embedding_endpoint_200",
        }:
            framework = "OpenAI-compatible"
            protocol = "OpenAI-compatible"
            signals.append("live_surface")
        elif "ml_endpoint_200" in keys:
            framework = "chat_api"
            protocol = "chat_api"
            signals.append("ai_shaped_ml_surface")

        fw_confidence = 0.8 if framework else 0.0
        if framework:
            evidence.append(EvidenceItem.make(
                "framework_identified",
                f"Protocol: {framework} signals={signals[:4]}",
                "passive", framework=framework))

        return {
            "evidence":             evidence,
            "framework":            framework,
            "framework_confidence": fw_confidence,
            "framework_signals":    signals,
            "protocol":             protocol or framework,
            "inference_server":     framework,
        }

    def _infer_capabilities(
        self,
        framework: str,
        all_frameworks: List[str],
        openapi_schema: Dict,
        endpoints: List[str],
        graphql_types: List[str],
    ) -> List[str]:
        caps: Set[str] = set()

        if framework in ("OpenAI-compatible", "chat_api"):
            caps.add("chat")

        if openapi_schema:
            spec_paths = openapi_schema.get("paths", {})
            for path in spec_paths:
                path_l = path.lower()
                if any(tok in path_l for tok in (
                    "chat", "complet", "generate", "message", "predict", "invoke",
                )):
                    caps.add("chat")
                if "embed" in path_l:
                    caps.add("embedding")
                if "image" in path_l or "vision" in path_l:
                    caps.add("image_generation")
                if "audio" in path_l or "speech" in path_l:
                    caps.add("speech")
                if "search" in path_l:
                    caps.add("web_search")
                if "tool" in path_l or "function" in path_l:
                    caps.add("tool_calling")
                if "memory" in path_l or "remember" in path_l:
                    caps.add("memory")

        for ep in endpoints:
            path = urllib.parse.urlparse(ep).path.lower()
            if "embed" in path:
                caps.add("embedding")
            if "rag" in path or "retriev" in path or "vector" in path:
                caps.add("rag")
            if "memory" in path:
                caps.add("memory")
            if "agent" in path:
                caps.add("multi_agent")
            if "tool" in path:
                caps.add("tool_calling")
            if "code" in path or "exec" in path:
                caps.add("code_execution")

        for t in graphql_types:
            t_l = t.lower()
            if "chat" in t_l or "message" in t_l:
                caps.add("chat")
            if "tool" in t_l:
                caps.add("tool_calling")
            if "embed" in t_l:
                caps.add("embedding")
            if "agent" in t_l:
                caps.add("multi_agent")

        return sorted(cap for cap in caps if cap in KNOWN_CAPABILITIES)

    async def _confirm_js_extracted_paths(
        self, base: str, paths: List[str], verbose: bool
    ) -> Dict:
        return await self._confirm_js_paths(base, paths, verbose)

    async def _confirm_js_paths(
        self, base: str, paths: List[str], verbose: bool,
        spa_catch_all: bool = False,
    ) -> Dict:
        evidence:  List[EvidenceItem] = []
        endpoints: List[str]          = []
        auth_signals: List[AuthSignal] = []
        loop = asyncio.get_event_loop()

        async def probe(path: str):
            if not path.startswith("/"):
                path = "/" + path.lstrip("/")
            url = base + path
            try:
                status, hdrs, body = await loop.run_in_executor(
                    None, lambda: self._fetch(
                        url, data=b"{}", method="POST"))
                ct = hdrs.get("Content-Type") or hdrs.get("content-type") or ""
                if status == 0 or _looks_html_body(ct, body):
                    status, hdrs, body = await loop.run_in_executor(
                        None, lambda: self._fetch(url))
                r = self._evaluate_probe_response(
                    url, path, "chat", status, hdrs, body, auth_signals,
                    spa_catch_all=spa_catch_all)
                if r:
                    evidence.extend(r.get("evidence", []))
                    endpoints.extend(r.get("endpoints", []))
            except Exception:
                pass

        await asyncio.gather(*[probe(p) for p in paths[:8]], return_exceptions=True)
        return {
            "evidence": evidence,
            "endpoints": list(dict.fromkeys(endpoints)),
            "auth_signals": auth_signals,
        }

    async def _check_discovered_endpoints(
        self, endpoints: List[Dict]
    ) -> Dict:
        evidence:    List[EvidenceItem] = []
        ml_endpoints: List[str]         = []
        loop = asyncio.get_event_loop()

        for ep in (endpoints or []):
            url    = ep.get("url", "")
            status = ep.get("status", 0)
            if status == 200:
                try:
                    _, hdrs, body = await loop.run_in_executor(
                        None, lambda u=url: self._fetch(u))
                    if "json" in hdrs.get("Content-Type", "").lower() and body:
                        if _looks_ai_api_json(body):
                            ml_endpoints.append(url)
                            path = urllib.parse.urlparse(url).path
                            evidence.append(EvidenceItem.make(
                                "ml_endpoint_200",
                                f"AI body at discovered endpoint {path}",
                                "probe", endpoint=url))
                except Exception:
                    pass

        return {"evidence": evidence, "endpoints": list(dict.fromkeys(ml_endpoints))}

    def _fetch(
        self,
        url: str,
        max_bytes: int = 8192,
        method: str = "GET",
        data: Optional[bytes] = None,
    ) -> Tuple[int, Dict, str]:
        def _once(target: str) -> Tuple[int, Dict, str]:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; AISecurityScanner/1.0)",
                "Accept":     "text/html,application/json,application/xhtml+xml,*/*;q=0.8",
                **({"Content-Type": "application/json"} if data else {}),
                **self.auth_headers,
            }
            req = urllib.request.Request(
                target, data=data,
                headers=headers,
                method=method,
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read(max_bytes).decode("utf-8", errors="replace")
                return resp.status, dict(resp.headers), body

        try:
            return _once(url)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read(max(2048, max_bytes)).decode("utf-8", errors="replace")
            except Exception:
                body = ""
            return exc.code, dict(exc.headers) if exc.headers else {}, body
        except Exception as exc:
            alt = None
            if "://localhost" in url.lower() or "://[::1]" in url.lower():
                alt = (url.replace("://localhost", "://127.0.0.1")
                          .replace("://Localhost", "://127.0.0.1")
                          .replace("://[::1]", "://127.0.0.1"))
            if alt and alt != url:
                try:
                    return _once(alt)
                except urllib.error.HTTPError as exc2:
                    try:
                        body = exc2.read(max(2048, max_bytes)).decode("utf-8", errors="replace")
                    except Exception:
                        body = ""
                    return exc2.code, dict(exc2.headers) if exc2.headers else {}, body
                except Exception as exc2:
                    logger.warning("[fingerprint] fetch failed %s (%s); retry %s also failed (%s)",
                                   url, exc, alt, exc2)
                    return 0, {}, ""
            logger.warning("[fingerprint] fetch failed %s: %s", url, exc)
            return 0, {}, ""
