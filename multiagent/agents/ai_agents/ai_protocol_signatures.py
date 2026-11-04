"""Protocol / API-format identification for discovered AI surfaces.

This module is deliberately separate from ``ai_fingerprint.py``:

* ``ai_fingerprint`` answers **AI presence** ("is there a live, attackable AI
  surface?") and owns the evidence-scoring verdict.
* This module answers the *orthogonal* question **AI protocol / format**
  ("which API wire format does that surface appear to speak?").

Protocol identification never changes AI-presence confidence. It reads the same
``EvidenceItem`` stream (via a ``protocol`` tag each item may carry) and resolves
a single protocol label plus a confidence. Presence scoring in
``ai_evidence_scoring.score_evidence`` ignores the ``protocol`` tag entirely, so
adding or removing a protocol signature can never flip ``is_ai``.

Phase 1 implements **Ollama** and the existing **OpenAI-compatible** signature
end-to-end; Phase 2 adds **Anthropic Messages** (response shape, ``content[]``
text extraction, error envelope, response headers); Phase 3 adds **Google
Gemini** (``candidates[].content.parts[].text`` response shape and extraction,
``generativelanguage`` error envelope, and ``:generateContent`` endpoint
recognition — non-streaming only); Phase 4 adds **Cohere** (Chat/Generate
response shape via ``generation_id``/``response_id``/``generations[]``/uppercase
``finish_reason``/``meta`` combinations, and text extraction); Phase 5 adds
**Hugging Face TGI** (native ``/generate`` object with a ``details`` block and
the ``[{"generated_text": ...}]`` batch form, text extraction, ``error_type``
error envelope, and compute/token response headers) — a bare ``generated_text``
stays generic. Phase 6 adds the **provider axis** — **Azure OpenAI** is detected
as a *provider* (host ``*.openai.azure.com`` / ``/openai/deployments/`` path /
APIM+Azure-ML headers / content-filter error) while its protocol stays
``OpenAI-compatible``; ``provider`` and ``provider_confidence`` resolve
independently of the protocol. Phase 7 adds **AWS Bedrock** — both a protocol
(Converse ``output.message.content[]`` + camelCase ``stopReason``/``usage``;
Titan ``results[].outputText``; Llama ``generation``) and a provider (host
``bedrock-runtime.*.amazonaws.com`` / ``/model/{id}/(invoke|converse…)`` path /
``x-amzn-bedrock-*`` headers). Bedrock InvokeModel bodies that match another
vendor (e.g. Anthropic on Bedrock) resolve to that protocol with provider
"AWS Bedrock". Plus a generic fallback. The signature registry and resolver stay
open for further protocols without touching the detection pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

try:
    from .ai_evidence_scoring import EvidenceClass, classify_evidence
except ImportError:
    try:
        from multiagent.agents.ai_agents.ai_evidence_scoring import (
            EvidenceClass,
            classify_evidence,
        )
    except ImportError:
        from ai_evidence_scoring import EvidenceClass, classify_evidence


# --- Internal protocol identifiers (stable keys used on EvidenceItem.protocol) ---
OPENAI = "openai"
OLLAMA = "ollama"
ANTHROPIC = "anthropic_messages"
GEMINI = "gemini"
COHERE = "cohere"
HF_TGI = "hf_tgi"
BEDROCK = "bedrock"
GENERIC_CHAT = "generic_chat"

UNKNOWN_AI_PROTOCOL = "Unknown AI Protocol"

# Internal id -> human-facing label shown in the final result.
# NOTE: Azure OpenAI is intentionally NOT a protocol — it is a *provider* that
# exposes the OpenAI-compatible wire protocol (see the provider axis below).
PROTOCOL_DISPLAY: Dict[str, str] = {
    OPENAI: "OpenAI-compatible",
    ANTHROPIC: "Anthropic Messages",
    OLLAMA: "Ollama",
    GEMINI: "Google Gemini",
    COHERE: "Cohere",
    HF_TGI: "Hugging Face TGI",
    BEDROCK: "AWS Bedrock",
    GENERIC_CHAT: "Generic Chat API",
}

# Fallback provider implied by certain protocols (e.g. AWS Bedrock is both a
# distinct protocol AND a provider). Azure is NOT here — it rides OpenAI's wire
# protocol, so it is detected on the independent provider axis below.
PROTOCOL_PROVIDER: Dict[str, str] = {
    BEDROCK: "AWS Bedrock",
}

# Deterministic precedence when two specific protocols tie on confidence.
_SPECIFIC_PRIORITY: Tuple[str, ...] = (
    OLLAMA, ANTHROPIC, GEMINI, COHERE, HF_TGI, BEDROCK, OPENAI,
)

# --- Provider axis (independent of the wire protocol) -------------------------
# A provider is who hosts/fronts the surface; the protocol is the API shape it
# speaks. They are orthogonal: Azure OpenAI => provider "Azure OpenAI" with
# protocol "OpenAI-compatible". Provider is NEVER emitted as standalone scored
# evidence — it is attached to evidence the response already produced, so it can
# never alter AI-presence scoring.
PROVIDER_AZURE = "azure"
PROVIDER_OPENAI = "openai"
PROVIDER_AWS = "aws_bedrock"
PROVIDER_DISPLAY: Dict[str, str] = {
    PROVIDER_AZURE: "Azure OpenAI",
    PROVIDER_OPENAI: "OpenAI",
    PROVIDER_AWS: "AWS Bedrock",
}

# Azure OpenAI / Azure AI Foundry deployment host suffixes.
_AZURE_HOST_SUFFIXES: Tuple[str, ...] = (
    ".openai.azure.com",
    ".cognitiveservices.azure.com",
    ".services.ai.azure.com",
    ".inference.ai.azure.com",
    ".openai.azure-api.net",
)
# Headers that are specific to Azure's AI serving layer (APIM / Azure ML), as
# opposed to generic Azure infra headers (x-ms-request-id / x-ms-region) which
# any Azure service emits and are therefore NOT sufficient on their own.
_AZURE_STRONG_HEADERS: Tuple[str, ...] = (
    "apim-request-id", "azureml-model-session", "azureml-model-group",
)

# AWS Bedrock Runtime invocation-verb endpoint suffixes.
_BEDROCK_VERBS: Tuple[str, ...] = (
    "/invoke", "/invoke-with-response-stream", "/converse", "/converse-stream",
)
# Bedrock Runtime exception names. The AWS-generic ones (Validation/Throttling/
# AccessDenied/ResourceNotFound) only count toward Bedrock when seen on a Bedrock
# path; ModelNotReady/ModelTimeout are Bedrock-specific.
BEDROCK_EXCEPTIONS: Tuple[str, ...] = (
    "ValidationException", "ModelNotReadyException", "ModelTimeoutException",
    "ThrottlingException", "AccessDeniedException", "ResourceNotFoundException",
    "ServiceQuotaExceededException", "ModelErrorException",
    "ModelStreamErrorException",
)

# Protocol confidence is derived from the evidence tier the signature rode in on
# (live/behavioral observation > confirmed route/error body > passive hint). The
# per-signature HIGH/MEDIUM strength returned by detectors is advisory for future
# use; tier is authoritative so a passive mention can never read as HIGH.
_TIER_CONFIDENCE: Dict[EvidenceClass, str] = {
    EvidenceClass.BEHAVIORAL: "HIGH",
    EvidenceClass.LIVE_SURFACE: "HIGH",
    EvidenceClass.ROUTE_CONFIRMED: "MEDIUM",
    EvidenceClass.PASSIVE_STRUCTURAL: "LOW",
    EvidenceClass.PASSIVE_HINT: "LOW",
    EvidenceClass.BRAND: "LOW",
}
_CONF_RANK: Dict[str, int] = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

# Distinctive error-body fingerprints -> protocol. Only genuinely vendor-specific
# error envelopes belong here; a bare ``{"error": "..."}`` string or a generic
# HTTP 400/401 is NOT distinctive and must stay out.
ERROR_KEY_PROTOCOL: Dict[str, str] = {
    "error_openai_format": OPENAI,
    "error_anthropic_format": ANTHROPIC,
    "error_gemini_format": GEMINI,
    "error_tgi_format": HF_TGI,
    "error_bedrock_format": BEDROCK,
}

# Hugging Face TGI response ``details`` sub-keys — distinctive to native TGI.
_TGI_DETAIL_KEYS: frozenset = frozenset({
    "finish_reason", "generated_tokens", "prefill", "tokens", "seed",
    "best_of_sequences", "top_tokens",
})

# TGI-specific response headers (compute/token timing), advisory hint only.
TGI_HEADER_TOKENS: Tuple[str, ...] = (
    "x-compute-type", "x-compute-time", "x-compute-characters",
    "x-prompt-tokens", "x-generated-tokens", "x-total-time", "x-queue-time",
)

# Distinctive response headers -> protocol (advisory hint only; see
# ``detect_protocol_from_headers``). These never set AI presence — they are
# attached to evidence the response already produced, never emitted standalone.
ANTHROPIC_HEADER_TOKENS: Tuple[str, ...] = (
    "anthropic-ratelimit-requests-limit",
    "anthropic-ratelimit-tokens-limit",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-tokens-remaining",
    "anthropic-organization-id",
    "anthropic-version",
)


def _as_dict(response: Any) -> Optional[dict]:
    """Coerce a parsed object or a JSON string into a dict, else None."""
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        text = response.strip()
        if not text or text[:1] not in "{[":
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
        return data if isinstance(data, dict) else None
    return None


# --------------------------------------------------------------------------- #
# Per-protocol detectors. Each returns (protocol_id, strength) for a
# sufficiently distinctive signature, or None. Strength is advisory only.
# --------------------------------------------------------------------------- #

def _detect_ollama(data: dict) -> Optional[Tuple[str, str]]:
    """Ollama native API shapes: /api/tags, /api/chat, /api/generate.

    Conservative: a lone ``model``, lone ``response``, or lone ``done`` is NOT
    Ollama. Only the distinctive *combinations* below qualify.
    """
    # /api/tags — model listing with Ollama-specific entry fields. OpenAI's
    # /v1/models uses {"object":"list","data":[...]}, never a top-level
    # "models" list, so this key plus a digest/modified_at is distinctive.
    models = data.get("models")
    if isinstance(models, list) and models and isinstance(models[0], dict):
        entry = models[0]
        if any(k in entry for k in ("digest", "modified_at")) or (
            "model" in entry and "name" in entry
        ):
            return (OLLAMA, "HIGH")

    done = data.get("done")
    has_model = isinstance(data.get("model"), str) and bool(data.get("model"))
    done_is_bool = isinstance(done, bool)

    # /api/chat — a top-level assistant message object plus a done flag and a
    # model. OpenAI nests the message under choices[]; a top-level message +
    # done is Ollama's chat shape.
    msg = data.get("message")
    if (
        done_is_bool
        and has_model
        and isinstance(msg, dict)
        and isinstance(msg.get("content"), str)
        and "role" in msg
    ):
        strong = any(k in data for k in (
            "eval_count", "total_duration", "done_reason",
            "load_duration", "eval_duration", "prompt_eval_count",
        ))
        return (OLLAMA, "HIGH" if strong else "MEDIUM")

    # /api/generate — model + response string + done flag together.
    if done_is_bool and has_model and isinstance(data.get("response"), str):
        strong = any(k in data for k in (
            "eval_count", "total_duration", "done_reason", "context",
            "load_duration", "eval_duration", "prompt_eval_count",
        ))
        return (OLLAMA, "HIGH" if strong else "MEDIUM")

    return None


def _detect_anthropic(data: dict) -> Optional[Tuple[str, str]]:
    """Anthropic Messages API response shape.

    Distinctive = a top-level ``content`` LIST of typed blocks (OpenAI nests a
    string under ``choices[].message.content``; Ollama under
    ``message.content``), PLUS at least two corroborating Anthropic markers.
    A lone ``usage``, ``messages``, or ``model`` field never qualifies.
    """
    content = data.get("content")
    if not (isinstance(content, list) and content and isinstance(content[0], dict)):
        return None
    block = content[0]
    if "type" not in block:
        return None

    usage = data.get("usage")
    has_anthropic_usage = isinstance(usage, dict) and (
        "input_tokens" in usage and "output_tokens" in usage
    )
    ident = data.get("id")

    markers = 0
    if data.get("type") == "message":
        markers += 1
    if data.get("role") == "assistant":
        markers += 1
    if "stop_reason" in data:
        markers += 1
    if isinstance(ident, str) and ident.startswith("msg_"):
        markers += 1
    if has_anthropic_usage:
        markers += 1
    if markers < 2:
        return None

    strong = (
        data.get("type") == "message"
        and ("stop_reason" in data or has_anthropic_usage)
        and block.get("type") in ("text", "tool_use", "thinking")
    )
    return (ANTHROPIC, "HIGH" if strong else "MEDIUM")


def _detect_gemini(data: dict) -> Optional[Tuple[str, str]]:
    """Google Gemini generateContent response shape.

    Distinctive = a ``candidates[]`` list whose candidate nests
    ``content.parts[]`` (Gemini's double nesting), PLUS at least one more Gemini
    marker (``finishReason``, camelCase ``usageMetadata`` token counts,
    ``modelVersion``, or ``content.role == "model"``). A lone ``candidates`` — or
    a lone ``contents``/``parts`` field — never qualifies.
    """
    candidates = data.get("candidates")
    if not (isinstance(candidates, list) and candidates
            and isinstance(candidates[0], dict)):
        return None
    c0 = candidates[0]
    content = c0.get("content")
    parts_nested = (
        isinstance(content, dict)
        and isinstance(content.get("parts"), list)
        and bool(content["parts"])
        and isinstance(content["parts"][0], dict)
    )
    if not parts_nested:
        return None

    markers = 1  # the nested content.parts structure itself
    if "finishReason" in c0:
        markers += 1
    um = data.get("usageMetadata")
    has_gemini_usage = isinstance(um, dict) and any(
        k in um for k in (
            "promptTokenCount", "candidatesTokenCount", "totalTokenCount",
        )
    )
    if has_gemini_usage:
        markers += 1
    if isinstance(data.get("modelVersion"), str):
        markers += 1
    if isinstance(content, dict) and content.get("role") == "model":
        markers += 1

    if markers < 2:
        return None
    strong = has_gemini_usage or "finishReason" in c0
    return (GEMINI, "HIGH" if strong else "MEDIUM")


def is_gemini_endpoint(path: Any) -> bool:
    """True for Gemini generateContent-style endpoint paths.

    e.g. ``/v1beta/models/gemini-pro:generateContent`` or
    ``:streamGenerateContent``. The ``:<verb>`` suffix is distinctive to the
    Google generative-language API.
    """
    if not isinstance(path, str):
        return False
    p = path.split("?")[0].split("#")[0].lower()
    return (
        ":generatecontent" in p
        or ":streamgeneratecontent" in p
        or ":batchgeneratecontent" in p
    )


_COHERE_FINISH_REASONS: frozenset = frozenset({
    "COMPLETE", "STOP_SEQUENCE", "MAX_TOKENS", "TOOL_CALL",
    "ERROR", "ERROR_TOXIC", "ERROR_LIMIT", "USER_CANCEL",
})


def _detect_cohere(data: dict) -> Optional[Tuple[str, str]]:
    """Cohere Chat / Generate response shapes.

    Distinctive = the Cohere-specific ``generation_id`` / ``response_id`` ids, a
    ``generations[]`` list, a ``chat_history`` of ``{role, message}`` turns, the
    uppercase ``finish_reason`` enum, or ``meta.billed_units``/``tokens`` — in
    combination. A bare ``{"text": "..."}`` or a generic ``message``/``model``
    field never qualifies.
    """
    # Cohere Generate API: generations[] of {text, ...}.
    gens = data.get("generations")
    if (isinstance(gens, list) and gens and isinstance(gens[0], dict)
            and isinstance(gens[0].get("text"), str)):
        return (COHERE, "HIGH")

    has_text = isinstance(data.get("text"), str)
    has_gen_id = isinstance(data.get("generation_id"), str)
    has_resp_id = isinstance(data.get("response_id"), str)

    chat_history = data.get("chat_history")
    ch_cohere = (
        isinstance(chat_history, list) and bool(chat_history)
        and isinstance(chat_history[0], dict)
        and "role" in chat_history[0] and "message" in chat_history[0]
    )
    fr = data.get("finish_reason")
    cohere_fr = isinstance(fr, str) and fr in _COHERE_FINISH_REASONS
    meta = data.get("meta")
    cohere_meta = isinstance(meta, dict) and (
        isinstance(meta.get("billed_units"), dict)
        or isinstance(meta.get("tokens"), dict)
        or isinstance(meta.get("api_version"), dict)
    )
    markers = sum((ch_cohere, cohere_fr, cohere_meta, has_resp_id))

    # generation_id is a Cohere-specific field; with generated text it suffices.
    if has_gen_id and (has_text or markers >= 1):
        return (COHERE, "HIGH")
    # response_id plus a corroborating Cohere marker.
    if has_resp_id and has_text and (ch_cohere or cohere_fr or cohere_meta):
        return (COHERE, "HIGH" if (cohere_fr and cohere_meta) else "MEDIUM")
    # text plus two independent Cohere markers (never a single generic field).
    if has_text and markers >= 2:
        return (COHERE, "MEDIUM")
    return None


def _detect_tgi(data: dict) -> Optional[Tuple[str, str]]:
    """Hugging Face TGI native ``/generate`` response (object form).

    Distinctive = ``generated_text`` plus a ``details`` object carrying TGI's
    own fields (``finish_reason``/``generated_tokens``/``prefill``/``tokens``/
    ``seed``). A bare ``generated_text`` with no ``details`` is ambiguous and is
    left to the generic fallback — not labelled TGI.
    """
    gt = data.get("generated_text")
    if not isinstance(gt, str):
        return None
    details = data.get("details")
    if isinstance(details, dict) and any(k in details for k in _TGI_DETAIL_KEYS):
        return (HF_TGI, "HIGH")
    return None


def _detect_tgi_list(response: Any) -> Optional[Tuple[str, str]]:
    """Native TGI / HF text-generation batch form: ``[{"generated_text": ...}]``.

    A top-level JSON array whose first element is ``{"generated_text": <str>}``
    is the HF/TGI generation shape. ``details`` on the item raises it to HIGH.
    """
    parsed = response
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except Exception:
            return None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        item = parsed[0]
        if isinstance(item.get("generated_text"), str):
            details = item.get("details")
            strong = isinstance(details, dict) and any(
                k in details for k in _TGI_DETAIL_KEYS)
            return (HF_TGI, "HIGH" if strong else "MEDIUM")
    return None


def _detect_bedrock(data: dict) -> Optional[Tuple[str, str]]:
    """AWS Bedrock response shapes (Converse + Bedrock-specific InvokeModel).

    - **Converse**: ``output.message.content[]`` nesting plus a camelCase Bedrock
      marker (``stopReason`` / ``usage.inputTokens|outputTokens|totalTokens`` /
      ``metrics``). Distinct from Anthropic (snake_case, top-level ``content``)
      and Gemini (``candidates``).
    - **InvokeModel (Titan)**: ``results[].outputText`` + ``completionReason`` /
      ``inputTextTokenCount``.
    - **InvokeModel (Llama)**: ``generation`` + ``prompt_token_count`` /
      ``generation_token_count``.

    Model-specific InvokeModel bodies that match another vendor (e.g. Anthropic
    on Bedrock) are intentionally left to that vendor's detector — the Bedrock
    *provider* is identified separately from host/path.
    """
    output = data.get("output")
    if isinstance(output, dict):
        msg = output.get("message")
        if (isinstance(msg, dict) and isinstance(msg.get("content"), list)
                and msg["content"]):
            usage = data.get("usage")
            camel_usage = isinstance(usage, dict) and any(
                k in usage for k in ("inputTokens", "outputTokens", "totalTokens"))
            has_stop = "stopReason" in data
            has_metrics = isinstance(data.get("metrics"), dict)
            if has_stop or camel_usage or has_metrics:
                return (BEDROCK, "HIGH" if (has_stop and camel_usage) else "MEDIUM")

    results = data.get("results")
    if (isinstance(results, list) and results and isinstance(results[0], dict)
            and isinstance(results[0].get("outputText"), str)
            and ("completionReason" in results[0] or "inputTextTokenCount" in data)):
        return (BEDROCK, "HIGH")

    if isinstance(data.get("generation"), str) and (
        "prompt_token_count" in data or "generation_token_count" in data
    ):
        return (BEDROCK, "HIGH")

    return None


def is_bedrock_endpoint(path: Any) -> bool:
    """True for Bedrock Runtime invocation paths: /model/{id}/(invoke|converse…)."""
    if not isinstance(path, str):
        return False
    p = path.split("?")[0].split("#")[0].lower()
    return "/model/" in p and any(p.endswith(v) for v in _BEDROCK_VERBS)


def looks_bedrock_error(body: Any = None, headers: Any = None) -> bool:
    """True when a response carries an AWS service-exception marker.

    Not Bedrock-specific on its own (AWS services share these names) — callers
    gate this on a Bedrock endpoint path before treating it as Bedrock evidence.
    """
    if headers:
        try:
            et = str({str(k).lower(): v for k, v in dict(headers).items()}
                     .get("x-amzn-errortype", ""))
        except Exception:
            et = ""
        if any(exc in et for exc in BEDROCK_EXCEPTIONS):
            return True
    if isinstance(body, str) and body and any(exc in body for exc in BEDROCK_EXCEPTIONS):
        return True
    return False


def _detect_openai(data: dict) -> Optional[Tuple[str, str]]:
    """OpenAI-compatible chat/completion/embedding/model-list response shapes."""
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        c0 = choices[0]
        msg = c0.get("message")
        if (
            "text" in c0 or "delta" in c0 or "finish_reason" in c0
            or (isinstance(msg, dict) and "content" in msg)
        ):
            return (OPENAI, "HIGH")

    obj = data.get("object")
    if isinstance(obj, str) and obj in (
        "chat.completion", "chat.completion.chunk", "text_completion", "embedding",
    ):
        return (OPENAI, "HIGH")

    if obj == "list":
        rows = data.get("data")
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            r0 = rows[0]
            if (
                isinstance(r0.get("embedding"), list)
                or r0.get("object") in ("model", "embedding")
                or ("id" in r0 and "model" in r0)
            ):
                return (OPENAI, "MEDIUM")

    return None


def _looks_generic_ai(data: dict) -> bool:
    """AI-shaped JSON that is not attributable to a specific vendor."""
    usage = data.get("usage")
    if isinstance(usage, dict) and any(
        k in usage for k in (
            "prompt_tokens", "completion_tokens", "total_tokens",
            "input_tokens", "output_tokens",
        )
    ):
        return True
    emb = data.get("embedding")
    if isinstance(emb, list) and len(emb) >= 4 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in emb[:8]
    ):
        return True
    if any(k in data for k in ("tool_calls", "function_call", "generated_text")):
        return True
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        return True
    return False


# Ordered detector registry. New protocols append here (before the generic
# fallback) without changing callers.
_DETECTORS: Tuple[Callable[[dict], Optional[Tuple[str, str]]], ...] = (
    _detect_ollama,
    _detect_anthropic,
    _detect_gemini,
    _detect_cohere,
    _detect_bedrock,
    _detect_tgi,
    _detect_openai,
)


def detect_protocol_from_headers(headers: Any) -> Optional[Tuple[str, str]]:
    """Protocol hint from distinctive response headers (advisory only).

    Returns (protocol_id, strength) or None. The caller attaches this to
    evidence the response already produced — it is never emitted as standalone
    evidence, so it cannot change AI-presence scoring.
    """
    if not headers:
        return None
    try:
        low = {str(k).lower() for k in dict(headers).keys()}
    except Exception:
        return None
    if any(h in low for h in ANTHROPIC_HEADER_TOKENS):
        return (ANTHROPIC, "MEDIUM")
    if any(h in low for h in TGI_HEADER_TOKENS):
        return (HF_TGI, "MEDIUM")
    return None


def _host_of(url_or_host: str) -> str:
    s = (url_or_host or "").strip()
    if not s:
        return ""
    netloc = urlsplit(s if "://" in s else "//" + s).netloc.lower()
    return netloc.split("@")[-1].split(":")[0]


def detect_provider_from_host(url_or_host: Any) -> Optional[Tuple[str, str]]:
    """Provider from the endpoint host (provider evidence, not AI proof)."""
    if not isinstance(url_or_host, str):
        return None
    host = _host_of(url_or_host)
    if not host:
        return None
    if any(host.endswith(suf) for suf in _AZURE_HOST_SUFFIXES):
        return (PROVIDER_AZURE, "HIGH")
    if host == "api.openai.com" or host.endswith(".api.openai.com"):
        return (PROVIDER_OPENAI, "HIGH")
    if (host.startswith("bedrock-runtime.") or host.startswith("bedrock.")) \
            and host.endswith(".amazonaws.com"):
        return (PROVIDER_AWS, "HIGH")
    return None


def detect_provider_from_headers(headers: Any) -> Optional[Tuple[str, str]]:
    """Provider from distinctive serving-layer headers (advisory, never presence).

    Only Azure's AI-serving headers (APIM / Azure ML) qualify. Generic Azure
    infra headers (``x-ms-request-id``/``x-ms-region``) are emitted by every
    Azure service and are NOT sufficient on their own.
    """
    if not headers:
        return None
    try:
        low = {str(k).lower() for k in dict(headers).keys()}
    except Exception:
        return None
    if any(h in low for h in _AZURE_STRONG_HEADERS) or any(
        h.startswith("azureml-model-") for h in low
    ):
        return (PROVIDER_AZURE, "HIGH")
    if any(h.startswith("x-amzn-bedrock-") for h in low):
        return (PROVIDER_AWS, "HIGH")
    return None


def detect_provider(url: str = "", headers: Any = None, body: Any = None
                    ) -> Optional[Tuple[str, str]]:
    """Resolve the hosting provider from host + path + headers + error body.

    Combines multiple signals (never hostname alone as proof of AI). Returns
    (provider_id, strength) or None. Azure's ``/openai/deployments/`` path and
    Responsible-AI ``content_filter_result`` error envelope are Azure-specific.
    """
    host_hit = detect_provider_from_host(url)
    if host_hit:
        return host_hit
    if isinstance(url, str) and url:
        path = urlsplit(url if "://" in url else "//" + url).path
        if "/openai/deployments/" in path.lower():
            return (PROVIDER_AZURE, "HIGH")
        if is_bedrock_endpoint(path):
            return (PROVIDER_AWS, "HIGH")
    hdr_hit = detect_provider_from_headers(headers)
    if hdr_hit:
        return hdr_hit
    if isinstance(body, str) and body and (
        "content_filter_result" in body or '"innererror"' in body.lower()
    ):
        return (PROVIDER_AZURE, "MEDIUM")
    return None


def detect_protocol_from_json(response: Any) -> Optional[Tuple[str, str]]:
    """Classify a JSON body/dict into (protocol_id, strength), or None.

    Specific vendor detectors run first; an AI-shaped body with no distinctive
    signature returns (GENERIC_CHAT, "LOW"); non-AI JSON returns None.
    """
    data = _as_dict(response)
    if data is None:
        # Native TGI batch/list response: [{"generated_text": "...", ...}]
        return _detect_tgi_list(response)
    for detector in _DETECTORS:
        hit = detector(data)
        if hit:
            return hit
    if _looks_generic_ai(data):
        return (GENERIC_CHAT, "LOW")
    return None


def extract_generated_text(response: Any, protocol: str = "") -> str:
    """Pull model-generated text out of a response across protocol formats.

    Phase 1 handles OpenAI (choices[].message/text/delta) and Ollama
    (/api/generate ``response``, /api/chat ``message.content``) plus common
    single-field shapes. ``protocol`` is accepted as a forward-compatible hint;
    the auto-ordered fallback already resolves every Phase-1 format, and later
    protocols (Anthropic ``content[].text``, Gemini ``candidates[]...parts[]``,
    Cohere) extend this function rather than duplicating extraction elsewhere.
    """
    data = _as_dict(response)
    if data is None:
        # Native TGI batch/list form: [{"generated_text": "..."}]
        parsed = response
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
            except Exception:
                parsed = None
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            gt = parsed[0].get("generated_text")
            if isinstance(gt, str) and gt:
                return gt
        return ""

    # OpenAI-compatible: choices[0].message.content / text / delta.content
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        c0 = choices[0]
        msg = c0.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg["content"]:
            return msg["content"]
        if isinstance(c0.get("text"), str) and c0["text"]:
            return c0["text"]
        delta = c0.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str) and delta["content"]:
            return delta["content"]

    # Anthropic Messages: top-level content[] of typed blocks; join text blocks.
    content = data.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        joined = "".join(
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") == "text"
            and isinstance(b.get("text"), str) and b["text"]
        )
        if joined:
            return joined

    # Google Gemini: candidates[].content.parts[].text — first text-bearing
    # candidate, all its text parts joined (handles multi-part / multi-candidate).
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            cand_content = cand.get("content")
            parts = cand_content.get("parts") if isinstance(cand_content, dict) else None
            if isinstance(parts, list):
                joined = "".join(
                    p["text"] for p in parts
                    if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"]
                )
                if joined:
                    return joined

    # Cohere Generate: generations[].text (first text-bearing generation).
    # (Cohere Chat's top-level ``text`` is covered by the single-field loop below.)
    gens = data.get("generations")
    if isinstance(gens, list) and gens:
        for g in gens:
            if isinstance(g, dict) and isinstance(g.get("text"), str) and g["text"]:
                return g["text"]

    # AWS Bedrock Converse: output.message.content[].text (joined).
    output = data.get("output")
    if isinstance(output, dict):
        msg = output.get("message")
        parts = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(parts, list):
            joined = "".join(
                b["text"] for b in parts
                if isinstance(b, dict) and isinstance(b.get("text"), str) and b["text"]
            )
            if joined:
                return joined
    # Bedrock Titan InvokeModel: results[].outputText
    results = data.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        ot = results[0].get("outputText")
        if isinstance(ot, str) and ot:
            return ot
    # Bedrock Llama InvokeModel: generation
    gen = data.get("generation")
    if isinstance(gen, str) and gen:
        return gen

    # Ollama /api/generate and other common single-field reply shapes.
    for key in ("response", "output", "answer", "reply", "result",
                "generated_text", "text"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value

    # Ollama /api/chat and other top-level message shapes.
    msg = data.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg["content"]:
        return msg["content"]
    if isinstance(msg, str) and msg:
        return msg

    return ""


@dataclass
class ProtocolResolution:
    """Resolved protocol label + confidence for a fingerprinted surface."""
    protocol: str                       # display label, or "" when no AI surface
    protocol_id: str = ""               # internal id ("ollama", "openai", ...)
    confidence: str = "NONE"            # HIGH / MEDIUM / LOW / NONE
    provider: str = ""                  # hosting provider, independent of protocol
    provider_confidence: str = "NONE"   # HIGH / MEDIUM / LOW / NONE
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "protocol_id": self.protocol_id,
            "protocol_confidence": self.confidence,
            "provider": self.provider,
            "provider_confidence": self.provider_confidence,
            "signals": self.signals,
        }


def resolve_protocol(
    evidence_items: Any,
    has_live_surface: bool = False,
    has_route_confirmed: bool = False,
) -> ProtocolResolution:
    """Aggregate protocol-tagged evidence into one label + confidence.

    Rules:
      1. Aggregate every EvidenceItem's ``protocol`` tag, keeping the highest
         confidence (derived from the item's evidence tier) seen per protocol.
      2. Prefer a specific vendor protocol over the generic fallback.
      3. On a tie, resolve deterministically via ``_SPECIFIC_PRIORITY``.
      4. Only generic AI evidence -> "Generic Chat API".
      5. An AI surface with no protocol evidence -> "Unknown AI Protocol"
         (never downgrades presence — the caller already set is_ai).
      6. No AI evidence at all -> "" (empty, as before).
    """
    best: Dict[str, str] = {}
    prov_best: Dict[str, str] = {}
    for ev in evidence_items or []:
        key = getattr(ev, "key", "") or ""
        tier = classify_evidence(key) if key else EvidenceClass.BRAND
        conf = _TIER_CONFIDENCE.get(tier, "LOW")
        pid = (getattr(ev, "protocol", "") or "").strip()
        if pid and _CONF_RANK[conf] > _CONF_RANK.get(best.get(pid, "NONE"), 0):
            best[pid] = conf
        prov = (getattr(ev, "provider", "") or "").strip()
        if prov and _CONF_RANK[conf] > _CONF_RANK.get(prov_best.get(prov, "NONE"), 0):
            prov_best[prov] = conf

    # Provider axis — resolved independently; never competes with the protocol.
    provider = ""
    provider_conf = "NONE"
    if prov_best:
        prov_id, provider_conf = max(
            prov_best.items(), key=lambda kv: (_CONF_RANK[kv[1]], kv[0]))
        provider = PROVIDER_DISPLAY.get(prov_id, "")

    def _with_provider(res: "ProtocolResolution") -> "ProtocolResolution":
        if provider:                       # a detected provider tag wins
            res.provider = provider
            res.provider_confidence = provider_conf
        elif res.protocol_id in PROTOCOL_PROVIDER:   # provider implied by protocol
            res.provider = PROTOCOL_PROVIDER[res.protocol_id]
            res.provider_confidence = res.confidence
        return res

    specific = {pid: c for pid, c in best.items() if pid != GENERIC_CHAT}
    if specific:
        def _rank(item: Tuple[str, str]) -> Tuple[int, int]:
            pid, conf = item
            pri = (
                _SPECIFIC_PRIORITY.index(pid)
                if pid in _SPECIFIC_PRIORITY else len(_SPECIFIC_PRIORITY)
            )
            return (_CONF_RANK[conf], -pri)

        pid, conf = max(specific.items(), key=_rank)
        return _with_provider(ProtocolResolution(
            protocol=PROTOCOL_DISPLAY.get(pid, UNKNOWN_AI_PROTOCOL),
            protocol_id=pid,
            confidence=conf,
            signals=sorted(specific),
        ))

    if GENERIC_CHAT in best:
        return _with_provider(ProtocolResolution(
            protocol=PROTOCOL_DISPLAY[GENERIC_CHAT],
            protocol_id=GENERIC_CHAT,
            confidence=best[GENERIC_CHAT],
            signals=[GENERIC_CHAT],
        ))

    if has_live_surface or has_route_confirmed:
        return _with_provider(ProtocolResolution(
            protocol=UNKNOWN_AI_PROTOCOL, protocol_id="", confidence="LOW",
        ))

    return _with_provider(
        ProtocolResolution(protocol="", protocol_id="", confidence="NONE"))
