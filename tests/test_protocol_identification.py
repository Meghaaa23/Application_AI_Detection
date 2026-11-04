"""Phase 1 — protocol / API-format identification (Ollama + resolver scaffold).

These tests cover the new, orthogonal protocol layer only. AI-presence behavior
(is_ai / score_evidence) is exercised by the existing suites and must stay
unchanged; a couple of guards here assert the protocol tag cannot influence it.
"""
import asyncio
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from multiagent.agents.ai_agents.ai_fingerprint import (
    AIFingerprintAgent,
    EvidenceItem,
)
from multiagent.agents.ai_agents.ai_evidence_scoring import score_evidence
from multiagent.agents.ai_agents.ai_protocol_signatures import (
    ANTHROPIC,
    BEDROCK,
    COHERE,
    GEMINI,
    GENERIC_CHAT,
    HF_TGI,
    OLLAMA,
    OPENAI,
    UNKNOWN_AI_PROTOCOL,
    detect_protocol_from_headers,
    detect_protocol_from_json,
    detect_provider,
    detect_provider_from_headers,
    detect_provider_from_host,
    extract_generated_text,
    is_bedrock_endpoint,
    is_gemini_endpoint,
    resolve_protocol,
)
from multiagent.agents.ai_agents.ai_fingerprint import (
    _is_inference_path,
    _looks_ai_api_json,
)


# --------------------------------------------------------------------------- #
# Ollama response samples
# --------------------------------------------------------------------------- #
OLLAMA_GENERATE = {
    "model": "llama3",
    "created_at": "2026-10-30T00:00:00Z",
    "response": "The ocean is vast and deep.",
    "done": True,
    "done_reason": "stop",
    "context": [1, 2, 3],
    "total_duration": 123456,
    "eval_count": 7,
}
OLLAMA_CHAT = {
    "model": "llama3",
    "created_at": "2026-10-30T00:00:00Z",
    "message": {"role": "assistant", "content": "Hello, how can I help?"},
    "done": True,
    "eval_count": 5,
    "total_duration": 98765,
}
OLLAMA_TAGS = {
    "models": [
        {
            "name": "llama3:latest",
            "model": "llama3:latest",
            "modified_at": "2026-10-30T00:00:00Z",
            "size": 4661224576,
            "digest": "abc123",
        }
    ]
}


# --------------------------------------------------------------------------- #
# detect_protocol_from_json — Ollama positives
# --------------------------------------------------------------------------- #
def test_detect_ollama_generate():
    det = detect_protocol_from_json(OLLAMA_GENERATE)
    assert det is not None and det[0] == OLLAMA


def test_detect_ollama_chat():
    det = detect_protocol_from_json(OLLAMA_CHAT)
    assert det is not None and det[0] == OLLAMA


def test_detect_ollama_tags():
    det = detect_protocol_from_json(OLLAMA_TAGS)
    assert det is not None and det[0] == OLLAMA


def test_detect_ollama_accepts_json_string():
    det = detect_protocol_from_json(json.dumps(OLLAMA_GENERATE))
    assert det is not None and det[0] == OLLAMA


# --------------------------------------------------------------------------- #
# detect_protocol_from_json — OpenAI stays OpenAI-compatible
# --------------------------------------------------------------------------- #
def test_detect_openai_chat_completion():
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }
    det = detect_protocol_from_json(body)
    assert det is not None and det[0] == OPENAI


def test_detect_openai_models_list():
    body = {"object": "list", "data": [{"id": "gpt-3.5-turbo", "object": "model"}]}
    det = detect_protocol_from_json(body)
    assert det is not None and det[0] == OPENAI


# --------------------------------------------------------------------------- #
# detect_protocol_from_json — conservative negatives (must NOT be Ollama)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("body", [
    {"model": "llama3"},                       # bare model
    {"response": "hello"},                     # bare response
    {"done": True},                            # bare done
    {"model": "m", "done": True},              # model + done, no response/message
    {"response": "x", "done": True},           # response + done, no model
    {"id": 1, "name": "row"},                  # generic REST record
    {"message": "hello"},                      # plain string message
    {"items": [1, 2, 3], "status": "ok"},      # ordinary list payload
])
def test_non_ollama_bodies_do_not_claim_ollama(body):
    det = detect_protocol_from_json(body)
    assert det is None or det[0] != OLLAMA


def test_generic_rest_json_is_not_a_protocol():
    assert detect_protocol_from_json({"id": 1, "name": "row"}) is None
    assert detect_protocol_from_json({"status": "ok", "done": True}) is None


def test_generic_ai_shape_is_generic_not_vendor():
    # model + usage tokens but no choices/object => AI-shaped but not a vendor.
    det = detect_protocol_from_json(
        {"model": "gpt-4", "usage": {"total_tokens": 10}})
    assert det is not None and det[0] == GENERIC_CHAT


# --------------------------------------------------------------------------- #
# extract_generated_text — across formats
# --------------------------------------------------------------------------- #
def test_extract_ollama_generate_text():
    assert extract_generated_text(OLLAMA_GENERATE) == "The ocean is vast and deep."


def test_extract_ollama_chat_message_content():
    assert extract_generated_text(OLLAMA_CHAT) == "Hello, how can I help?"


def test_extract_openai_choice_content():
    body = {"choices": [{"message": {"content": "Blue water everywhere."}}]}
    assert extract_generated_text(body) == "Blue water everywhere."


def test_extract_handles_json_string_and_missing():
    assert extract_generated_text(json.dumps(OLLAMA_GENERATE)) == \
        "The ocean is vast and deep."
    assert extract_generated_text("not json") == ""
    assert extract_generated_text({"nothing": "useful"}) == ""


# --------------------------------------------------------------------------- #
# resolve_protocol — aggregation rules
# --------------------------------------------------------------------------- #
def _ev(key, protocol=""):
    return EvidenceItem.make(key, f"{key} detail", "probe", protocol=protocol)


def test_resolve_ollama_from_live_surface():
    res = resolve_protocol(
        [_ev("chat_endpoint_200", OLLAMA)], has_live_surface=True)
    assert res.protocol == "Ollama"
    assert res.protocol_id == OLLAMA
    assert res.confidence == "HIGH"
    assert res.provider == ""


def test_resolve_openai_from_error_route_is_medium():
    res = resolve_protocol(
        [_ev("error_openai_format", OPENAI)], has_route_confirmed=True)
    assert res.protocol == "OpenAI-compatible"
    assert res.confidence == "MEDIUM"


def test_resolve_generic_chat():
    res = resolve_protocol(
        [_ev("ml_endpoint_200", GENERIC_CHAT)], has_live_surface=True)
    assert res.protocol == "Generic Chat API"


def test_resolve_prefers_specific_over_generic():
    res = resolve_protocol(
        [_ev("ml_endpoint_200", GENERIC_CHAT), _ev("chat_endpoint_200", OLLAMA)],
        has_live_surface=True,
    )
    assert res.protocol == "Ollama"


def test_resolve_conflict_is_deterministic():
    # Ollama outranks OpenAI at equal confidence (specific-priority order).
    res = resolve_protocol(
        [_ev("chat_endpoint_200", OPENAI), _ev("chat_endpoint_200", OLLAMA)],
        has_live_surface=True,
    )
    assert res.protocol == "Ollama"


def test_resolve_unknown_when_ai_surface_but_no_protocol_tag():
    res = resolve_protocol([_ev("ml_endpoint_401_403")], has_route_confirmed=True)
    assert res.protocol == UNKNOWN_AI_PROTOCOL


def test_resolve_empty_when_no_ai():
    res = resolve_protocol([], has_live_surface=False, has_route_confirmed=False)
    assert res.protocol == ""
    assert res.confidence == "NONE"


# --------------------------------------------------------------------------- #
# Presence is independent of the protocol tag
# --------------------------------------------------------------------------- #
def test_protocol_tag_does_not_change_presence():
    tagged = [_ev("chat_endpoint_200", OLLAMA), _ev("page_title_ai")]
    untagged = [_ev("chat_endpoint_200"), _ev("page_title_ai")]
    s1 = score_evidence([e.key for e in tagged])
    s2 = score_evidence([e.key for e in untagged])
    assert s1.detection == s2.detection
    assert s1.is_ai == s2.is_ai is True


# --------------------------------------------------------------------------- #
# _evaluate_probe_response tags protocol on the emitted evidence
# --------------------------------------------------------------------------- #
def test_probe_response_tags_ollama_generate():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "http://x/api/generate", "/api/generate", "chat",
        200, {"Content-Type": "application/json"},
        json.dumps(OLLAMA_GENERATE), [],
    )
    assert any(e.protocol == OLLAMA for e in r["evidence"])
    res = resolve_protocol(r["evidence"], has_live_surface=True)
    assert res.protocol == "Ollama"


def test_probe_response_tags_ollama_tags_listing():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "http://x/api/tags", "/api/tags", "schema",
        200, {"Content-Type": "application/json"},
        json.dumps(OLLAMA_TAGS), [],
    )
    assert any(e.protocol == OLLAMA for e in r["evidence"])


def test_probe_response_openai_passthrough_still_openai():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    body = ('{"id":"chatcmpl-1","object":"chat.completion",'
            '"choices":[{"message":{"content":"ok"}}]}')
    r = agent._evaluate_probe_response(
        "http://x/v1/chat/completions", "/v1/chat/completions", "chat",
        200, {"Content-Type": "application/json"}, body, [],
    )
    res = resolve_protocol(r["evidence"], has_live_surface=True)
    assert res.protocol == "OpenAI-compatible"


# --------------------------------------------------------------------------- #
# Full fingerprint against a fake Ollama server
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_server(handler_class, port: int):
    srv = HTTPServer(("127.0.0.1", port), handler_class)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


class OllamaHandler(BaseHTTPRequestHandler):
    """Minimal Ollama native API: /api/tags (GET), /api/chat + /api/generate (POST)."""

    def _json(self, status: int, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html(self, status=200):
        data = b"<!DOCTYPE html><html><head><title>Ollama</title></head><body>ok</body></html>"
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/tags":
            return self._json(200, OLLAMA_TAGS)
        if path in ("/", "/index.html"):
            return self._html(200)
        # chat/generate are POST-only; make GET retry as POST.
        return self._html(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        path = self.path.split("?")[0]
        if path == "/api/chat":
            return self._json(200, OLLAMA_CHAT)
        if path == "/api/generate":
            return self._json(200, OLLAMA_GENERATE)
        return self._html(404)

    def log_message(self, *args):
        pass


# =========================================================================== #
# Phase 2 — Anthropic Messages
# =========================================================================== #
ANTHROPIC_RESPONSE = {
    "id": "msg_01ABC",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello! How can I help?"}],
    "model": "claude-3-haiku-20240307",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 20},
}
ANTHROPIC_ERROR = {
    "type": "error",
    "error": {"type": "invalid_request_error", "message": "max_tokens: field required"},
}


def test_detect_anthropic_messages_response():
    det = detect_protocol_from_json(ANTHROPIC_RESPONSE)
    assert det is not None and det[0] == ANTHROPIC
    assert det[1] == "HIGH"


def test_detect_anthropic_accepts_json_string():
    det = detect_protocol_from_json(json.dumps(ANTHROPIC_RESPONSE))
    assert det is not None and det[0] == ANTHROPIC


def test_extract_anthropic_content_blocks():
    assert extract_generated_text(ANTHROPIC_RESPONSE) == "Hello! How can I help?"
    multi = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "..."},
            {"type": "text", "text": "Part one. "},
            {"type": "text", "text": "Part two."},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    assert extract_generated_text(multi) == "Part one. Part two."


def test_anthropic_headers_detected():
    assert detect_protocol_from_headers(
        {"anthropic-ratelimit-requests-limit": "50"})[0] == ANTHROPIC
    assert detect_protocol_from_headers(
        {"Anthropic-Version": "2023-06-01"})[0] == ANTHROPIC
    assert detect_protocol_from_headers({"content-type": "application/json"}) is None
    assert detect_protocol_from_headers({}) is None


def test_anthropic_error_body_classified_as_anthropic_not_openai():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "http://x/v1/messages", "/v1/messages", "chat",
        400, {"Content-Type": "application/json"},
        json.dumps(ANTHROPIC_ERROR), [],
    )
    keys = {e.key for e in r["evidence"]}
    assert "error_anthropic_format" in keys
    assert "error_openai_format" not in keys
    res = resolve_protocol(r["evidence"], has_route_confirmed=True)
    assert res.protocol == "Anthropic Messages"
    assert res.confidence == "MEDIUM"


def test_anthropic_live_response_resolves_high():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "http://x/v1/messages", "/v1/messages", "chat",
        200, {"Content-Type": "application/json"},
        json.dumps(ANTHROPIC_RESPONSE), [],
    )
    assert any(e.protocol == ANTHROPIC for e in r["evidence"])
    res = resolve_protocol(r["evidence"], has_live_surface=True)
    assert res.protocol == "Anthropic Messages"
    assert res.confidence == "HIGH"


def test_anthropic_header_tags_auth_gated_route():
    # A 401 on an inference route carrying Anthropic headers is route-confirmed
    # and tagged Anthropic — but presence stays False (no model output seen).
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    auth = []
    r = agent._evaluate_probe_response(
        "http://x/v1/messages", "/v1/messages", "chat",
        401,
        {"Content-Type": "application/json",
         "anthropic-ratelimit-tokens-limit": "100000"},
        '{"type":"error","error":{"type":"authentication_error","message":"x"}}',
        auth,
    )
    assert any(
        e.key == "ml_endpoint_401_403" and e.protocol == ANTHROPIC
        for e in r["evidence"]
    )
    res = resolve_protocol(r["evidence"], has_route_confirmed=True)
    assert res.protocol == "Anthropic Messages"
    s = score_evidence([e.key for e in r["evidence"]])
    assert s.is_ai is False  # protocol tag never promotes presence


@pytest.mark.parametrize("body", [
    {"messages": [{"role": "user", "content": "hi"}], "model": "claude-3"},  # request
    {"usage": {"input_tokens": 10, "output_tokens": 20}},                   # lone usage
    {"model": "claude-3-opus"},                                             # lone model
    {"content": "just a string"},                                          # not block list
    {"content": [{"foo": "bar"}]},                                         # blocks w/o type
    {"id": 1, "name": "row"},                                              # generic REST
])
def test_non_anthropic_bodies_do_not_claim_anthropic(body):
    det = detect_protocol_from_json(body)
    assert det is None or det[0] != ANTHROPIC


def test_anthropic_requires_multiple_markers():
    # content blocks + a single weak marker is not enough.
    weak = {"content": [{"type": "text", "text": "hi"}]}
    det = detect_protocol_from_json(weak)
    assert det is None or det[0] != ANTHROPIC


def test_ollama_and_openai_still_detected_after_anthropic_added():
    assert detect_protocol_from_json(OLLAMA_GENERATE)[0] == OLLAMA
    assert detect_protocol_from_json(OLLAMA_CHAT)[0] == OLLAMA
    openai_body = {"object": "chat.completion",
                   "choices": [{"message": {"content": "ok"}}]}
    assert detect_protocol_from_json(openai_body)[0] == OPENAI


def test_anthropic_does_not_override_stronger_openai():
    # HIGH OpenAI (live) must win over MEDIUM Anthropic (route/error).
    res = resolve_protocol(
        [_ev("chat_endpoint_200", OPENAI), _ev("error_anthropic_format", ANTHROPIC)],
        has_live_surface=True,
    )
    assert res.protocol == "OpenAI-compatible"


# =========================================================================== #
# Phase 3 — Google Gemini
# =========================================================================== #
GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "Hello! How can I help?"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 20,
        "totalTokenCount": 30,
    },
    "modelVersion": "gemini-1.5-flash",
}
GEMINI_MULTI = {
    "candidates": [
        {"content": {"role": "model", "parts": [
            {"text": "Part one. "}, {"text": "Part two."}]},
         "finishReason": "STOP"},
        {"content": {"role": "model", "parts": [{"text": "Alternative."}]},
         "finishReason": "STOP"},
    ],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5,
                      "totalTokenCount": 8},
}
GEMINI_ERROR = {
    "error": {
        "code": 400,
        "message": "API key not valid.",
        "status": "INVALID_ARGUMENT",
        "details": [{
            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
            "reason": "API_KEY_INVALID",
            "metadata": {"service": "generativelanguage.googleapis.com"},
        }],
    }
}
GENERIC_GOOGLE_ERROR = {
    "error": {"code": 400, "message": "Invalid value.", "status": "INVALID_ARGUMENT"}
}


def test_detect_gemini_response():
    det = detect_protocol_from_json(GEMINI_RESPONSE)
    assert det is not None and det[0] == GEMINI
    assert det[1] == "HIGH"


def test_detect_gemini_multiple_candidates():
    det = detect_protocol_from_json(GEMINI_MULTI)
    assert det is not None and det[0] == GEMINI


def test_extract_gemini_single_and_multiparts():
    assert extract_generated_text(GEMINI_RESPONSE) == "Hello! How can I help?"
    # multiple parts in the first candidate are joined; multi-candidate is safe.
    assert extract_generated_text(GEMINI_MULTI) == "Part one. Part two."


def test_extract_gemini_accepts_json_string():
    assert extract_generated_text(json.dumps(GEMINI_RESPONSE)) == \
        "Hello! How can I help?"


def test_gemini_endpoint_recognition():
    assert is_gemini_endpoint("/v1beta/models/gemini-pro:generateContent")
    assert is_gemini_endpoint("/v1/models/gemini-1.5-flash:streamGenerateContent")
    assert not is_gemini_endpoint("/v1/chat/completions")
    assert not is_gemini_endpoint("/api/generate")
    # wired into inference-path discovery like every other vendor's route
    assert _is_inference_path("/v1beta/models/gemini-pro:generateContent")
    assert not _is_inference_path("/models/catalog")


def test_gemini_request_shape_is_not_a_response_claim():
    # A generateContent *request* has contents/parts but no candidates — it must
    # not be classified as a Gemini response.
    req = {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
           "generationConfig": {"temperature": 0.7}}
    det = detect_protocol_from_json(req)
    assert det is None or det[0] != GEMINI


def test_gemini_error_classified_via_generativelanguage_token():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "http://x/v1beta/models/gemini-pro:generateContent",
        "/v1beta/models/gemini-pro:generateContent", "chat",
        400, {"Content-Type": "application/json"},
        json.dumps(GEMINI_ERROR), [],
    )
    keys = {e.key for e in r["evidence"]}
    assert "error_gemini_format" in keys
    res = resolve_protocol(r["evidence"], has_route_confirmed=True)
    assert res.protocol == "Google Gemini"


def test_generic_google_error_is_not_gemini():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "http://x/v1/data:fetch", "/v1/data:fetch", "schema",
        400, {"Content-Type": "application/json"},
        json.dumps(GENERIC_GOOGLE_ERROR), [],
    )
    keys = {e.key for e in r["evidence"]}
    assert "error_gemini_format" not in keys
    res = resolve_protocol(r["evidence"])
    assert res.protocol != "Google Gemini"


def test_gemini_live_200_is_live_surface_and_tagged():
    # A genuine Gemini 200 is now recognised as an AI-shaped live surface
    # (chat_endpoint_200) AND carries the Gemini protocol tag.
    assert _looks_ai_api_json(json.dumps(GEMINI_RESPONSE)) is True
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "http://x/v1beta/models/gemini-pro:generateContent",
        "/v1beta/models/gemini-pro:generateContent", "chat",
        200, {"Content-Type": "application/json"},
        json.dumps(GEMINI_RESPONSE), [],
    )
    gem_live = [
        e for e in r["evidence"]
        if e.key == "chat_endpoint_200" and e.protocol == GEMINI
    ]
    assert gem_live, [(e.key, e.protocol) for e in r["evidence"]]
    # chat_endpoint_200 is live-surface -> presence confirms via existing rules.
    s = score_evidence([e.key for e in r["evidence"]])
    assert s.has_live_surface is True and s.is_ai is True
    res = resolve_protocol(r["evidence"], has_live_surface=True)
    assert res.protocol == "Google Gemini"
    assert res.confidence == "HIGH"


def test_generic_candidates_json_is_not_live_surface():
    # A generic "candidates" list (e.g. a recruiting API) is not AI-shaped.
    assert _looks_ai_api_json('{"candidates":[{"name":"Bob","score":9}]}') is False
    # candidates+parts but NO corroborating Gemini marker is also rejected.
    assert _looks_ai_api_json(
        '{"candidates":[{"content":{"parts":[{"text":"x"}]}}]}') is False


def test_generic_rest_json_is_not_live_surface():
    assert _looks_ai_api_json('{"id":1,"name":"row","items":[1,2,3]}') is False
    assert _looks_ai_api_json(
        '{"kind":"customsearch#search","items":[{"title":"x"}]}') is False


def test_gemini_protocol_evidence_alone_does_not_confirm_presence():
    # Route/error-tier Gemini evidence (no live surface) must not set is_ai.
    s = score_evidence(["error_gemini_format", "page_title_ai"])
    assert s.is_ai is False
    assert s.has_live_surface is False
    res = resolve_protocol(
        [_ev("error_gemini_format", GEMINI)], has_route_confirmed=True)
    assert res.protocol == "Google Gemini"  # labelled, but presence stays False


def test_existing_protocols_still_ai_shaped_after_gemini_presence_change():
    assert _looks_ai_api_json(
        '{"object":"chat.completion","choices":[{"message":{"content":"ok"}}]}')
    assert _looks_ai_api_json(json.dumps(ANTHROPIC_RESPONSE))
    assert _looks_ai_api_json(json.dumps(OLLAMA_GENERATE))
    assert _looks_ai_api_json(json.dumps(OLLAMA_TAGS))


@pytest.mark.parametrize("body", [
    {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},  # request shape
    {"candidates": [{"name": "Bob"}]},                            # generic candidates
    {"candidates": [{"content": {"parts": [{"text": "x"}]}}]},    # parts w/o markers
    {"kind": "customsearch#search", "items": [{"title": "x"}]},   # generic Google REST
    {"id": 1, "name": "row"},
])
def test_non_gemini_bodies_do_not_claim_gemini(body):
    det = detect_protocol_from_json(body)
    assert det is None or det[0] != GEMINI


def test_gemini_conflict_resolution():
    # HIGH OpenAI (live) beats MEDIUM Gemini (error/route).
    res_weak = resolve_protocol(
        [_ev("error_gemini_format", GEMINI), _ev("chat_endpoint_200", OPENAI)],
        has_live_surface=True,
    )
    assert res_weak.protocol == "OpenAI-compatible"
    # Equal HIGH: Gemini outranks OpenAI by specific-priority order.
    res_tie = resolve_protocol(
        [_ev("chat_endpoint_200", GEMINI), _ev("chat_endpoint_200", OPENAI)],
        has_live_surface=True,
    )
    assert res_tie.protocol == "Google Gemini"
    # Strong Gemini beats generic chat.
    res_generic = resolve_protocol(
        [_ev("ml_endpoint_200", GENERIC_CHAT), _ev("chat_endpoint_200", GEMINI)],
        has_live_surface=True,
    )
    assert res_generic.protocol == "Google Gemini"


def test_other_protocols_still_detected_after_gemini_added():
    assert detect_protocol_from_json(OLLAMA_GENERATE)[0] == OLLAMA
    assert detect_protocol_from_json(ANTHROPIC_RESPONSE)[0] == ANTHROPIC
    openai_body = {"object": "chat.completion",
                   "choices": [{"message": {"content": "ok"}}]}
    assert detect_protocol_from_json(openai_body)[0] == OPENAI
    # extraction for each still works
    assert extract_generated_text(ANTHROPIC_RESPONSE) == "Hello! How can I help?"
    assert extract_generated_text(OLLAMA_CHAT) == "Hello, how can I help?"


def test_gemini_protocol_tag_does_not_change_presence():
    tagged = score_evidence([e.key for e in [
        _ev("chat_endpoint_200", GEMINI), _ev("page_title_ai")]])
    untagged = score_evidence([e.key for e in [
        _ev("chat_endpoint_200"), _ev("page_title_ai")]])
    assert tagged.detection == untagged.detection
    assert tagged.is_ai is untagged.is_ai is True


# =========================================================================== #
# Phase 4 — Cohere
# =========================================================================== #
COHERE_CHAT = {
    "response_id": "r-1",
    "text": "Hello! How can I help?",
    "generation_id": "gen-1",
    "chat_history": [
        {"role": "USER", "message": "Hi"},
        {"role": "CHATBOT", "message": "Hello! How can I help?"},
    ],
    "finish_reason": "COMPLETE",
    "meta": {
        "api_version": {"version": "1"},
        "billed_units": {"input_tokens": 10, "output_tokens": 20},
        "tokens": {"input_tokens": 10, "output_tokens": 20},
    },
}
COHERE_GENERATE = {
    "id": "x",
    "generations": [{"id": "g1", "text": "Blue ocean everywhere."}],
    "prompt": "Describe the ocean",
    "meta": {"api_version": {"version": "1"}},
}
COHERE_REQUEST = {
    "model": "command-r",
    "message": "Hello",
    "chat_history": [{"role": "USER", "message": "Hi"}],
    "preamble": "You are helpful.",
    "connectors": [{"id": "web-search"}],
}
COHERE_ERROR = {"message": "invalid request: model 'bogus' not found"}


def test_detect_cohere_chat():
    det = detect_protocol_from_json(COHERE_CHAT)
    assert det is not None and det[0] == COHERE
    assert det[1] == "HIGH"


def test_detect_cohere_generate():
    det = detect_protocol_from_json(COHERE_GENERATE)
    assert det is not None and det[0] == COHERE


def test_detect_cohere_accepts_json_string():
    det = detect_protocol_from_json(json.dumps(COHERE_CHAT))
    assert det is not None and det[0] == COHERE


def test_cohere_request_shape_is_not_a_response_claim():
    # A Cohere *request* (message/chat_history/model/preamble/connectors) has no
    # generation_id/response_id/generations/text → not a response classification.
    det = detect_protocol_from_json(COHERE_REQUEST)
    assert det is None or det[0] != COHERE


def test_extract_cohere_chat_and_generate():
    assert extract_generated_text(COHERE_CHAT) == "Hello! How can I help?"
    assert extract_generated_text(COHERE_GENERATE) == "Blue ocean everywhere."


def test_cohere_error_not_distinctive_no_claim():
    # Cohere's error body is a bare {"message": ...} — indistinguishable from a
    # generic API error, so it must NOT yield a Cohere classification.
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "http://x/v1/chat", "/v1/chat", "chat",
        400, {"Content-Type": "application/json"},
        json.dumps(COHERE_ERROR), [],
    )
    keys = {e.key for e in r["evidence"]}
    assert "error_cohere_format" not in keys
    assert not any(e.protocol == COHERE for e in r["evidence"])
    res = resolve_protocol(r["evidence"], has_route_confirmed=True)
    assert res.protocol != "Cohere"


@pytest.mark.parametrize("body", [
    {"text": "Hello"},                          # bare generic text
    {"message": "hi", "model": "command-r"},    # generic message/model
    {"id": 1, "name": "row"},                   # generic REST
    {"text": "hi", "finish_reason": "COMPLETE"},  # text + 1 marker only
])
def test_non_cohere_bodies_do_not_claim_cohere(body):
    det = detect_protocol_from_json(body)
    assert det is None or det[0] != COHERE


def test_cohere_live_200_is_live_surface_and_tagged():
    assert _looks_ai_api_json(json.dumps(COHERE_CHAT)) is True
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "http://x/v1/chat", "/v1/chat", "chat",
        200, {"Content-Type": "application/json"},
        json.dumps(COHERE_CHAT), [],
    )
    gem_live = [
        e for e in r["evidence"]
        if e.key == "chat_endpoint_200" and e.protocol == COHERE
    ]
    assert gem_live, [(e.key, e.protocol) for e in r["evidence"]]
    s = score_evidence([e.key for e in r["evidence"]])
    assert s.has_live_surface is True and s.is_ai is True
    res = resolve_protocol(r["evidence"], has_live_surface=True)
    assert res.protocol == "Cohere"
    assert res.confidence == "HIGH"


def test_cohere_generate_200_is_live_surface():
    assert _looks_ai_api_json(json.dumps(COHERE_GENERATE)) is True


def test_cohere_protocol_evidence_alone_does_not_confirm_presence():
    # Route-tier Cohere evidence (no live surface) must not set is_ai.
    s = score_evidence(["ml_endpoint_401_403", "page_title_ai"])
    assert s.is_ai is False
    assert s.has_live_surface is False
    res = resolve_protocol(
        [_ev("ml_endpoint_401_403", COHERE)], has_route_confirmed=True)
    assert res.protocol == "Cohere"  # labelled, presence still False


def test_cohere_conflict_resolution():
    # HIGH OpenAI (live) beats route-tier (MEDIUM) Cohere.
    res_weak = resolve_protocol(
        [_ev("ml_endpoint_401_403", COHERE), _ev("chat_endpoint_200", OPENAI)],
        has_live_surface=True,
    )
    assert res_weak.protocol == "OpenAI-compatible"
    # Equal HIGH: Cohere outranks OpenAI by specific-priority order.
    res_tie = resolve_protocol(
        [_ev("chat_endpoint_200", COHERE), _ev("chat_endpoint_200", OPENAI)],
        has_live_surface=True,
    )
    assert res_tie.protocol == "Cohere"
    # Strong Cohere beats generic chat.
    res_generic = resolve_protocol(
        [_ev("ml_endpoint_200", GENERIC_CHAT), _ev("chat_endpoint_200", COHERE)],
        has_live_surface=True,
    )
    assert res_generic.protocol == "Cohere"


def test_other_protocols_regress_after_cohere_added():
    assert detect_protocol_from_json(GEMINI_RESPONSE)[0] == GEMINI
    assert detect_protocol_from_json(ANTHROPIC_RESPONSE)[0] == ANTHROPIC
    assert detect_protocol_from_json(OLLAMA_GENERATE)[0] == OLLAMA
    openai_body = {"object": "chat.completion",
                   "choices": [{"message": {"content": "ok"}}]}
    assert detect_protocol_from_json(openai_body)[0] == OPENAI
    # extraction still intact per protocol
    assert extract_generated_text(GEMINI_RESPONSE) == "Hello! How can I help?"
    assert extract_generated_text(ANTHROPIC_RESPONSE) == "Hello! How can I help?"
    assert extract_generated_text(OLLAMA_CHAT) == "Hello, how can I help?"


# =========================================================================== #
# Phase 5 — Hugging Face TGI
# =========================================================================== #
TGI_GENERATE = {
    "generated_text": "Hello! How can I help?",
    "details": {
        "finish_reason": "eos_token",
        "generated_tokens": 8,
        "prefill": [],
        "tokens": [],
        "seed": None,
    },
}
TGI_LIST = [{"generated_text": "Hello! How can I help?"}]
TGI_BATCH = [{"generated_text": "First."}, {"generated_text": "Second."}]
TGI_ERROR = {
    "error": "Input validation error: `inputs` must be non-empty",
    "error_type": "validation",
}
HF_OPENAI_COMPAT = {
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
}
BARE_GENERATED_TEXT = {"generated_text": "Hello"}


def test_detect_tgi_generate_object():
    det = detect_protocol_from_json(TGI_GENERATE)
    assert det is not None and det[0] == HF_TGI
    assert det[1] == "HIGH"


def test_detect_tgi_list_form():
    det = detect_protocol_from_json(TGI_LIST)
    assert det is not None and det[0] == HF_TGI
    det_str = detect_protocol_from_json(json.dumps(TGI_LIST))
    assert det_str is not None and det_str[0] == HF_TGI


def test_detect_tgi_batch():
    det = detect_protocol_from_json(TGI_BATCH)
    assert det is not None and det[0] == HF_TGI


def test_tgi_request_shape_is_not_a_response_claim():
    req = {"inputs": "Hello", "parameters": {"max_new_tokens": 16}, "stream": False}
    det = detect_protocol_from_json(req)
    assert det is None or det[0] != HF_TGI


def test_extract_tgi_object_list_and_batch():
    assert extract_generated_text(TGI_GENERATE) == "Hello! How can I help?"
    assert extract_generated_text(TGI_LIST) == "Hello! How can I help?"
    assert extract_generated_text(json.dumps(TGI_LIST)) == "Hello! How can I help?"
    assert extract_generated_text(TGI_BATCH) == "First."


def test_bare_generated_text_is_generic_not_tgi():
    det = detect_protocol_from_json(BARE_GENERATED_TEXT)
    assert det is not None and det[0] == GENERIC_CHAT  # ambiguous -> generic


def test_generic_rest_json_is_not_tgi():
    assert detect_protocol_from_json({"id": 1, "name": "row"}) is None
    assert detect_protocol_from_json([{"label": "POSITIVE", "score": 0.9}]) is None


def test_openai_compatible_on_hf_is_openai_not_tgi():
    det = detect_protocol_from_json(HF_OPENAI_COMPAT)
    assert det is not None and det[0] == OPENAI


def test_tgi_headers_detected():
    assert detect_protocol_from_headers({"x-prompt-tokens": "10"})[0] == HF_TGI
    assert detect_protocol_from_headers({"X-Compute-Type": "gpu"})[0] == HF_TGI
    assert detect_protocol_from_headers({"content-type": "application/json"}) is None


def test_tgi_error_classified_via_error_type():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "http://x/generate", "/generate", "chat",
        422, {"Content-Type": "application/json"},
        json.dumps(TGI_ERROR), [],
    )
    keys = {e.key for e in r["evidence"]}
    assert "error_tgi_format" in keys
    res = resolve_protocol(r["evidence"], has_route_confirmed=True)
    assert res.protocol == "Hugging Face TGI"


def test_tgi_live_200_object_and_list_are_live_surface_tagged():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    for body in (TGI_GENERATE, TGI_LIST):
        payload = json.dumps(body)
        assert _looks_ai_api_json(payload) is True
        r = agent._evaluate_probe_response(
            "http://x/generate", "/generate", "chat",
            200, {"Content-Type": "application/json"}, payload, [],
        )
        tgi_live = [
            e for e in r["evidence"]
            if e.key == "chat_endpoint_200" and e.protocol == HF_TGI
        ]
        assert tgi_live, [(e.key, e.protocol) for e in r["evidence"]]
        s = score_evidence([e.key for e in r["evidence"]])
        assert s.has_live_surface is True and s.is_ai is True
        res = resolve_protocol(r["evidence"], has_live_surface=True)
        assert res.protocol == "Hugging Face TGI"


def test_tgi_protocol_evidence_alone_does_not_confirm_presence():
    s = score_evidence(["error_tgi_format", "page_title_ai"])
    assert s.is_ai is False
    assert s.has_live_surface is False
    res = resolve_protocol(
        [_ev("error_tgi_format", HF_TGI)], has_route_confirmed=True)
    assert res.protocol == "Hugging Face TGI"  # labelled, presence stays False


def test_tgi_conflict_resolution():
    # HIGH OpenAI (live) beats route-tier (MEDIUM) TGI.
    res_weak = resolve_protocol(
        [_ev("error_tgi_format", HF_TGI), _ev("chat_endpoint_200", OPENAI)],
        has_live_surface=True,
    )
    assert res_weak.protocol == "OpenAI-compatible"
    # Equal HIGH: TGI outranks OpenAI by specific-priority order.
    res_tie = resolve_protocol(
        [_ev("chat_endpoint_200", HF_TGI), _ev("chat_endpoint_200", OPENAI)],
        has_live_surface=True,
    )
    assert res_tie.protocol == "Hugging Face TGI"
    # Strong TGI beats generic chat.
    res_generic = resolve_protocol(
        [_ev("ml_endpoint_200", GENERIC_CHAT), _ev("chat_endpoint_200", HF_TGI)],
        has_live_surface=True,
    )
    assert res_generic.protocol == "Hugging Face TGI"


def test_other_protocols_regress_after_tgi_added():
    assert detect_protocol_from_json(COHERE_CHAT)[0] == COHERE
    assert detect_protocol_from_json(GEMINI_RESPONSE)[0] == GEMINI
    assert detect_protocol_from_json(ANTHROPIC_RESPONSE)[0] == ANTHROPIC
    assert detect_protocol_from_json(OLLAMA_GENERATE)[0] == OLLAMA
    assert detect_protocol_from_json(
        {"object": "chat.completion", "choices": [{"message": {"content": "x"}}]}
    )[0] == OPENAI
    # extraction intact
    assert extract_generated_text(COHERE_GENERATE) == "Blue ocean everywhere."
    assert extract_generated_text(GEMINI_RESPONSE) == "Hello! How can I help?"


# =========================================================================== #
# Phase 6 — Azure OpenAI (provider axis)
# =========================================================================== #
AZURE_CHAT_URL = ("https://my-res.openai.azure.com/openai/deployments/"
                  "gpt4o/chat/completions?api-version=2024-06-01")
AZURE_EMB_URL = ("https://my-res.openai.azure.com/openai/deployments/"
                 "text-embedding-3-large/embeddings?api-version=2024-06-01")
AZURE_HEADERS = {
    "Content-Type": "application/json",
    "apim-request-id": "a1b2c3",
    "x-ms-region": "East US",
}
OPENAI_CHAT_BODY = {
    "id": "chatcmpl-1", "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
}
OPENAI_EMB_BODY = {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3, 0.4], "index": 0}],
    "model": "text-embedding-3-large",
}
# Azure content-filter error — OpenAI-shaped envelope (type/param/code) with an
# Azure Responsible-AI innererror + content_filter_result.
AZURE_ERROR = {
    "error": {
        "message": "The response was filtered due to Azure OpenAI content policy",
        "type": None, "param": "prompt", "code": "content_filter", "status": 400,
        "innererror": {
            "code": "ResponsibleAIPolicyViolation",
            "content_filter_result": {"hate": {"filtered": True, "severity": "high"}},
        },
    }
}
GENERIC_AZURE_HEADERS = {"x-ms-request-id": "r1", "x-ms-region": "West US"}


def test_azure_hostname_recognition():
    assert detect_provider_from_host(AZURE_CHAT_URL)[0] == "azure"
    assert detect_provider_from_host(
        "https://foundry.services.ai.azure.com/x")[0] == "azure"
    assert detect_provider_from_host(
        "https://api.openai.com/v1/chat/completions")[0] == "openai"
    assert detect_provider_from_host("https://example.com/v1/chat") is None
    # hostname is provider evidence only — not proof of a live AI endpoint.
    assert "azure" == detect_provider(url=AZURE_CHAT_URL)[0]


def test_azure_header_recognition():
    assert detect_provider_from_headers({"apim-request-id": "x"})[0] == "azure"
    assert detect_provider_from_headers({"azureml-model-session": "x"})[0] == "azure"
    assert detect_provider_from_headers({"content-type": "application/json"}) is None
    # generic Azure infra headers alone are NOT sufficient.
    assert detect_provider_from_headers(GENERIC_AZURE_HEADERS) is None


def test_azure_response_is_openai_protocol_unaffected():
    # Provider detection must not change protocol classification of the body.
    assert detect_protocol_from_json(OPENAI_CHAT_BODY)[0] == OPENAI


def test_azure_chat_endpoint_provider_and_protocol_separation():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        AZURE_CHAT_URL,
        "/openai/deployments/gpt4o/chat/completions", "chat",
        200, AZURE_HEADERS, json.dumps(OPENAI_CHAT_BODY), [],
    )
    live = [e for e in r["evidence"] if e.key == "chat_endpoint_200"]
    assert live and live[0].protocol == OPENAI and live[0].provider == "azure"
    res = resolve_protocol(r["evidence"], has_live_surface=True)
    assert res.protocol == "OpenAI-compatible"
    assert res.provider == "Azure OpenAI"
    assert res.provider_confidence == "HIGH"
    s = score_evidence([e.key for e in r["evidence"]])
    assert s.is_ai is True  # confirmed by the OpenAI live body, not by provider


def test_azure_embeddings_endpoint():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        AZURE_EMB_URL,
        "/openai/deployments/text-embedding-3-large/embeddings", "embedding",
        200, AZURE_HEADERS, json.dumps(OPENAI_EMB_BODY), [],
    )
    # Embeddings paths are inference paths, so the existing ev_key logic emits a
    # live-surface *_endpoint_200 key; what matters here is provider+protocol.
    live = [e for e in r["evidence"] if e.key.endswith("_endpoint_200")]
    assert live and live[0].protocol == OPENAI and live[0].provider == "azure"
    res = resolve_protocol(r["evidence"], has_live_surface=True)
    assert res.protocol == "OpenAI-compatible"
    assert res.provider == "Azure OpenAI"


def test_azure_error_handling_provider_with_openai_protocol():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        AZURE_CHAT_URL,
        "/openai/deployments/gpt4o/chat/completions", "chat",
        400, AZURE_HEADERS, json.dumps(AZURE_ERROR), [],
    )
    res = resolve_protocol(r["evidence"], has_route_confirmed=True)
    assert res.provider == "Azure OpenAI"
    # Error envelope is OpenAI-shaped (type/param/code) -> OpenAI-compatible.
    assert res.protocol == "OpenAI-compatible"


def test_azure_host_non_ai_response_is_not_confirmed():
    # An Azure host returning a non-AI body: provider recognisable from host,
    # but no AI evidence -> not confirmed, nothing tagged into the resolver.
    assert detect_provider_from_host(
        "https://my-res.openai.azure.com/health")[0] == "azure"
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "https://my-res.openai.azure.com/health", "/health", "schema",
        200, {"Content-Type": "application/json"}, '{"status":"ok"}', [],
    )
    assert r.get("endpoints", []) == []
    res = resolve_protocol(r["evidence"])
    assert res.provider == ""  # no AI evidence -> no provider claim in resolver


def test_generic_azure_headers_without_ai_evidence():
    # Generic Azure infra headers + no AI body -> no AI, no provider.
    assert detect_provider(headers=GENERIC_AZURE_HEADERS) is None
    assert score_evidence([]).is_ai is False


def test_public_openai_endpoint_provider_is_openai():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "https://api.openai.com/v1/chat/completions",
        "/v1/chat/completions", "chat",
        200, {"Content-Type": "application/json"},
        json.dumps(OPENAI_CHAT_BODY), [],
    )
    res = resolve_protocol(r["evidence"], has_live_surface=True)
    assert res.protocol == "OpenAI-compatible"
    assert res.provider == "OpenAI"


def test_generic_openai_compatible_provider_unknown():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "http://localhost:8000/v1/chat/completions",
        "/v1/chat/completions", "chat",
        200, {"Content-Type": "application/json"},
        json.dumps(OPENAI_CHAT_BODY), [],
    )
    res = resolve_protocol(r["evidence"], has_live_surface=True)
    assert res.protocol == "OpenAI-compatible"
    assert res.provider == ""  # unknown provider


def test_weak_azure_hint_does_not_override_protocol():
    # Azure provider rides along; a non-OpenAI protocol body still wins protocol.
    r = resolve_protocol(
        [EvidenceItem.make("chat_endpoint_200", "d", "probe",
                           protocol=OLLAMA, provider="azure")],
        has_live_surface=True,
    )
    assert r.protocol == "Ollama"
    assert r.provider == "Azure OpenAI"


def test_all_protocols_regress_after_azure_provider_added():
    assert detect_protocol_from_json(TGI_GENERATE)[0] == HF_TGI
    assert detect_protocol_from_json(COHERE_CHAT)[0] == COHERE
    assert detect_protocol_from_json(GEMINI_RESPONSE)[0] == GEMINI
    assert detect_protocol_from_json(ANTHROPIC_RESPONSE)[0] == ANTHROPIC
    assert detect_protocol_from_json(OLLAMA_GENERATE)[0] == OLLAMA
    assert detect_protocol_from_json(OPENAI_CHAT_BODY)[0] == OPENAI


# =========================================================================== #
# Phase 7 — AWS Bedrock (protocol AND provider)
# =========================================================================== #
BEDROCK_CONVERSE = {
    "output": {"message": {"role": "assistant",
                           "content": [{"text": "Hello!"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
    "metrics": {"latencyMs": 420},
}
BEDROCK_TITAN = {
    "inputTextTokenCount": 7,
    "results": [{"tokenCount": 9, "outputText": "Blue ocean.",
                 "completionReason": "FINISH"}],
}
BEDROCK_LLAMA = {
    "generation": "Hello from Llama.",
    "prompt_token_count": 5,
    "generation_token_count": 4,
    "stop_reason": "stop",
}
# Anthropic model invoked THROUGH Bedrock InvokeModel — body is Anthropic's.
BEDROCK_ANTHROPIC_BODY = ANTHROPIC_RESPONSE
BEDROCK_CONVERSE_REQUEST = {
    "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
    "inferenceConfig": {"maxTokens": 100},
}
BEDROCK_HOST_CHAT = ("https://bedrock-runtime.us-east-1.amazonaws.com/"
                     "model/anthropic.claude-3-sonnet-20240229-v1:0/converse")
BEDROCK_HOST_INVOKE = ("https://bedrock-runtime.us-east-1.amazonaws.com/"
                       "model/amazon.titan-text-express-v1/invoke")
BEDROCK_HEADERS = {
    "Content-Type": "application/json",
    "x-amzn-requestid": "req-1",
    "x-amzn-bedrock-input-token-count": "10",
    "x-amzn-bedrock-output-token-count": "20",
}
BEDROCK_VALIDATION_ERROR = {
    "message": "The provided model identifier is invalid.",
}
GENERIC_AWS_ERROR = {"message": "Access denied", "__type": "AccessDeniedException"}


def test_detect_bedrock_converse():
    det = detect_protocol_from_json(BEDROCK_CONVERSE)
    assert det is not None and det[0] == BEDROCK
    assert det[1] == "HIGH"


def test_detect_bedrock_invokemodel_titan_and_llama():
    assert detect_protocol_from_json(BEDROCK_TITAN)[0] == BEDROCK
    assert detect_protocol_from_json(BEDROCK_LLAMA)[0] == BEDROCK


def test_bedrock_converse_request_is_not_a_response_claim():
    det = detect_protocol_from_json(BEDROCK_CONVERSE_REQUEST)
    assert det is None or det[0] != BEDROCK


def test_extract_bedrock_converse_and_invokemodel():
    assert extract_generated_text(BEDROCK_CONVERSE) == "Hello!"
    assert extract_generated_text(BEDROCK_TITAN) == "Blue ocean."
    assert extract_generated_text(BEDROCK_LLAMA) == "Hello from Llama."
    # multiple content blocks join
    multi = {"output": {"message": {"role": "assistant", "content": [
        {"text": "A "}, {"text": "B"}]}}, "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3}}
    assert extract_generated_text(multi) == "A B"


def test_bedrock_invokemodel_with_anthropic_body_is_anthropic_protocol():
    # A Claude model invoked via Bedrock returns the Anthropic body; the protocol
    # is honestly Anthropic Messages (provider is identified separately by host).
    det = detect_protocol_from_json(BEDROCK_ANTHROPIC_BODY)
    assert det is not None and det[0] == ANTHROPIC


def test_bedrock_endpoint_recognition():
    assert is_bedrock_endpoint(
        "/model/anthropic.claude-3-sonnet-20240229-v1:0/converse")
    assert is_bedrock_endpoint("/model/amazon.titan-text-express-v1/invoke")
    assert is_bedrock_endpoint(
        "/model/meta.llama3-70b/invoke-with-response-stream")
    assert not is_bedrock_endpoint("/v1/chat/completions")
    assert not is_bedrock_endpoint("/model/catalog")
    assert _is_inference_path(
        "/model/anthropic.claude-3/converse")


def test_aws_provider_identification():
    assert detect_provider_from_host(BEDROCK_HOST_CHAT)[0] == "aws_bedrock"
    assert detect_provider_from_headers(
        {"x-amzn-bedrock-input-token-count": "10"})[0] == "aws_bedrock"
    # generic AWS request-id header alone is not sufficient
    assert detect_provider_from_headers({"x-amzn-requestid": "r1"}) is None
    # path-based provider signal
    assert detect_provider(
        url=BEDROCK_HOST_CHAT)[0] == "aws_bedrock"


def test_bedrock_provider_and_protocol_separation():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        BEDROCK_HOST_CHAT,
        "/model/anthropic.claude-3-sonnet-20240229-v1:0/converse", "chat",
        200, BEDROCK_HEADERS, json.dumps(BEDROCK_CONVERSE), [],
    )
    live = [e for e in r["evidence"] if e.key == "chat_endpoint_200"]
    assert live and live[0].protocol == BEDROCK and live[0].provider == "aws_bedrock"
    res = resolve_protocol(r["evidence"], has_live_surface=True)
    assert res.protocol == "AWS Bedrock"
    assert res.provider == "AWS Bedrock"   # same display, separate fields
    s = score_evidence([e.key for e in r["evidence"]])
    assert s.is_ai is True


def test_bedrock_specific_error_on_bedrock_path():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        BEDROCK_HOST_CHAT,
        "/model/anthropic.claude-3/converse", "chat",
        400,
        {"Content-Type": "application/json",
         "x-amzn-errortype": "ValidationException"},
        json.dumps(BEDROCK_VALIDATION_ERROR), [],
    )
    keys = {e.key for e in r["evidence"]}
    assert "error_bedrock_format" in keys
    res = resolve_protocol(r["evidence"], has_route_confirmed=True)
    assert res.protocol == "AWS Bedrock"
    assert res.provider == "AWS Bedrock"


def test_generic_aws_error_off_bedrock_path_is_not_bedrock():
    agent = AIFingerprintAgent(run_behavioral_probes=False)
    r = agent._evaluate_probe_response(
        "https://s3.us-east-1.amazonaws.com/bucket/key", "/bucket/key", "schema",
        403,
        {"Content-Type": "application/json",
         "x-amzn-errortype": "AccessDeniedException"},
        json.dumps(GENERIC_AWS_ERROR), [],
    )
    keys = {e.key for e in r["evidence"]}
    assert "error_bedrock_format" not in keys
    res = resolve_protocol(r["evidence"])
    assert res.protocol != "AWS Bedrock"
    assert res.provider != "AWS Bedrock"


def test_generic_aws_endpoint_negative():
    # An AWS host that is not Bedrock, non-AI body -> no Bedrock, not confirmed.
    assert detect_provider_from_host(
        "https://dynamodb.us-east-1.amazonaws.com/") is None
    assert detect_protocol_from_json({"TableNames": ["Users"]}) is None


def test_bedrock_live_200_converse_is_live_surface():
    assert _looks_ai_api_json(json.dumps(BEDROCK_CONVERSE)) is True
    assert _looks_ai_api_json(json.dumps(BEDROCK_TITAN)) is True
    assert _looks_ai_api_json(json.dumps(BEDROCK_LLAMA)) is True


def test_bedrock_protocol_evidence_alone_does_not_confirm_presence():
    s = score_evidence(["error_bedrock_format", "page_title_ai"])
    assert s.is_ai is False
    assert s.has_live_surface is False
    res = resolve_protocol(
        [_ev("error_bedrock_format", BEDROCK)], has_route_confirmed=True)
    assert res.protocol == "AWS Bedrock"


def test_bedrock_conflict_resolution():
    # HIGH OpenAI (live) beats route-tier Bedrock.
    res_weak = resolve_protocol(
        [_ev("error_bedrock_format", BEDROCK), _ev("chat_endpoint_200", OPENAI)],
        has_live_surface=True,
    )
    assert res_weak.protocol == "OpenAI-compatible"
    # Equal HIGH: Bedrock outranks OpenAI by specific-priority order.
    res_tie = resolve_protocol(
        [_ev("chat_endpoint_200", BEDROCK), _ev("chat_endpoint_200", OPENAI)],
        has_live_surface=True,
    )
    assert res_tie.protocol == "AWS Bedrock"


def test_all_protocols_regress_after_bedrock_added():
    assert detect_provider_from_host("https://my-res.openai.azure.com/x")[0] == "azure"
    assert detect_protocol_from_json(TGI_GENERATE)[0] == HF_TGI
    assert detect_protocol_from_json(COHERE_CHAT)[0] == COHERE
    assert detect_protocol_from_json(GEMINI_RESPONSE)[0] == GEMINI
    assert detect_protocol_from_json(ANTHROPIC_RESPONSE)[0] == ANTHROPIC
    assert detect_protocol_from_json(OLLAMA_GENERATE)[0] == OLLAMA
    assert detect_protocol_from_json(
        {"object": "chat.completion", "choices": [{"message": {"content": "x"}}]}
    )[0] == OPENAI


@pytest.mark.asyncio
async def test_fingerprint_ollama_server_reports_ollama_protocol():
    port = _free_port()
    srv = _run_server(OllamaHandler, port)
    try:
        agent = AIFingerprintAgent(timeout=3, run_behavioral_probes=False)
        base = f"http://127.0.0.1:{port}"
        result = await agent.fingerprint(
            base,
            discovered_endpoints=[
                {"url": f"{base}/api/chat"},
                {"url": f"{base}/api/generate"},
                {"url": f"{base}/api/tags"},
            ],
            verbose=False,
        )
        assert result["protocol"] == "Ollama"
        assert result["protocol_confidence"] == "HIGH"
        assert result["provider"] == ""
        # Presence still driven solely by live-surface evidence.
        assert result["has_live_surface"] is True
        assert result["is_ai_powered"] is True
    finally:
        srv.shutdown()
