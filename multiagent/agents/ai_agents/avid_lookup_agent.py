import re
import json
import asyncio
import logging
import urllib.request
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

AVID_BASE_URL   = "https://avidml.org"
AVID_DB_URL     = "https://avidml.org/database/"
AVID_API_URL    = "https://avidml.org/database/AVID"

LIBRARY_SEARCH_TERMS = {
    "pytorch":       ["pytorch", "torch"],
    "tensorflow":    ["tensorflow", "keras"],
    "hugging face":  ["huggingface", "transformers", "hugging"],
    "langchain":     ["langchain"],
    "llama":         ["llama", "llama.cpp", "llamaindex"],
    "openai":        ["openai"],
    "anthropic":     ["anthropic"],
    "onnx":          ["onnx"],
    "scikit-learn":  ["sklearn", "scikit"],
    "mlflow":        ["mlflow"],
    "bentoml":       ["bentoml"],
    "gradio":        ["gradio"],
    "streamlit":     ["streamlit"],
    "fastapi":       ["fastapi"],
    "pickle":        ["pickle", "picklescan", "deserialization"],
    "jupyter":       ["jupyter", "notebook"],
}

AVID_BACKDOOR_ENTRIES = [
    {
        "avid_id":     "AVID-2026-R0035",
        "cve_id":      "CVE-2025-1716",
        "title":       "picklescan Security Scanning Bypass via pip main",
        "description": "picklescan can be bypassed allowing malicious pickle files "
                       "to evade detection. Directly enables backdoored model delivery "
                       "via ML supply chain.",
        "severity":    "High",
        "cvss_score":  7.5,
        "source":      "AVID",
        "url":         "https://avidml.org/database/AVID-2026-R0035",
        "atlas_technique": "AML.T0018.002 - Embed Malware",
        "report_type": "Advisory",
        "ai_backdoor_relevant": True,
    },
    {
        "avid_id":     "AVID-2026-R0036",
        "cve_id":      "CVE-2025-1889",
        "title":       "picklescan Bypass via Non-Standard File Extensions",
        "description": "picklescan fails to detect malicious pickle content when "
                       "non-standard file extensions are used. Attackers can deliver "
                       "backdoored models that bypass security scanning.",
        "severity":    "High",
        "cvss_score":  7.5,
        "source":      "AVID",
        "url":         "https://avidml.org/database/AVID-2026-R0036",
        "atlas_technique": "AML.T0018.002 - Embed Malware",
        "report_type": "Advisory",
        "ai_backdoor_relevant": True,
    },
    {
        "avid_id":     "AVID-2026-R0037",
        "cve_id":      "CVE-2025-1944",
        "title":       "picklescan ZIP Archive Manipulation Attack",
        "description": "A ZIP archive manipulation attack causes picklescan to crash "
                       "without detecting malicious content, allowing backdoored model "
                       "files to pass through undetected.",
        "severity":    "High",
        "cvss_score":  7.2,
        "source":      "AVID",
        "url":         "https://avidml.org/database/AVID-2026-R0037",
        "atlas_technique": "AML.T0018.002 - Embed Malware",
        "report_type": "Advisory",
        "ai_backdoor_relevant": True,
    },
    {
        "avid_id":     "AVID-2026-R0034",
        "cve_id":      "CVE-2025-1550",
        "title":       "Arbitrary Code Execution via Crafted Keras Config",
        "description": "Maliciously crafted Keras model configuration files can trigger "
                       "arbitrary code execution when the model is loaded. Classic "
                       "backdoor delivery mechanism via model deserialization.",
        "severity":    "Critical",
        "cvss_score":  9.1,
        "source":      "AVID",
        "url":         "https://avidml.org/database/AVID-2026-R0034",
        "atlas_technique": "AML.T0018 - Backdoor ML Model",
        "report_type": "Advisory",
        "ai_backdoor_relevant": True,
    },
    {
        "avid_id":     "AVID-2026-R0023",
        "cve_id":      "CVE-2024-12029",
        "title":       "Remote Code Execution via Model Deserialization in InvokeAI",
        "description": "InvokeAI allows Remote Code Execution through malicious model "
                       "deserialization. Attacker can deliver backdoored model files "
                       "that execute code upon loading.",
        "severity":    "Critical",
        "cvss_score":  9.8,
        "source":      "AVID",
        "url":         "https://avidml.org/database/AVID-2026-R0023",
        "atlas_technique": "AML.T0018.002 - Embed Malware",
        "report_type": "Advisory",
        "ai_backdoor_relevant": True,
    },
    {
        "avid_id":     "AVID-2026-R0045",
        "cve_id":      "CVE-2025-27520",
        "title":       "BentoML Allows RCE via Insecure Deserialization",
        "description": "BentoML's runner server is vulnerable to insecure deserialization "
                       "leading to Remote Code Execution. ML model serving frameworks "
                       "are prime targets for backdoor injection.",
        "severity":    "Critical",
        "cvss_score":  9.8,
        "source":      "AVID",
        "url":         "https://avidml.org/database/AVID-2026-R0045",
        "atlas_technique": "AML.T0018 - Backdoor ML Model",
        "report_type": "Advisory",
        "ai_backdoor_relevant": True,
    },
    {
        "avid_id":     "AVID-2026-R0057",
        "cve_id":      "CVE-2025-32375",
        "title":       "Insecure Deserialization leads to RCE in BentoML Runner Server",
        "description": "BentoML runner server deserializes untrusted data without "
                       "validation. Enables backdoor model delivery that executes "
                       "arbitrary code on the ML serving infrastructure.",
        "severity":    "Critical",
        "cvss_score":  9.8,
        "source":      "AVID",
        "url":         "https://avidml.org/database/AVID-2026-R0057",
        "atlas_technique": "AML.T0018.002 - Embed Malware",
        "report_type": "Advisory",
        "ai_backdoor_relevant": True,
    },
    {
        "avid_id":     "AVID-2026-R0058",
        "cve_id":      "CVE-2025-3248",
        "title":       "Langflow Unauthenticated RCE via /api/v1/validate/code",
        "description": "Langflow versions before 1.3.0 allow unauthenticated remote "
                       "code execution via the /api/v1/validate/code endpoint. "
                       "AI workflow platforms are high-value targets for backdoor attacks.",
        "severity":    "Critical",
        "cvss_score":  9.8,
        "source":      "AVID",
        "url":         "https://avidml.org/database/AVID-2026-R0058",
        "atlas_technique": "AML.T0049 - Exploit Public-Facing Application",
        "report_type": "Advisory",
        "ai_backdoor_relevant": True,
    },
]

class AVIDLookupAgent:

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._cache: Dict[str, List[Dict]] = {}

    async def lookup(self, ai_fingerprint: Dict,
                     ml_stack: List[str],
                     verbose: bool = False) -> Dict:
        if verbose:
            print(f"\n  AVID Lookup Agent")
            print(f"   ML Stack: {ml_stack[:5]}")

        result = {
            "avid_findings":     [],
            "backdoor_relevant": [],
            "total_found":       0,
            "sources_queried":   ["AVID (avidml.org)"],
        }

        search_terms = self._build_search_terms(ml_stack, ai_fingerprint)

        if verbose:
            print(f"   Search terms: {search_terms[:5]}")

        live_results = await self._query_avid_live(search_terms, verbose)

        static_results = self._get_static_backdoor_entries(ml_stack, ai_fingerprint)

        all_results = live_results + static_results
        seen_ids    = set()
        deduped     = []
        for r in all_results:
            rid = r.get("cve_id") or r.get("avid_id") or r.get("title")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                deduped.append(r)

        severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Unknown": 4}
        deduped.sort(key=lambda x: severity_order.get(x.get("severity", "Unknown"), 4))

        result["avid_findings"]     = deduped
        result["backdoor_relevant"] = [
            r for r in deduped if r.get("ai_backdoor_relevant")
        ]
        result["total_found"]       = len(deduped)

        if verbose:
            print(f" AVID findings: {len(deduped)} total "
                  f"({len(result['backdoor_relevant'])} backdoor-relevant)")

        return result

    def _build_search_terms(self, ml_stack: List[str],
                             ai_fingerprint: Dict) -> List[str]:
        terms = set()

        for lib in ml_stack:
            lib_lower = lib.lower()
            for key, search_list in LIBRARY_SEARCH_TERMS.items():
                if key in lib_lower or lib_lower in key:
                    terms.update(search_list)
                    break
            terms.add(lib_lower.split("/")[0])

        if ai_fingerprint.get("model_upload_endpoints"):
            terms.update(["pickle", "deserialization", "model loading"])

        if ai_fingerprint.get("chat_components"):
            terms.update(["prompt injection", "llm", "jailbreak"])

        terms.update(["backdoor", "model poisoning", "supply chain"])

        return list(terms)[:10]

    async def _query_avid_live(self, search_terms: List[str],
                                verbose: bool) -> List[Dict]:
        results = []

        try:
            loop = asyncio.get_event_loop()
            html = await loop.run_in_executor(
                None, lambda: self._fetch_text(AVID_DB_URL)
            )

            if not html:
                if verbose:
                    print(" AVID live fetch failed - using cached entries")
                return results

            table_pattern = re.compile(
                r'\[AVID-(\d{4}-R\d+)\]\(([^)]+)\)\s*\|\s*([^|]+)\|',
                re.MULTILINE
            )

            for match in table_pattern.finditer(html):
                avid_id     = f"AVID-{match.group(1)}"
                url         = match.group(2).strip()
                description = match.group(3).strip()

                desc_lower = description.lower()
                relevant   = any(term.lower() in desc_lower
                                 for term in search_terms)

                if relevant:
                    cve_match = re.search(r"CVE-\d{4}-\d+", description)
                    cve_id    = cve_match.group(0) if cve_match else None

                    severity = self._infer_severity(description)

                    results.append({
                        "avid_id":     avid_id,
                        "cve_id":      cve_id,
                        "title":       description[:120],
                        "description": description,
                        "severity":    severity,
                        "cvss_score":  None,
                        "source":      "AVID",
                        "url":         f"https://avidml.org/database/{avid_id}",
                        "atlas_technique": self._map_to_atlas(description),
                        "report_type": "Advisory",
                        "ai_backdoor_relevant": self._is_backdoor_relevant(description),
                    })

            if verbose and results:
                print(f"  Live AVID: {len(results)} relevant entries found")

        except Exception as e:
            logger.warning(f"AVID live query failed: {e}")

        return results[:20]

    def _get_static_backdoor_entries(self, ml_stack: List[str],
                                      ai_fingerprint: Dict) -> List[Dict]:
        relevant = []
        observed = " ".join(
            [str(x) for x in (ml_stack or [])]
            + [str(x) for x in (ai_fingerprint or {}).get("technologies") or []]
            + [str(x) for x in (ai_fingerprint or {}).get("discovered_paths") or []]
            + [str(x) for x in (ai_fingerprint or {}).get("ml_endpoints") or []]
        ).lower()

        for entry in AVID_BACKDOOR_ENTRIES:
            title_lower = entry["title"].lower()
            desc_lower  = entry["description"].lower()
            blob = title_lower + " " + desc_lower

            if ai_fingerprint.get("model_upload_endpoints"):
                if any(kw in blob for kw in ["pickle", "deserialization", "model load"]):
                    relevant.append(entry)
                    continue

            subject_tokens = [
                tok for tok in re.findall(r"[a-z][a-z0-9_.-]{3,}", blob)
                if tok not in {
                    "allows", "remote", "code", "execution", "insecure",
                    "deserialization", "malicious", "model", "files",
                    "arbitrary", "via", "server", "leads", "through",
                }
            ]
            if any(tok in observed for tok in subject_tokens[:8]) and observed.strip():
                if entry not in relevant:
                    relevant.append(entry)

        return relevant

    def _infer_severity(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in
               ["rce", "remote code execution", "arbitrary code", "critical"]):
            return "Critical"
        if any(kw in text_lower for kw in
               ["high", "privilege escalation", "authentication bypass"]):
            return "High"
        if any(kw in text_lower for kw in
               ["medium", "injection", "xss", "information disclosure"]):
            return "Medium"
        return "Low"

    def _is_backdoor_relevant(self, text: str) -> bool:
        backdoor_keywords = [
            "backdoor", "poison", "deserializ", "pickle", "supply chain",
            "model load", "rce", "remote code", "arbitrary code",
            "malicious model", "model tamper", "model injection",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in backdoor_keywords)

    def _map_to_atlas(self, text: str) -> str:
        text_lower = text.lower()
        if "pickle" in text_lower or "deserializ" in text_lower:
            return "AML.T0018.002 - Embed Malware"
        if "poison" in text_lower or "backdoor" in text_lower:
            return "AML.T0018 - Backdoor ML Model"
        if "supply chain" in text_lower:
            return "AML.T0010 - ML Supply Chain Compromise"
        if "prompt injection" in text_lower:
            return "AML.T0051 - LLM Prompt Injection"
        if "jailbreak" in text_lower:
            return "AML.T0054 - LLM Jailbreak"
        return "AML.T0019 - Exploit Software Dependencies"

    def _fetch_text(self, url: str) -> Optional[str]:
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (SecurityScanner)",
                    "Accept":     "text/html,*/*"
                }
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read(500_000).decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"AVID fetch failed: {e}")
            return None
