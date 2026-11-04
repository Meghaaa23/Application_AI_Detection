"""Authenticated-scan support."""

from __future__ import annotations

import asyncio
import json
import logging
from . import ssrf_guard
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOGOUT_RE = re.compile(
    r"(log|sign)[\s_\-]?(out|off)\b|/logout|/signout|/sign-out|end[_\-]?session|"
    r"destroy[_\-]?session",
    re.I,
)
LOGIN_TEXT_RE = re.compile(r"\b(log[\s\-]?in|sign[\s\-]?in|signin|login)\b", re.I)
ACCOUNT_TEXT_RE = re.compile(r"\b(account|my account|user)\b", re.I)
JWT_RE = re.compile(r"^eyJ[\w-]+\.[\w-]+\.[\w-]*$")

COMMON_LOGIN_PATHS = [
    "/login", "/signin", "/sign-in", "/sign_in", "/user/login",
    "/users/sign_in", "/account/login", "/accounts/login/", "/auth/login",
    "/#/login", "/#/signin", "/login.php", "/wp-login.php",
]

_STATIC_RE = re.compile(
    r"\.(js|css|png|jpe?g|gif|svg|ico|woff2?|ttf|map|webp)(\?|$)", re.I
)

_UA = "Mozilla/5.0 (SecurityScanner)"


def is_logout_url(url: str) -> bool:
    return bool(LOGOUT_RE.search(url or ""))


def _norm_host(host: str) -> str:
    host = (host or "").lower().lstrip(".")
    return "localhost" if host in ("127.0.0.1", "::1") else host


def _host_key(url: str) -> str:
    try:
        p = urllib.parse.urlparse(url)
    except Exception:
        return ""
    host = _norm_host(p.hostname or "")
    port = p.port or (443 if p.scheme == "https" else 80)
    return f"{host}:{port}"


@dataclass
class AuthSession:
    target_url: str
    username: str = ""
    login_url: str = ""
    username_selector: str = ""
    password_selector: str = ""
    success_indicator: str = ""

    logged_in: bool = False
    method: str = "none"
    login_page_url: str = ""
    landing_url: str = ""
    cookies: List[Dict] = field(default_factory=list)
    storage_origins: List[Dict] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    auth_header_source: str = ""
    error: str = ""
    relogin_count: int = 0
    probe_url: str = ""
    probe_signature: Optional[Tuple] = None

    _password: str = field(default="", repr=False)

    def cookie_header(self) -> str:
        return "; ".join(
            f"{c['name']}={c['value']}" for c in self.cookies
            if c.get("name")
        )

    def request_headers(self) -> Dict[str, str]:
        out = dict(self.headers)
        ch = self.cookie_header()
        if ch:
            out["Cookie"] = ch
        return out

    def summary(self) -> Dict:
        """Report-safe view — no password, no cookie/token values."""
        return {
            "enabled":            True,
            "logged_in":          self.logged_in,
            "method":             self.method,
            "username":           self.username,
            "login_page_url":     self.login_page_url or self.login_url,
            "landing_url":        self.landing_url,
            "cookie_names":       sorted({c.get("name", "") for c in self.cookies}),
            "auth_header":        bool(self.headers.get("Authorization")),
            "auth_header_source": self.auth_header_source,
            "expiry_monitored":   bool(self.probe_url),
            "relogin_count":      self.relogin_count,
            "error":              self.error,
        }

    async def ensure_alive(self, verbose: bool = False) -> bool:
        if not self.logged_in or self.method != "form_login":
            return self.logged_in
        if not self.probe_url or self.probe_signature is None:
            return True
        loop = asyncio.get_event_loop()
        sig = await loop.run_in_executor(
            None, lambda: _probe_signature(self.probe_url, self.request_headers())
        )
        if sig == self.probe_signature:
            return True
        if self.relogin_count >= 3:
            if verbose:
                print("   ! Session appears expired; re-login limit reached")
            return False
        if verbose:
            print("   -> Session expired - logging in again...")
        fresh = await login(
            target_url=self.target_url, username=self.username,
            password=self._password, login_url=self.login_url,
            username_selector=self.username_selector,
            password_selector=self.password_selector,
            success_indicator=self.success_indicator,
            verbose=False,
        )
        if not fresh.logged_in:
            self.error = f"re-login failed: {fresh.error}"
            return False
        self.cookies, self.storage_origins = fresh.cookies, fresh.storage_origins
        self.headers, self.landing_url = fresh.headers, fresh.landing_url
        self.probe_url, self.probe_signature = fresh.probe_url, fresh.probe_signature
        self.relogin_count += 1
        return True


_ACTIVE: Dict[str, AuthSession] = {}
_PREV_OPENER = None
_INSTALLED = False


def session_for(url: str) -> Optional[AuthSession]:
    s = _ACTIVE.get(_host_key(url))
    return s if s and s.logged_in else None


def any_active() -> bool:
    return any(s.logged_in for s in _ACTIVE.values())


class _AuthInjectHandler(urllib.request.BaseHandler):
    """Attach the session to same-host requests; never to logout URLs."""

    def http_request(self, req):
        url = req.full_url
        sess = session_for(url)
        if not sess or is_logout_url(url):
            return req
        for name, value in sess.headers.items():
            if not req.has_header(name.capitalize()):
                req.add_unredirected_header(name.capitalize(), value)
        ours = sess.cookie_header()
        if ours:
            existing = req.get_header("Cookie") or req.unredirected_hdrs.get("Cookie")
            if existing:
                have = {p.split("=", 1)[0].strip() for p in existing.split(";")}
                extra = "; ".join(
                    f"{c['name']}={c['value']}" for c in sess.cookies
                    if c.get("name") not in have
                )
                if extra:
                    req.headers.pop("Cookie", None)
                    req.add_unredirected_header("Cookie", f"{existing}; {extra}")
            else:
                req.add_unredirected_header("Cookie", ours)
        return req

    https_request = http_request


_HANDLER = _AuthInjectHandler()


def auth_handlers() -> list:
    """Extra handlers for call sites that build their own opener."""
    return [_HANDLER]


def activate(session: AuthSession) -> None:
    global _PREV_OPENER, _INSTALLED
    _ACTIVE[_host_key(session.target_url)] = session
    if not _INSTALLED:
        _PREV_OPENER = urllib.request._opener
        urllib.request.install_opener(urllib.request.build_opener(_HANDLER))
        _INSTALLED = True


def deactivate(session: AuthSession) -> None:
    global _INSTALLED
    key = _host_key(session.target_url)
    if _ACTIVE.get(key) is session:
        _ACTIVE.pop(key, None)
    if not _ACTIVE and _INSTALLED:
        urllib.request.install_opener(_PREV_OPENER)
        _INSTALLED = False


async def prepare_context(context) -> None:
    """Seed a Playwright context with every active session (host-scoped)."""
    sessions = [s for s in _ACTIVE.values() if s.logged_in]
    if not sessions:
        return
    try:
        cookies = [c for s in sessions for c in s.cookies]
        if cookies:
            await context.add_cookies(cookies)
        origins = [o for s in sessions for o in s.storage_origins]
        if origins:
            await context.add_init_script(_storage_seed_script(origins))

        async def _route(route):
            url = route.request.url
            if is_logout_url(url) and route.request.resource_type in ("document", "xhr", "fetch"):
                await route.abort()
                return
            sess = session_for(url)
            if sess and sess.headers:
                hdrs = dict(route.request.headers)
                for k, v in sess.headers.items():
                    hdrs.setdefault(k.lower(), v)
                await route.continue_(headers=hdrs)
            else:
                await route.continue_()

        hosts = {_host_key(s.target_url) for s in sessions}
        await context.route(lambda u: _host_key(u) in hosts, _route)
    except Exception as e:
        logger.warning(f"prepare_context failed: {e}")


def _storage_seed_script(origins: List[Dict]) -> str:
    data = {
        o.get("origin", ""): o.get("localStorage") or []
        for o in origins if o.get("origin")
    }
    return (
        "(() => { try {"
        f" const d = {json.dumps(data)};"
        " const items = d[location.origin]; if (!items) return;"
        " for (const it of items) {"
        "   if (localStorage.getItem(it.name) === null) localStorage.setItem(it.name, it.value);"
        " } } catch (e) {} })();"
    )


def _probe_signature(url: str, headers: Optional[Dict[str, str]]) -> Tuple:
    """(status, final_path, has_password_input) — bypasses the global handler."""
    opener = urllib.request.build_opener()
    req = urllib.request.Request(url, headers={"User-Agent": _UA, **(headers or {})})
    try:
        with opener.open(req, timeout=8) as resp:
            body = resp.read(200_000).decode("utf-8", errors="replace")
            final = urllib.parse.urlparse(resp.geturl()).path
            return (resp.status, final, 'type="password"' in body.lower()
                    or "type='password'" in body.lower())
    except urllib.error.HTTPError as e:
        return (e.code, urllib.parse.urlparse(url).path, False)
    except Exception as e:
        return ("error", type(e).__name__, False)


def _pick_probe(candidates: List[str], auth_headers: Dict[str, str]) -> Tuple[str, Optional[Tuple]]:
    for url in candidates:
        authed = _probe_signature(url, auth_headers)
        if authed[0] == "error":
            continue
        anon = _probe_signature(url, None)
        if authed != anon:
            return url, authed
    return "", None


_FIND_FORM_JS = r"""
({userSel, passSel}) => {
  const visible = el => {
    if (!el) return false;
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  document.querySelectorAll('[data-scan-auth]').forEach(e => e.removeAttribute('data-scan-auth'));
  const pw = passSel ? document.querySelector(passSel)
           : [...document.querySelectorAll('input[type=password]')].find(visible);
  if (!pw) return {found: false};
  pw.setAttribute('data-scan-auth', 'pass');
  const scope = pw.form || pw.closest('form, [role=form], mat-card, .card, main, div') || document;
  let user = userSel ? document.querySelector(userSel) : null;
  if (!user) {
    const cands = [...scope.querySelectorAll('input')].filter(i =>
      visible(i) && i !== pw && ['text','email','tel',''].includes((i.getAttribute('type')||'').toLowerCase()));
    const hint = i => /user|email|login|account|name|ident/i.test(
      [i.name, i.id, i.autocomplete, i.placeholder, i.getAttribute('aria-label')].join(' '));
    const before = cands.filter(i => i.compareDocumentPosition(pw) & Node.DOCUMENT_POSITION_FOLLOWING);
    user = before.filter(hint).pop() || before.pop() || cands.find(hint) || cands[0] || null;
  }
  if (user) user.setAttribute('data-scan-auth', 'user');
  const btns = [...scope.querySelectorAll('button, input[type=submit], [role=button]')].filter(visible);
  const txt = b => (b.innerText || b.value || b.getAttribute('aria-label') || '').trim();
  const submit = btns.find(b => /log\s*-?in|sign\s*-?in|submit|continue/i.test(txt(b)))
              || btns.find(b => (b.getAttribute('type')||'').toLowerCase() === 'submit')
              || btns[0] || null;
  if (submit) submit.setAttribute('data-scan-auth', 'submit');
  return {found: true, user: !!user, submit: !!submit};
}
"""

_ERROR_TEXT_JS = r"""
() => {
  const el = document.querySelector('[role=alert], .error, .alert-danger, .invalid-feedback, mat-error, .error-message');
  return el ? (el.innerText || '').trim().slice(0, 160) : '';
}
"""


async def _find_login_form(page, sess: AuthSession) -> bool:
    args = {"userSel": sess.username_selector or None,
            "passSel": sess.password_selector or None}
    sel = sess.password_selector or "input[type=password]"
    try:
        await page.wait_for_selector(sel, state="visible", timeout=4000)
    except Exception:
        return False
    try:
        res = await page.evaluate(_FIND_FORM_JS, args)
        return bool(res.get("found"))
    except Exception:
        return False


async def _goto(page, url: str) -> bool:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass
        return True
    except Exception:
        return False


async def _click_login_link(page) -> bool:
    """Follow a visible login / account control (behavioural, text-based)."""
    for pattern in (LOGIN_TEXT_RE, ACCOUNT_TEXT_RE):
        loc = page.locator("a, button, [role=button], [role=menuitem]").filter(
            has_text=pattern
        )
        try:
            n = min(await loc.count(), 3)
        except Exception:
            n = 0
        for i in range(n):
            el = loc.nth(i)
            try:
                if not await el.is_visible():
                    continue
                await el.click(timeout=4000)
                await page.wait_for_timeout(1200)
                if await page.locator("input[type=password]").first.is_visible():
                    return True
            except Exception:
                continue
    return False


_SETUP_URL_RE = re.compile(
    r"/(?:setup|install|installer|installation|configure|config|wizard)"
    r"(?:\.php|\.jsp|\.aspx)?(?:/|$|\?)", re.I)
_SETUP_CONTENT_MARKERS = (
    "create / reset database", "create/reset database", "installation wizard",
    "setup wizard", "run the installer", "database setup",
    "installation is not complete", "please complete the installation",
)

def _looks_like_setup_or_install(url: str, content: str) -> bool:
    """True when a page is an application setup/installation/initialization page."""
    if url and _SETUP_URL_RE.search(url):
        return True
    low = (content or "").lower()
    return any(m in low for m in _SETUP_CONTENT_MARKERS)

async def login(target_url: str, username: str, password: str,
                login_url: str = "", username_selector: str = "",
                password_selector: str = "", success_indicator: str = "",
                verbose: bool = False) -> AuthSession:
    sess = AuthSession(
        target_url=target_url, username=username, login_url=login_url or "",
        username_selector=username_selector or "",
        password_selector=password_selector or "",
        success_indicator=success_indicator or "",
        method="form_login", _password=password,
    )
    try:
        from playwright.async_api import async_playwright
    except Exception:
        sess.error = "Playwright not installed — cannot perform form login"
        return sess

    target_host = _host_key(target_url)
    observed_auth: Dict[str, str] = {}
    api_gets: List[str] = []

    def _on_request(req):
        try:
            if _host_key(req.url) != target_host:
                return
            a = req.headers.get("authorization")
            if a and not observed_auth:
                observed_auth["Authorization"] = a
        except Exception:
            pass

    def _on_response(resp):
        try:
            req = resp.request
            if (req.method == "GET" and resp.status == 200
                    and _host_key(resp.url) == target_host
                    and req.resource_type in ("xhr", "fetch")
                    and not _STATIC_RE.search(resp.url)
                    and not is_logout_url(resp.url)
                    and resp.url not in api_gets):
                api_gets.append(resp.url)
        except Exception:
            pass

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = await browser.new_context(user_agent=_UA)
            await ssrf_guard.install_browser_guard(context)
            page = await context.new_page()
            page.on("request", _on_request)

            found = False
            if login_url:
                found = await _goto(page, login_url) and await _find_login_form(page, sess)
            else:
                if await _goto(page, target_url):
                    found = await _find_login_form(page, sess)
                    if not found and await _click_login_link(page):
                        found = await _find_login_form(page, sess)
                if not found:
                    base = target_url.rstrip("/")
                    for path in COMMON_LOGIN_PATHS:
                        if await _goto(page, base + path) and await _find_login_form(page, sess):
                            found = True
                            break
            if not found:
                sess.error = ("No login form (visible password field) found"
                              + (f" at {login_url}" if login_url else
                                 " — provide the login page URL"))
                await browser.close()
                return sess
            sess.login_page_url = page.url
            if verbose:
                print(f"   Login form found at {page.url}")

            before_cookies = {c["name"] for c in await context.cookies()}

            user_loc = page.locator('[data-scan-auth="user"]')
            if username and await user_loc.count():
                await user_loc.first.fill(username, timeout=4000)
            await page.locator('[data-scan-auth="pass"]').first.fill(password, timeout=4000)
            submit = page.locator('[data-scan-auth="submit"]')
            if await submit.count():
                await submit.first.click(timeout=5000)
            else:
                await page.locator('[data-scan-auth="pass"]').first.press("Enter")
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            await page.wait_for_timeout(1500)

            content = ""
            try:
                content = await page.content()
            except Exception:
                pass
            pw_still_visible = False
            try:
                pw_still_visible = await page.locator(
                    sess.password_selector or "input[type=password]"
                ).first.is_visible()
            except Exception:
                pass
            after_cookies = await context.cookies()
            new_cookie = any(c["name"] not in before_cookies for c in after_cookies)
            state = await context.storage_state()
            storage_token = _token_from_storage(state.get("origins") or [], target_host)

            if _looks_like_setup_or_install(page.url, content):
                sess.error = (
                    f"Login reached an application setup/initialization page "
                    f"({page.url}). Initialize the app (e.g. create its database) "
                    f"before authenticated scanning."
                )
                await browser.close()
                return sess

            if success_indicator:
                ok = success_indicator.lower() in content.lower()
            else:
                ok = not pw_still_visible and (
                    new_cookie or bool(storage_token) or bool(observed_auth)
                    or page.url.rstrip("/") != sess.login_page_url.rstrip("/")
                )
            if not ok:
                msg = ""
                try:
                    msg = await page.evaluate(_ERROR_TEXT_JS)
                except Exception:
                    pass
                sess.error = "Login did not succeed" + (f": {msg}" if msg else
                             " (login form still shown / no session established)")
                await browser.close()
                return sess

            page.on("response", _on_response)
            sess.landing_url = page.url
            await _goto(page, target_url)
            await page.wait_for_timeout(1500)

            state = await context.storage_state()
            target_name = target_host.rsplit(":", 1)[0]
            sess.cookies = [
                c for c in state.get("cookies") or []
                if _norm_host(c.get("domain", "")) == target_name
                or target_name.endswith("." + _norm_host(c.get("domain", "")))
            ]
            sess.storage_origins = [
                o for o in state.get("origins") or []
                if _host_key(o.get("origin", "")) == target_host
            ]
            if observed_auth:
                sess.headers.update(observed_auth)
                sess.auth_header_source = "observed"
            else:
                tok = _token_from_storage(sess.storage_origins, target_host)
                if tok:
                    sess.headers["Authorization"] = f"Bearer {tok}"
                    sess.auth_header_source = "web_storage"
            await browser.close()
    except Exception as e:
        sess.error = f"Login automation error: {type(e).__name__}: {e}"
        return sess

    sess.logged_in = True
    loop = asyncio.get_event_loop()
    candidates = [u for u in [sess.landing_url, target_url] if u] + api_gets[:6]
    sess.probe_url, sess.probe_signature = await loop.run_in_executor(
        None, lambda: _pick_probe(candidates, sess.request_headers())
    )
    if verbose:
        print(f"   Logged in as {username!r} | cookies={len(sess.cookies)} | "
              f"auth header={'yes (' + sess.auth_header_source + ')' if sess.headers else 'no'} | "
              f"expiry check={'on' if sess.probe_url else 'unavailable'}")
    return sess


def _token_from_storage(origins: List[Dict], target_host: str) -> str:
    for o in origins:
        if _host_key(o.get("origin", "")) != target_host:
            continue
        for item in o.get("localStorage") or []:
            v = str(item.get("value") or "").strip().strip('"')
            if JWT_RE.match(v):
                return v
    return ""

