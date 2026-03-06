"""EUR-Lex client: SOAP (zeep) primary, REST/HTML fallback.

SOAP endpoint:  https://eur-lex.europa.eu/EURLexWebService
WSDL:           https://eur-lex.europa.eu/eurlex-ws?wsdl
REST HTML:      https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WSDL_URL = "https://eur-lex.europa.eu/eurlex-ws?wsdl"
SOAP_ENDPOINT = "https://eur-lex.europa.eu/EURLexWebService"
REST_HTML_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"

EXPERT_QUERIES: dict[str, str] = {
    "DORA": "SELECT CELEX WHERE DN = 32022R2554",
    "NIS2": "SELECT CELEX WHERE DN = 32022L2555",
}

_DEFAULT_RETRY_DELAYS = (1.0, 2.0, 4.0)  # seconds between retries


# ---------------------------------------------------------------------------
# SOAP client (optional — only available when zeep is installed + creds set)
# ---------------------------------------------------------------------------


def _build_soap_client(username: str, password: str):  # type: ignore[return]
    """Build a zeep SOAP client with WS-Security UsernameToken.

    Returns None if zeep is not installed.
    """
    try:
        from zeep import Client  # type: ignore
        from zeep.transports import Transport  # type: ignore
        from zeep.wsse.username import UsernameToken  # type: ignore
    except ImportError:
        logger.warning("zeep not installed — SOAP client unavailable")
        return None

    transport = Transport(timeout=30)
    wsse = UsernameToken(username, password, use_digest=False)
    client = Client(wsdl=WSDL_URL, wsse=wsse, transport=transport)
    # Bind to the concrete endpoint
    client.service._binding_options["address"] = SOAP_ENDPOINT
    return client


def _fetch_via_soap(celex: str, username: str, password: str) -> Optional[str]:
    """Fetch regulation HTML via EUR-Lex SOAP service.

    Returns the raw HTML string or None on failure.
    """
    client = _build_soap_client(username, password)
    if client is None:
        return None

    # Determine the expert query key from celex
    query = None
    for reg_id, q in EXPERT_QUERIES.items():
        if celex in q:
            query = q
            break
    if query is None:
        query = f"SELECT CELEX WHERE DN = {celex}"

    try:
        logger.info("Fetching %s via SOAP (query: %s)", celex, query)
        response = client.service.doQuery(expertQuery=query, page=1, pageSize=10)
        # The SOAP response usually contains result metadata; we still need
        # the actual HTML text, typically fetched by a separate getDocument call.
        # Try common operation names:
        for op_name in ("getDocumentHtml", "getDocument", "doQuery"):
            if hasattr(client.service, op_name) and op_name != "doQuery":
                result = getattr(client.service, op_name)(celex=celex)
                if isinstance(result, str) and "<html" in result.lower():
                    return result
        # Fallback: extract any embedded HTML in the doQuery response
        if hasattr(response, "result"):
            return str(response.result)
        return str(response)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SOAP fetch failed for %s: %s", celex, exc)
        return None


# ---------------------------------------------------------------------------
# REST / HTML fallback
# ---------------------------------------------------------------------------


def _fetch_via_rest(celex: str, retries: tuple[float, ...] = _DEFAULT_RETRY_DELAYS) -> str:
    """Fetch regulation HTML from EUR-Lex REST endpoint.

    Raises:
        httpx.HTTPError: After exhausting all retries.
    """
    url = REST_HTML_URL.format(celex=celex)
    headers = {
        "Accept": "text/html,application/xhtml+xml",
        "User-Agent": "eu-regulatory-rag/0.1 (+https://github.com/your-org/eu-regulatory-rag)",
    }

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt, delay in enumerate([0.0, *retries], start=1):
        if delay:
            logger.debug("Retry %d/%d — sleeping %.1fs", attempt, len(retries) + 1, delay)
            time.sleep(delay)
        try:
            logger.info("Fetching %s via REST (attempt %d)", celex, attempt)
            with httpx.Client(follow_redirects=True, timeout=30) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                return response.text
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                # Rate limited — use longer backoff
                wait = delay * 3 or 5.0
                logger.warning("Rate limited (429) — waiting %.1fs", wait)
                time.sleep(wait)
            last_exc = exc
        except httpx.HTTPError as exc:
            last_exc = exc

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class EurLexClient:
    """High-level EUR-Lex document fetcher.

    Uses SOAP when credentials are provided, otherwise falls back to REST.
    """

    def __init__(self, username: str = "", password: str = "") -> None:
        self.username = username
        self.password = password

    @property
    def _use_soap(self) -> bool:
        return bool(self.username and self.password)

    def fetch_html(self, celex: str) -> str:
        """Return the full HTML of a regulation identified by *celex*.

        Tries SOAP first (if credentials configured), then REST.

        Raises:
            RuntimeError: If both strategies fail.
        """
        if self._use_soap:
            html = _fetch_via_soap(celex, self.username, self.password)
            if html:
                # Basic sanity check — if it looks like actual HTML, use it
                if "<" in html:
                    return html
                logger.warning("SOAP returned non-HTML content — falling back to REST")

        logger.info("Using REST fallback for %s", celex)
        return _fetch_via_rest(celex)

    def fetch_regulation(self, regulation_id: str) -> str:
        """Convenience: fetch by regulation ID (DORA / NIS2).

        Raises:
            KeyError: If regulation_id is not in the known registry.
        """
        from ingestion.sources.regulations import get_regulation

        meta = get_regulation(regulation_id)
        return self.fetch_html(meta.celex)

    # ------------------------------------------------------------------
    # HTML utilities
    # ------------------------------------------------------------------

    @staticmethod
    def extract_text(html: str) -> str:
        """Quick utility: strip HTML tags and return plain text."""
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(separator="\n", strip=True)
