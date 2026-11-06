"""Origin-scoped SGLang credentials, shared by data and control requests."""
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def origin(url):
    value = urlsplit(url)
    if value.scheme not in {"https", "http"} or not value.hostname or value.username or value.password:
        raise ValueError("inference URL must be a credential-free HTTP(S) URL")
    return value.scheme, value.hostname, value.port or (443 if value.scheme == "https" else 80)


def request_headers(url, *, api_key=None):
    if api_key is None:
        filename = os.environ.get("REMORY_SGLANG_API_KEY_FILE")
        token = os.environ.get("REMORY_SGLANG_API_KEY")
        if not filename and not token:
            return {}
        configured = os.environ.get("REMORY_SGLANG_URL")
        if not configured:
            raise ValueError("REMORY_SGLANG_URL is required to scope inference credentials")
        if origin(url) != origin(configured):
            return {}  # Never forward an origin's key to another server.
        api_key = Path(filename).expanduser().read_text().strip() if filename else token
    if not isinstance(api_key, str) or not api_key or "\n" in api_key or "\r" in api_key:
        raise ValueError("invalid inference API key")
    return {"Authorization": "Bearer " + api_key}


def open_request(request, *, timeout, api_key=None):
    if isinstance(request, str):
        request = Request(request)
    for key, value in request_headers(request.full_url, api_key=api_key).items():
        request.add_header(key, value)
    # Keep credentials and model traffic on this explicitly configured origin.
    return build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=timeout)
