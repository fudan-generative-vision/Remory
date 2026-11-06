"""Narrow, versioned wire-format fixes around the pinned official API tools.

No business logic, endpoint substitution, state inspection or retry lives here.
The official simulator still owns service restrictions and all mutations.
"""
from __future__ import annotations

import base64
import binascii
from email import policy
from email.parser import BytesParser
import json
import re
from urllib.parse import urlsplit

API_COMPAT_VERSION = "automationbench_api_wire_compat_v2"
API_COMPAT_RULES = (
    "bamboohr_redundant_service_prefix",
    "api_fetch_top_level_json_null",
    "api_fetch_recoverable_parameter_validation",
    "gmail_raw_unicode_subject",
    "catalog_unimplemented_endpoint_notice",
)

# These declarations have no matching route in the pinned official simulator.
# Disclose the limitation; never synthesize financial state or a success result.
_XERO = r"https://api\.xero\.com/api\.xro/2\.0/"
_BAMBOO = r"https://api\.bamboohr\.com/api/gateway\.php/[^/?#]+/v1/"
UNIMPLEMENTED_ENDPOINTS = {
    "xero.contactgroups.list": ("GET", _XERO + "ContactGroups"),
    "xero.accounts.create": ("PUT", _XERO + "Accounts"),
    "xero.reports.agedpayables": ("GET", _XERO + "Reports/AgedPayablesByContact"),
    "bamboohr.employees.list": ("GET", _BAMBOO + "employees"),
    "bamboohr.lastchanged": ("GET", _BAMBOO + "last_changed/employees"),
    "bamboohr.meta.fields": ("GET", _BAMBOO + "meta/fields"),
    "bamboohr.company.info": ("GET", _BAMBOO + "company_information"),
}
UNIMPLEMENTED_NOTICE = (
    "This endpoint is documented but not implemented in the pinned local simulator. "
    "This is not a network, VPN or credential failure; retrying the same endpoint "
    "will not help. Use other supported documented APIs."
)


def normalize_bamboohr_url(url):
    if not isinstance(url, str):
        return url
    # Work on the raw URL, not decoded/re-encoded IDs or query values. Only the
    # known catalog's host and duplicated service prefix are in scope.
    return re.sub(
        r"\A(https://api\.bamboohr\.com/api/gateway\.php/[^/?#]+/)"
        r"bamboohr/(?=v1(?:_\d+)?/)", r"\1", url, count=1)


def _quickbooks_text_body(url):
    """Keep the official exception for this endpoint's text/plain SQL body."""
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return (parsed.scheme == "https" and parsed.netloc in {
        "quickbooks.api.intuit.com", "sandbox-quickbooks.api.intuit.com"}
        and re.fullmatch(r"/v3/company/[^/]+/query", parsed.path) is not None)


def unimplemented_endpoint(arguments):
    url = normalize_bamboohr_url(arguments.get("url"))
    if not isinstance(url, str):
        return None
    path = url.split("?", 1)[0].split("#", 1)[0]
    for endpoint_id, (method, pattern) in UNIMPLEMENTED_ENDPOINTS.items():
        if arguments.get("method") == method and re.fullmatch(pattern, path):
            return endpoint_id
    return None


def _gmail_unicode_subject(raw):
    """Read an unambiguous Unicode Subject, without changing raw/MIME bytes.

    The upstream compat32 parser can return Header instead of str for UTF-8
    headers. Copy that header's decoded value into its existing subject argument;
    do not guess missing values or override any explicitly supplied subject.
    """
    if not isinstance(raw, str):
        return None
    try:
        encoded = raw.strip()
        decoded = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        legacy = BytesParser(policy=policy.compat32).parsebytes(decoded, headersonly=True)
        subjects = legacy.get_all("Subject", [])
        if len(subjects) != 1 or isinstance(subjects[0], str):
            return None
        modern = BytesParser(policy=policy.default).parsebytes(decoded, headersonly=True)
        subject = str(modern["Subject"])
        if not subject or "\ufffd" in subject or any(0xD800 <= ord(c) <= 0xDFFF for c in subject):
            return None  # Do not guess how malformed bytes should be decoded.
        return subject
    except (ValueError, TypeError, binascii.Error, IndexError):
        return None


def _normalize_gmail_body(forwarded):
    url = forwarded.get("url")
    if (forwarded.get("method") != "POST" or not isinstance(url, str)
        or re.fullmatch(r"https://gmail\.googleapis\.com/gmail/v1/users/[^/?#]+/messages/send", url) is None):
        return False
    body = forwarded.get("body")
    body_dict = json.loads(body) if isinstance(body, str) and body else body
    if not isinstance(body_dict, dict):
        return False
    params = forwarded.get("params")
    params_dict = json.loads(params) if isinstance(params, str) and params else params
    for fields in (params_dict, body_dict):
        if isinstance(fields, dict) and fields.get("subject") not in (None, ""):
            return False
    # Leave the other request shapes and explicit payload headers to upstream.
    if "message" in body_dict or "payload" in body_dict:
        return False
    subject = _gmail_unicode_subject(body_dict.get("raw"))
    if subject is None:
        return False
    result = {**body_dict, "subject": subject}
    forwarded["body"] = json.dumps(result, ensure_ascii=False) if isinstance(body, str) else result
    return True


def normalize_fetch_arguments(arguments):
    """Return (forwarding candidate, applied rules, recoverable error or None)."""
    forwarded = dict(arguments)
    rules = []
    if "url" in forwarded:
        url = normalize_bamboohr_url(forwarded["url"])
        if url != forwarded["url"]:
            forwarded["url"] = url
            rules.append("bamboohr_redundant_service_prefix")
    for field in ("params", "body"):
        value = forwarded.get(field)
        if value is None or value == "":
            continue  # Official implementation already supports these forms.
        if isinstance(value, dict):
            # Preserve official structured-object argument normalization.
            continue
        if not isinstance(value, str):
            return forwarded, rules, f"api_fetch.{field} must be a JSON object string or null"
        try:
            parsed = json.loads(value)
        except (ValueError, RecursionError):
            if field == "body" and _quickbooks_text_body(forwarded.get("url")):
                continue
            return forwarded, rules, f"api_fetch.{field} contains invalid JSON; use a JSON object string or null"
        if parsed is None:
            forwarded[field] = None
            rules.append(f"api_fetch_top_level_json_null:{field}")
        elif isinstance(parsed, str) and field == "body" and _quickbooks_text_body(forwarded.get("url")):
            continue
        elif not isinstance(parsed, dict):
            return forwarded, rules, f"api_fetch.{field} must encode a JSON object or null"
        # Preserve valid object JSON verbatim, including nested nulls/strings.
    if _normalize_gmail_body(forwarded):
        rules.append("gmail_raw_unicode_subject")
    return forwarded, rules, None


def normalize_search_output(content):
    """Correct URLs and disclose known catalog gaps, without filtering results."""
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return content, []
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return content, []
    rules = []
    for endpoint in data["results"]:
        if not isinstance(endpoint, dict):
            continue
        old = endpoint.get("url")
        new = normalize_bamboohr_url(old) if str(endpoint.get("id", "")).startswith("bamboohr.") else old
        if new != old:
            endpoint["url"] = new
            rules.append("bamboohr_redundant_service_prefix")
        if endpoint.get("id") in UNIMPLEMENTED_ENDPOINTS and unimplemented_endpoint(endpoint) == endpoint["id"]:
            notice = {"supported": False, "reason": UNIMPLEMENTED_NOTICE}
            if endpoint.get("simulator_availability") != notice:
                endpoint["simulator_availability"] = notice
                rules.append("catalog_unimplemented_endpoint_notice")
    if not rules:
        return content, []
    return json.dumps(data, ensure_ascii=False), list(dict.fromkeys(rules))


def annotate_fetch_output(content, arguments):
    """Keep real status/errors; only explain a confirmed missing local route."""
    endpoint_id = unimplemented_endpoint(arguments)
    if endpoint_id is None:
        return content, []
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return content, []
    error = data.get("error") if isinstance(data, dict) else None
    if (not isinstance(error, dict) or error.get("code") != 404
        or not str(error.get("message", "")).startswith("No handler for ")):
        return content, []  # In particular, never relabel service-gating 401s.
    data["simulator_availability"] = {"supported": False, "endpoint_id": endpoint_id,
                                      "reason": UNIMPLEMENTED_NOTICE}
    return json.dumps(data, ensure_ascii=False), ["catalog_unimplemented_endpoint_notice"]
