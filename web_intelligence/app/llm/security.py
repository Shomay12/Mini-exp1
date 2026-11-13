"""Connection security boundary for user-hosted local and approved remote LLM endpoints.

This module intentionally maintains a separate trust policy from external web search:
- Loopback endpoints (localhost, 127.0.0.0/8, ::1) are allowed when approved.
- Non-loopback LAN or remote endpoints require explicit `allow_remote_endpoint=True`.
- Cloud metadata, link-local, multicast, unspecified IPs, embedded credentials,
  non-HTTP(S) schemes, and HTTP redirects are unconditionally blocked.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse, urlunparse

from web_intelligence.app.core.logging import redact_text
from web_intelligence.app.core.security import (
    AsyncResolver,
    SSRFError,
    URLValidationError,
    default_dns_resolver,
)

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

_ALWAYS_BLOCKED_HOSTNAMES = frozenset(
    {
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "0.0.0.0",
    }
)

_LOOPBACK_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    }
)

# Prevent SSRF against non-HTTP infrastructure daemons while permitting arbitrary LLM ports.
_BLOCKED_NON_HTTP_PORTS = frozenset(
    {
        21,
        22,
        23,
        25,
        53,
        110,
        111,
        135,
        139,
        143,
        445,
        465,
        587,
        993,
        995,
        1433,
        1521,
        2049,
        3306,
        3389,
        5432,
        6379,
        9200,
        11211,
        27017,
    }
)

_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / Cloud IMDS
    ipaddress.ip_network("224.0.0.0/4"),  # Multicast
    ipaddress.ip_network("240.0.0.0/4"),  # Reserved
    ipaddress.ip_network("::/128"),  # Unspecified
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("ff00::/8"),  # IPv6 multicast
)


class LLMURLValidationError(URLValidationError):
    """Raised when an LLM endpoint URL or path fails syntax validation."""


class LLMConnectionSecurityError(SSRFError):
    """Raised when an LLM endpoint violates local/remote or SSRF security policy."""


def _strip_brackets_and_zone(raw_host: str) -> str:
    host = raw_host.strip().rstrip(".").lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host.split("%", 1)[0]


def _parse_ip(raw_host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    cleaned = _strip_brackets_and_zone(raw_host)
    try:
        ip_obj = ipaddress.ip_address(cleaned)
    except ValueError:
        return None
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        return ip_obj.ipv4_mapped
    return ip_obj


def is_always_blocked_ip(raw_ip: str) -> bool:
    """Return True if an IP must never be dialed regardless of remote approval."""
    ip_obj = _parse_ip(raw_ip)
    if ip_obj is None:
        return True
    if ip_obj.is_loopback:
        return False
    if ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_unspecified or ip_obj.is_reserved:
        return True
    return any(ip_obj in net for net in _ALWAYS_BLOCKED_NETWORKS)


def is_loopback_ip(raw_ip: str) -> bool:
    """Return True if the literal IP is strictly a loopback address."""
    ip_obj = _parse_ip(raw_ip)
    if ip_obj is None or is_always_blocked_ip(raw_ip):
        return False
    return ip_obj.is_loopback


def is_loopback_hostname(hostname: str) -> bool:
    """Return True if hostname refers to a local loopback host or loopback IP."""
    cleaned = _strip_brackets_and_zone(hostname)
    if not cleaned:
        return False
    if cleaned in _LOOPBACK_HOSTNAMES or cleaned.endswith(".localhost"):
        return True
    return is_loopback_ip(cleaned)


def normalize_base_url(base_url: str) -> str:
    """Normalize an LLM endpoint base URL without altering meaningful path prefixes like /v1."""
    parsed = urlparse(base_url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc
    path = parsed.path.rstrip("/")
    if path == "/":
        path = ""
    return urlunparse((scheme, netloc, path, "", "", ""))


def validate_api_endpoint_path(api_endpoint: str | None) -> str | None:
    """Validate optional relative endpoint override so it cannot redirect to another host."""
    if api_endpoint is None:
        return None
    cleaned = api_endpoint.strip()
    if not cleaned:
        return None
    if _CONTROL_CHARS_RE.search(cleaned):
        raise LLMURLValidationError("API endpoint path contains control characters.")
    if not cleaned.startswith("/") or cleaned.startswith("//"):
        raise LLMURLValidationError(
            "API endpoint override must be a relative path starting with '/'."
        )
    if "://" in cleaned or "@" in cleaned or "?" in cleaned or "#" in cleaned:
        raise LLMURLValidationError(
            "API endpoint override must not include schemes, credentials, queries, or fragments."
        )
    segments = [seg for seg in cleaned.split("/") if seg]
    if any(seg in {".", ".."} for seg in segments):
        raise LLMURLValidationError("API endpoint override must not contain path traversal.")
    return cleaned


def validate_llm_endpoint_url(
    base_url: str,
    *,
    allow_remote_endpoint: bool = False,
    approved_by_user: bool = True,
    api_endpoint: str | None = None,
) -> str:
    """Validate LLM endpoint syntax and trust policy synchronously.

    Returns the normalized `base_url`. Raises `LLMURLValidationError` or
    `LLMConnectionSecurityError` on violation.
    """
    if not isinstance(base_url, str):
        raise LLMURLValidationError("Endpoint base_url must be a string.")
    stripped = base_url.strip()
    if not stripped or len(stripped) > 1024:
        raise LLMURLValidationError("Endpoint base_url is empty or exceeds 1024 characters.")
    if _CONTROL_CHARS_RE.search(stripped):
        raise LLMURLValidationError("Endpoint base_url contains control characters.")

    parsed = urlparse(stripped)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise LLMURLValidationError(
            f"Unsupported endpoint scheme '{scheme or '<empty>'}'; expected http or https."
        )
    if parsed.username is not None or parsed.password is not None:
        raise LLMURLValidationError("Endpoint URLs with embedded credentials are not allowed.")
    if parsed.query or parsed.fragment:
        raise LLMURLValidationError(
            "Endpoint base_url must not contain query parameters or URL fragments."
        )

    hostname = parsed.hostname
    if not hostname:
        raise LLMURLValidationError("Endpoint URL is missing a valid hostname.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise LLMURLValidationError("Endpoint URL contains an invalid port.") from exc
    if port is not None and port in _BLOCKED_NON_HTTP_PORTS:
        raise LLMConnectionSecurityError(
            f"Port {port} is reserved for non-HTTP services and blocked by security policy."
        )

    validate_api_endpoint_path(api_endpoint)

    cleaned_host = _strip_brackets_and_zone(hostname)
    if cleaned_host in _ALWAYS_BLOCKED_HOSTNAMES:
        raise LLMConnectionSecurityError(
            f"Endpoint host '{hostname}' is unconditionally blocked by security policy."
        )

    if not approved_by_user:
        raise LLMConnectionSecurityError(
            "LLM endpoint connection requires explicit user approval before activation or use."
        )

    literal_ip = _parse_ip(cleaned_host)
    if literal_ip is not None:
        if is_always_blocked_ip(cleaned_host):
            raise LLMConnectionSecurityError(
                f"IP address '{hostname}' is unconditionally blocked (link-local/metadata/multicast)."
            )
        if not literal_ip.is_loopback and not allow_remote_endpoint:
            raise LLMConnectionSecurityError(
                f"Non-loopback endpoint '{hostname}' is blocked by default. "
                "Enable allow_remote_endpoint to allow LAN or remote inference servers."
            )
    elif not is_loopback_hostname(cleaned_host) and not allow_remote_endpoint:
        raise LLMConnectionSecurityError(
            f"Remote host '{hostname}' is blocked by default. "
            "Enable allow_remote_endpoint to allow LAN or remote inference servers."
        )

    return normalize_base_url(stripped)


async def validate_llm_endpoint_async(
    base_url: str,
    *,
    allow_remote_endpoint: bool = False,
    approved_by_user: bool = True,
    api_endpoint: str | None = None,
    resolver: AsyncResolver | None = None,
) -> tuple[str, list[str]]:
    """Validate syntax, trust policy, and DNS-resolved IP addresses prior to dialing."""
    clean_url = validate_llm_endpoint_url(
        base_url,
        allow_remote_endpoint=allow_remote_endpoint,
        approved_by_user=approved_by_user,
        api_endpoint=api_endpoint,
    )
    parsed = urlparse(clean_url)
    hostname = _strip_brackets_and_zone(parsed.hostname or "")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    literal_ip = _parse_ip(hostname)
    if literal_ip is not None:
        return clean_url, [str(literal_ip)]

    resolve_fn = resolver or default_dns_resolver
    try:
        resolved_ips = await resolve_fn(hostname, port)
    except OSError as exc:
        raise LLMConnectionSecurityError(
            f"DNS resolution failed for LLM host '{hostname}'."
        ) from exc

    if not resolved_ips:
        raise LLMConnectionSecurityError(
            f"DNS resolution returned no IP addresses for LLM host '{hostname}'."
        )

    for raw_ip in resolved_ips:
        if is_always_blocked_ip(raw_ip):
            raise LLMConnectionSecurityError(
                f"Resolved IP address '{raw_ip}' for '{hostname}' is blocked by security policy."
            )
        if not allow_remote_endpoint and not is_loopback_ip(raw_ip):
            raise LLMConnectionSecurityError(
                f"Host '{hostname}' resolved to non-loopback IP '{raw_ip}' without remote approval."
            )

    return clean_url, resolved_ips


def redact_llm_error(exc: Exception | str, *, secrets: list[str | None] | None = None) -> str:
    """Produce a bounded, secret-safe error string suitable for API/UI display and logs."""
    raw = str(exc) if isinstance(exc, Exception) else exc
    cleaned = redact_text(raw, [s for s in (secrets or []) if s])
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > 320:
        return cleaned[:317] + "..."
    return cleaned
