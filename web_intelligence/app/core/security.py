"""URL validation, canonicalization, SSRF defense, and per-host rate limiting."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import time
from collections.abc import Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
AsyncResolver = Callable[[str, int], Awaitable[list[str]]]

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)

_BLOCKED_HOSTNAME_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".arpa",
    ".localdomain",
)

_EXTRA_BLOCKED_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / Cloud metadata
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8"),
)

_TRACKING_PARAM_EXACT = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "ref_src",
        "igsh",
        "igshid",
        "_hsenc",
        "_hsmi",
        "yclid",
    }
)
_TRACKING_PARAM_PREFIXES = ("utm_", "oly_", "vero_")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_MULTI_SLASH_RE = re.compile(r"/{2,}")


class URLValidationError(ValueError):
    """Raised when a URL is syntactically invalid or uses an unsupported scheme/port."""


class SSRFError(PermissionError):
    """Raised when a URL or resolved IP targets a restricted network destination."""


def is_blocked_ip(raw_ip: str) -> bool:
    """Return True if the IP literal is private, loopback, link-local, metadata, or non-global."""
    cleaned = raw_ip.split("%", 1)[0].strip()
    try:
        ip_obj: IPAddress = ipaddress.ip_address(cleaned)
    except ValueError:
        return True

    # Unwrap IPv4-mapped IPv6 addresses (e.g., ::ffff:127.0.0.1)
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        ip_obj = ip_obj.ipv4_mapped

    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
        or not ip_obj.is_global
    ):
        return True

    return any(ip_obj in net for net in _EXTRA_BLOCKED_NETWORKS)


def is_blocked_hostname(hostname: str) -> bool:
    """Return True if a hostname targets local/internal infrastructure or a blocked IP literal."""
    host = hostname.strip().rstrip(".").lower()
    if not host:
        return True

    # Strip IPv6 brackets if present
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]

    if host in _BLOCKED_HOSTNAMES:
        return True
    if any(host.endswith(suffix) for suffix in _BLOCKED_HOSTNAME_SUFFIXES):
        return True

    # Reject IP literals in blocked ranges
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return is_blocked_ip(host)


def validate_url_syntax(
    url: str,
    *,
    allowed_ports: list[int] | tuple[int, ...] = (80, 443),
) -> str:
    """Validate scheme, host, credentials, and port before outbound network use."""
    if not isinstance(url, str):
        raise URLValidationError("URL must be a string.")
    stripped = url.strip()
    if not stripped or len(stripped) > 2048:
        raise URLValidationError("URL is empty or exceeds 2048 characters.")
    if _CONTROL_CHARS_RE.search(stripped):
        raise URLValidationError("URL contains invalid control characters.")

    parsed = urlparse(stripped)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise URLValidationError(f"Unsupported URL scheme: {scheme or '<empty>'}")

    if parsed.username is not None or parsed.password is not None:
        raise URLValidationError("URLs with embedded credentials are not permitted.")

    hostname = parsed.hostname
    if not hostname:
        raise URLValidationError("URL is missing a valid hostname.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise URLValidationError("URL contains an invalid port.") from exc

    effective_port = port if port is not None else (443 if scheme == "https" else 80)
    if effective_port not in allowed_ports:
        raise SSRFError(f"Destination port {effective_port} is not in allowed ports {list(allowed_ports)}.")

    if is_blocked_hostname(hostname):
        raise SSRFError(f"Blocked destination host: {hostname}")

    return stripped


async def default_dns_resolver(hostname: str, port: int) -> list[str]:
    """Resolve hostname to unique IP address strings using the system resolver."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(
        hostname,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
    )
    ips: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            ip_str = str(sockaddr[0])
            if ip_str not in ips:
                ips.append(ip_str)
    return ips


async def validate_url_for_fetch(
    url: str,
    *,
    allowed_ports: list[int] | tuple[int, ...] = (80, 443),
    resolver: AsyncResolver | None = None,
) -> tuple[str, list[str]]:
    """Validate URL syntax and verify all DNS-resolved IPs are globally routable."""
    clean_url = validate_url_syntax(url, allowed_ports=allowed_ports)
    parsed = urlparse(clean_url)
    hostname = (parsed.hostname or "").strip()
    scheme = parsed.scheme.lower()
    port = parsed.port or (443 if scheme == "https" else 80)

    # Literal IP already validated by validate_url_syntax -> is_blocked_hostname
    try:
        ipaddress.ip_address(hostname.split("%", 1)[0])
        return clean_url, [hostname]
    except ValueError:
        pass

    resolve_fn = resolver or default_dns_resolver
    try:
        resolved_ips = await resolve_fn(hostname, port)
    except OSError as exc:
        raise SSRFError(f"DNS resolution failed for host '{hostname}': {exc}") from exc

    if not resolved_ips:
        raise SSRFError(f"DNS resolution returned no addresses for host '{hostname}'.")

    for ip_str in resolved_ips:
        if is_blocked_ip(ip_str):
            raise SSRFError(
                f"DNS resolution for '{hostname}' returned restricted IP address '{ip_str}'."
            )

    return clean_url, resolved_ips


def canonicalize_url(url: str) -> str:
    """Return a deterministic canonical representation of an HTTP(S) URL."""
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        port = None

    if port is None or (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        netloc = f"[{hostname}]" if ":" in hostname else hostname
    else:
        host_part = f"[{hostname}]" if ":" in hostname else hostname
        netloc = f"{host_part}:{port}"

    path = _MULTI_SLASH_RE.sub("/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    filtered_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAM_EXACT
        and not k.lower().startswith(_TRACKING_PARAM_PREFIXES)
    ]
    filtered_params.sort(key=lambda item: (item[0], item[1]))
    query = urlencode(filtered_params, doseq=True)

    return urlunparse((scheme, netloc, path, "", query, ""))


def extract_domain(url: str) -> str:
    """Return normalized hostname without leading 'www.'."""
    hostname = (urlparse(url).hostname or "").lower().rstrip(".")
    if hostname.startswith("www.") and len(hostname) > 4:
        return hostname[4:]
    return hostname


class DomainRateLimiter:
    """Async per-domain interval limiter preventing burst requests against one host."""

    def __init__(self, min_interval_seconds: float = 0.5) -> None:
        self._min_interval = max(0.0, min_interval_seconds)
        self._last_request: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, domain: str) -> asyncio.Lock:
        lock = self._locks.get(domain)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[domain] = lock
        return lock

    async def acquire(self, url_or_domain: str) -> None:
        """Wait until the minimum per-domain interval has elapsed."""
        if self._min_interval <= 0:
            return
        domain = (
            extract_domain(url_or_domain)
            if "://" in url_or_domain
            else url_or_domain.lower().strip()
        )
        lock = self._get_lock(domain)
        async with lock:
            now = time.monotonic()
            last = self._last_request.get(domain)
            if last is not None:
                elapsed = now - last
                if elapsed < self._min_interval:
                    await asyncio.sleep(self._min_interval - elapsed)
            self._last_request[domain] = time.monotonic()
