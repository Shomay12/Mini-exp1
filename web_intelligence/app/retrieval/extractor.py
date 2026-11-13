"""Primary Trafilatura + BeautifulSoup webpage content and metadata extractor."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse

import trafilatura
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from web_intelligence.app.core.security import canonicalize_url, validate_url_syntax
from web_intelligence.app.schemas.documents import DocumentHeading

_MIN_VALID_DATE = datetime(1990, 1, 1, tzinfo=timezone.utc)
_SPACE_RE = re.compile(r"\s+")


@dataclass
class ExtractedPage:
    """Structured content and bibliographic fields extracted from raw HTML or text."""

    title: str | None = None
    author: str | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    canonical_url: str | None = None
    language: str | None = None
    description: str | None = None
    text: str = ""
    headings: list[DocumentHeading] = field(default_factory=list)
    extractor_used: str | None = None
    warnings: list[str] = field(default_factory=list)


def validate_publication_date(
    raw_value: str | None,
    *,
    now: datetime | None = None,
) -> tuple[datetime | None, str | None]:
    """Validate and parse publication date strictly without inventing missing dates."""
    if not raw_value or not isinstance(raw_value, str):
        return None, None
    cleaned = raw_value.strip()
    if not cleaned:
        return None, None

    try:
        parsed = date_parser.parse(cleaned, fuzzy=False)
    except (ValueError, TypeError, OverflowError):
        return None, f"Ignored unparseable publication date: '{cleaned}'"

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)

    reference_now = now or datetime.now(timezone.utc)
    max_allowed = reference_now + timedelta(days=2)
    if parsed < _MIN_VALID_DATE or parsed > max_allowed:
        return None, f"Rejected implausible publication date: '{cleaned}'"

    return parsed, None


def _clean_text_field(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    cleaned = _SPACE_RE.sub(" ", value).strip()
    return cleaned or None


def _meta_content(soup: BeautifulSoup, *selectors: tuple[str, str]) -> str | None:
    for attr_name, attr_val in selectors:
        tag = soup.find("meta", attrs={attr_name: attr_val})
        if tag is not None:
            content = tag.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _extract_soup_fallback_text(soup: BeautifulSoup) -> str:
    """Extract clean structured blocks using BeautifulSoup when Trafilatura yields empty text."""
    for bad_tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "aside"]):
        bad_tag.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks: list[str] = []
    for elem in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = _SPACE_RE.sub(" ", elem.get_text(" ", strip=True)).strip()
        if len(text) < 3:
            continue
        if elem.name in {"h1", "h2", "h3", "h4"}:
            level = int(elem.name[1])
            blocks.append(f"{'#' * level} {text}")
        else:
            blocks.append(text)

    # Deduplicate consecutive identical blocks
    deduped: list[str] = []
    for block in blocks:
        if not deduped or deduped[-1] != block:
            deduped.append(block)
    return "\n\n".join(deduped)


class ContentExtractor:
    """Extracts main content and metadata using Trafilatura primary + BeautifulSoup fallback."""

    def extract(
        self,
        raw_bytes: bytes,
        *,
        final_url: str,
        content_type: str | None = None,
    ) -> ExtractedPage:
        """Decode raw response bytes and extract clean content and metadata."""
        if not raw_bytes:
            return ExtractedPage(warnings=["Empty response body."])

        decoded = raw_bytes.decode("utf-8", errors="replace")
        if content_type and content_type.startswith("text/plain"):
            cleaned_plain = decoded.strip()
            return ExtractedPage(
                text=cleaned_plain,
                extractor_used="plain_text",
                warnings=[] if cleaned_plain else ["Empty plain text body."],
            )

        warnings: list[str] = []

        # Parse structural HTML with BeautifulSoup (lxml parser, html.parser fallback)
        try:
            soup = BeautifulSoup(decoded, "lxml")
        except Exception:
            soup = BeautifulSoup(decoded, "html.parser")

        html_title = _clean_text_field(
            soup.title.get_text(" ", strip=True) if soup.title else None
        ) or _clean_text_field(
            _meta_content(soup, ("property", "og:title"), ("name", "twitter:title"))
        )
        html_author = _clean_text_field(
            _meta_content(
                soup,
                ("name", "author"),
                ("property", "article:author"),
                ("name", "dc.creator"),
            )
        )
        html_publisher = _clean_text_field(
            _meta_content(
                soup,
                ("property", "og:site_name"),
                ("name", "publisher"),
                ("name", "application-name"),
            )
        )
        html_description = _clean_text_field(
            _meta_content(
                soup,
                ("name", "description"),
                ("property", "og:description"),
            )
        )
        html_lang = None
        if soup.html and isinstance(soup.html.get("lang"), str):
            html_lang = soup.html["lang"].strip() or None

        raw_html_date = _meta_content(
            soup,
            ("property", "article:published_time"),
            ("name", "pubdate"),
            ("name", "publishdate"),
            ("name", "date"),
            ("name", "dc.date"),
            ("itemprop", "datePublished"),
        )
        if raw_html_date is None:
            time_tag = soup.find("time", attrs={"datetime": True})
            if time_tag and isinstance(time_tag.get("datetime"), str):
                raw_html_date = time_tag["datetime"].strip()

        # Extract canonical URL if valid HTTP(S) and same/safe origin
        canonical_url: str | None = None
        link_canon = soup.find("link", attrs={"rel": lambda r: r and "canonical" in r})
        if link_canon and isinstance(link_canon.get("href"), str):
            candidate = urljoin(final_url, link_canon["href"].strip())
            try:
                validate_url_syntax(candidate)
                if urlparse(candidate).scheme in {"http", "https"}:
                    canonical_url = canonicalize_url(candidate)
            except Exception:
                warnings.append("Ignored invalid canonical link tag.")

        # Capture HTML headings for structure preservation
        headings: list[DocumentHeading] = []
        heading_root = soup.find("article") or soup.find("main") or soup.body or soup
        if heading_root:
            for tag in heading_root.find_all(["h1", "h2", "h3", "h4"]):
                h_text = _clean_text_field(tag.get_text(" ", strip=True))
                if h_text:
                    headings.append(DocumentHeading(level=int(tag.name[1]), text=h_text, char_offset=0))

        # Primary extraction via Trafilatura
        traf_doc = None
        try:
            traf_doc = trafilatura.bare_extraction(
                decoded,
                url=final_url,
                with_metadata=True,
                include_comments=False,
                include_tables=True,
            )
        except Exception as exc:
            warnings.append(f"Trafilatura failed ({exc}); used BeautifulSoup fallback.")

        traf_text = _clean_block_text(traf_doc.text) if traf_doc and traf_doc.text else ""
        traf_title = _clean_text_field(traf_doc.title) if traf_doc else None
        traf_author = _clean_text_field(traf_doc.author) if traf_doc else None
        traf_sitename = _clean_text_field(traf_doc.sitename) if traf_doc else None
        traf_desc = _clean_text_field(traf_doc.description) if traf_doc else None
        traf_lang = _clean_text_field(traf_doc.language) if traf_doc else None
        traf_date_raw = _clean_text_field(traf_doc.date) if traf_doc else None

        extractor_used = "trafilatura"
        final_text = traf_text
        if len(final_text) < 40:
            bs_text = _extract_soup_fallback_text(soup)
            if len(bs_text) > len(final_text):
                final_text = bs_text
                extractor_used = "beautifulsoup_fallback"
                if not traf_text:
                    warnings.append("Trafilatura yielded empty text; fell back to BeautifulSoup.")

        # Validate publication date strictly (prefer explicit HTML meta date, then Trafilatura date)
        published_at: datetime | None = None
        for candidate_date in (raw_html_date, traf_date_raw):
            if candidate_date:
                parsed_dt, date_warn = validate_publication_date(candidate_date)
                if date_warn:
                    warnings.append(date_warn)
                if parsed_dt is not None:
                    published_at = parsed_dt
                    break

        return ExtractedPage(
            title=html_title or traf_title,
            author=html_author or traf_author,
            publisher=html_publisher or traf_sitename,
            published_at=published_at,
            canonical_url=canonical_url,
            language=html_lang or traf_lang,
            description=html_description or traf_desc,
            text=final_text,
            headings=headings,
            extractor_used=extractor_used if final_text else None,
            warnings=warnings,
        )


def _clean_block_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    for line in lines:
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)
