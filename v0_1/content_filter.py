"""
AION Structure-Aware Content Filter.
Keeps academic body content; drops title, copyright, author, TOC, index, and front/back matter.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF


# ─────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────

TOC_HEADINGS = [
    r"^\s*contents\s*$",
    r"^\s*table\s+of\s+contents\s*$",
    r"^\s*list\s+of\s+contents\s*$",
    r"^\s*index\s*$",  # some books misuse "Index" for TOC at front
    r"^\s*brief\s+contents\s*$",
]

FRONT_MATTER_HEADINGS = [
    r"^\s*preface\s*$",
    r"^\s*foreword\s*$",
    r"^\s*acknowledg(?:e)?ments?\s*$",
    r"^\s*about\s+the\s+authors?\s*$",
    r"^\s*about\s+this\s+book\s*$",
    r"^\s*dedication\s*$",
    r"^\s*copyright\s*$",
    r"^\s*publisher'?s?\s+note\s*$",
    r"^\s*syllabus\s*$",
    r"^\s*course\s+objectives?\s*$",
    r"^\s*question\s+paper\s+pattern\s*$",
]

BACK_MATTER_HEADINGS = [
    r"^\s*references?\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*further\s+reading\s*$",
    r"^\s*glossary\s*$",
    r"^\s*appendix(\s+[a-z0-9]+)?\s*$",
    r"^\s*index\s*$",  # end index
    r"^\s*answers?\s+to\s+(selected\s+)?exercises\s*$",
]

BODY_CHAPTER_HEADINGS = [
    r"^\s*chapter\s+\d+\b",
    r"^\s*unit\s+\d+\b",
    r"^\s*module\s+\d+\b",
    r"^\s*\d+(\.\d+)*\s+[A-Z].{3,80}$",  # 1.2 Gradient Descent
]

AUTHOR_PUBLISHER_PATTERNS = [
    r"^\s*by\s+[A-Z][a-z]+(\s+[A-Z][a-z]+){0,3}\s*$",
    r"^\s*author[s]?\s*:\s*.+$",
    r"^\s*isbn[-\s]?(10|13)?\s*[: ]\s*[0-9\-xX]+",
    r"^\s*published\s+by\b",
    r"^\s*all\s+rights\s+reserved\b",
    r"^\s*copyright\s*©?\b",
    r"^\s*doi\s*:\s*\S+",
    r"^\s*www\.\S+\s*$",
    r"^\s*email\s*:\s*\S+@\S+",
    r"^\s*edition\s*:\s*\d+",
    r"^\s*pearson|mcgraw\s*hill|wiley|springer|elsevier|oxford|cambridge|phi\s+learning|technical\s+publications|vtu\b",
]

BOILERPLATE_LINES = [
    r"^\s*this\s+page\s+(is\s+)?intentionally\s+left\s+blank\s*$",
    r"^\s*blank\s+page\s*$",
    r"^\s*page\s+\d+\s+of\s+\d+\s*$",
]


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}", line.strip()))


def _matches_any(line: str, patterns: List[str]) -> bool:
    return any(re.search(p, line, flags=re.I) for p in patterns)


@dataclass
class TocEntry:
    title: str
    page: Optional[int]
    level: int = 1


@dataclass
class PageInfo:
    page_number: int  # 1-based PDF page index
    text: str
    lines: List[str]
    page_class: str = "unknown"  # front_matter/toc/body/back_matter/unknown
    heading: Optional[str] = None


@dataclass
class FilterReport:
    total_pages: int
    kept_pages: List[int]
    dropped_pages: Dict[str, List[int]]
    toc_found: bool
    toc_entries: List[Dict]
    kept_word_count: int
    dropped_word_count: int
    method: str


class AcademicContentFilter:
    """
    Structure-aware filter for textbooks/notes/PPT-exported text.
    """

    def __init__(
        self,
        keep_appendices: bool = False,
        keep_exercises: bool = True,
        max_toc_scan_pages: int = 40,
        min_body_line_len: int = 3,
    ):
        self.keep_appendices = keep_appendices
        self.keep_exercises = keep_exercises
        self.max_toc_scan_pages = max_toc_scan_pages
        self.min_body_line_len = min_body_line_len

    def filter_pdf(self, pdf_path: str) -> Tuple[str, FilterReport]:
        pages = self._read_pdf_pages(pdf_path)
        return self._filter_pages(pages)

    def filter_text_pages(self, pages_text: List[str]) -> Tuple[str, FilterReport]:
        pages = []
        for i, t in enumerate(pages_text, start=1):
            lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
            pages.append(PageInfo(page_number=i, text=t, lines=lines))
        return self._filter_pages(pages)

    def _read_pdf_pages(self, pdf_path: str) -> List[PageInfo]:
        doc = fitz.open(pdf_path)
        pages: List[PageInfo] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            pages.append(PageInfo(page_number=i, text=text, lines=lines))
        doc.close()
        return pages

    def _filter_pages(self, pages: List[PageInfo]) -> Tuple[str, FilterReport]:
        if not pages:
            report = FilterReport(0, [], {}, False, [], 0, 0, "empty")
            return "", report

        # 1) Classify pages
        for p in pages:
            p.page_class, p.heading = self._classify_page(p)

        # 2) Find TOC page(s)
        toc_pages = [p for p in pages if p.page_class == "toc"]
        if not toc_pages:
            for p in pages[: self.max_toc_scan_pages]:
                if self._looks_like_toc_page(p):
                    p.page_class = "toc"
                    toc_pages.append(p)

        toc_entries: List[TocEntry] = []
        method = "body_heuristic"

        if toc_pages:
            toc_text_lines: List[str] = []
            for tp in toc_pages[:4]:
                toc_text_lines.extend(tp.lines)
            toc_entries = self._parse_toc(toc_text_lines)

        kept_idx = set()
        dropped = defaultdict(list)

        if toc_entries and any(e.page is not None for e in toc_entries):
            method = "toc_guided"
            page_map = self._map_printed_to_pdf_pages(pages, toc_entries)
            ranges = self._toc_to_ranges(toc_entries, fallback_last_printed=self._guess_last_printed(pages))

            for start, end, title in ranges:
                pdf_start = page_map.get(start)
                pdf_end = page_map.get(end)
                if pdf_start is None or pdf_end is None:
                    continue
                if not self.keep_appendices and re.search(r"appendix", title or "", re.I):
                    for pn in range(pdf_start, pdf_end + 1):
                        dropped["appendix"].append(pn)
                    continue
                for pn in range(pdf_start, pdf_end + 1):
                    kept_idx.add(pn)

        # ── If toc_guided kept very few pages, fall back to body_heuristic ──
        if method == "toc_guided" and len(kept_idx) < max(10, len(pages) // 10):
            print(f"[CONTENT_FILTER] TOC mapping kept only {len(kept_idx)}/{len(pages)} pages — falling back to body heuristic")
            kept_idx = set()
            dropped = defaultdict(list)
            method = "body_heuristic"

            for p in pages:
                if p.page_class == "body":
                    kept_idx.add(p.page_number)
                elif p.page_class == "toc":
                    dropped["toc"].append(p.page_number)
                elif p.page_class == "front_matter":
                    dropped["front_matter"].append(p.page_number)
                elif p.page_class == "back_matter":
                    dropped["back_matter"].append(p.page_number)
                else:
                    if self._is_dense_academic(p):
                        kept_idx.add(p.page_number)
                    else:
                        dropped["unknown_noise"].append(p.page_number)
        else:
            if method != "toc_guided":
                for p in pages:
                    if p.page_class == "body":
                        kept_idx.add(p.page_number)
                    elif p.page_class == "toc":
                        dropped["toc"].append(p.page_number)
                    elif p.page_class == "front_matter":
                        dropped["front_matter"].append(p.page_number)
                    elif p.page_class == "back_matter":
                        dropped["back_matter"].append(p.page_number)
                    else:
                        if self._is_dense_academic(p):
                            kept_idx.add(p.page_number)
                        else:
                            dropped["unknown_noise"].append(p.page_number)

        for p in pages:
            if p.page_number in kept_idx and p.page_class in {"front_matter", "toc"}:
                kept_idx.discard(p.page_number)
                dropped[p.page_class].append(p.page_number)
            if p.page_number in kept_idx and p.page_class == "back_matter":
                if not self.keep_appendices or not re.search(r"appendix", p.heading or "", re.I):
                    kept_idx.discard(p.page_number)
                    dropped["back_matter"].append(p.page_number)

        if not kept_idx:
            method = "emergency_middle_slice"
            n = len(pages)
            lo = max(1, int(n * 0.1))
            hi = max(lo, int(n * 0.9))
            kept_idx = set(range(lo, hi + 1))

        kept_pages = [p for p in pages if p.page_number in kept_idx]
        ban_lines = self._detect_repeated_header_footer(kept_pages)

        kept_parts = []
        kept_words = 0
        dropped_words = 0

        for p in pages:
            raw_words = len(p.text.split())
            if p.page_number not in kept_idx:
                dropped_words += raw_words
                if p.page_number not in sum(dropped.values(), []):
                    dropped[p.page_class or "other"].append(p.page_number)
                continue

            filtered_lines = self._filter_lines(p.lines, ban_lines)
            if not filtered_lines:
                dropped_words += raw_words
                dropped["empty_after_line_filter"].append(p.page_number)
                continue

            chunk = "\n".join(filtered_lines).strip()
            kept_parts.append(f"[PDF p.{p.page_number}]\n{chunk}")
            kept_words += len(chunk.split())

        final_text = "\n\n".join(kept_parts)
        report = FilterReport(
            method=method,
            total_pages=len(pages),
            kept_pages=sorted(list(kept_idx)),
            dropped_pages={k: sorted(v) for k, v in dropped.items() if v},
            kept_word_count=kept_words,
            dropped_word_count=dropped_words,
            toc_found=bool(toc_entries),
            toc_entries=[asdict(e) for e in toc_entries],
        )
        return final_text, report

    def _classify_page(self, page: PageInfo) -> Tuple[str, Optional[str]]:
        if not page.lines:
            return "unknown", None

        top = page.lines[:8]
        first = page.lines[0]

        if self._looks_like_toc_page(page):
            return "toc", first

        for ln in top:
            if _matches_any(ln, FRONT_MATTER_HEADINGS):
                return "front_matter", ln

        for ln in top:
            if _matches_any(ln, BACK_MATTER_HEADINGS):
                if re.search(r"^\s*index\s*$", ln, re.I) and page.page_number <= 20:
                    if self._looks_like_toc_page(page):
                        return "toc", ln
                return "back_matter", ln

        if page.page_number <= 5 and self._looks_like_title_page(page):
            return "front_matter", first

        if any(_matches_any(ln, BODY_CHAPTER_HEADINGS) for ln in top):
            return "body", first

        if self._is_dense_academic(page):
            return "body", None

        if self._looks_like_references(page):
            return "back_matter", first

        return "unknown", None

    def _looks_like_toc_page(self, page: PageInfo) -> bool:
        if not page.lines:
            return False
        head = "\n".join(page.lines[:5])
        has_toc_heading = _matches_any(head, TOC_HEADINGS)

        leader_hits = 0
        eol_page_hits = 0
        for ln in page.lines:
            if re.search(r"\.{3,}\s*\d+\s*$", ln) or re.search(r"\s{2,}\d+\s*$", ln):
                leader_hits += 1
            if re.search(r"\d+\s*$", ln) and len(ln) < 120:
                eol_page_hits += 1

        chapter_hits = sum(1 for ln in page.lines if re.search(r"\b(chapter|unit|module)\s+\d+", ln, re.I))

        if has_toc_heading and (leader_hits >= 3 or chapter_hits >= 2):
            return True
        if leader_hits >= 8 and page.page_number <= self.max_toc_scan_pages:
            return True
        if chapter_hits >= 5 and eol_page_hits >= 5 and page.page_number <= self.max_toc_scan_pages:
            return True
        return False

    def _looks_like_title_page(self, page: PageInfo) -> bool:
        lines = page.lines
        if len(lines) <= 12 and sum(len(x) for x in lines) < 500:
            author_hits = sum(1 for ln in lines if _matches_any(ln, AUTHOR_PUBLISHER_PATTERNS))
            if author_hits >= 1:
                return True
            caps = sum(1 for ln in lines if ln == ln.title() or ln.isupper())
            if caps >= max(3, len(lines) // 2):
                return True
        if sum(1 for ln in lines if _matches_any(ln, AUTHOR_PUBLISHER_PATTERNS)) >= 2:
            return True
        return False

    def _looks_like_references(self, page: PageInfo) -> bool:
        hits = 0
        for ln in page.lines:
            if re.search(r"\b(19|20)\d{2}\b", ln) and re.search(r"[\.,]\s*[A-Z]", ln):
                hits += 1
            if re.search(r"^\s*\[\d+\]\s+", ln) or re.search(r"^\s*\d+\.\s+[A-Z][a-z]+,", ln):
                hits += 1
        return hits >= 6

    def _is_dense_academic(self, page: PageInfo) -> bool:
        words = len(page.text.split())
        if words < 40:
            return False
        signals = 0
        if re.search(r"[=∑∫√≤≥±→←]|\\frac|\\sum", page.text):
            signals += 1
        if re.search(r"\b(definition|theorem|lemma|proof|example|figure|table)\b", page.text, re.I):
            signals += 1
        if words > 120:
            signals += 1
        return signals >= 1 and words >= 60

    def _parse_toc(self, lines: List[str]) -> List[TocEntry]:
        entries: List[TocEntry] = []

        for ln in lines:
            if _matches_any(ln, TOC_HEADINGS):
                continue
            if not ln.strip():
                continue

            # ── Format 1: Standard (title first, page at end) ──
            # "Chapter 8 Hashing .......... 395"
            m = re.search(
                r"^(?P<title>.+?)\s+(?:\.{2,}|-{2,}|\s{2,})\s*(?P<page>\d{1,4})\s*$",
                ln
            )

            # ── Format 2: Reversed (page first, title after) ──
            # "395  8.1  Hash Tables"
            if not m:
                m2 = re.match(
                    r"^\s*(?P<page>\d{1,4})\s{2,}(?P<title>[A-Z0-9].{3,80})\s*$",
                    ln
                )
                if m2:
                    title = m2.group("title").strip()
                    page  = int(m2.group("page"))
                    if 1 <= page <= 5000 and len(title) >= 3:
                        level = 2 if re.match(r"\d+\.\d+", title) else 1
                        entries.append(TocEntry(title=title, page=page, level=level))
                    continue

            # ── Format 3: Simple "title  page" with no leader dots ──
            if not m:
                m = re.search(r"^(?P<title>.{5,100}?)\s{2,}(?P<page>\d{1,4})\s*$", ln)
                if m:
                    title_cand = m.group("title").strip()
                    if len(title_cand) < 3 or len(title_cand) > 120:
                        continue
                    if _matches_any(title_cand, AUTHOR_PUBLISHER_PATTERNS):
                        continue
                    if not re.search(r"[A-Za-z]{3,}", title_cand):
                        continue

            if not m:
                continue

            title = re.sub(r"\s+", " ", m.group("title")).strip(" .-:\t")
            page  = int(m.group("page"))

            if page <= 0 or page > 5000:
                continue

            level = 1
            if re.match(r"\d+\.\d+(\.\d+)?", title):
                level = 2
            elif re.match(r"(chapter|unit|module)\s+\d+", title, re.I):
                level = 1

            entries.append(TocEntry(title=title, page=page, level=level))

        # De-duplicate
        seen, uniq = set(), []
        for e in entries:
            key = (_norm(e.title), e.page)
            if key not in seen:
                seen.add(key)
                uniq.append(e)
        return uniq

    def _toc_to_ranges(self, entries: List[TocEntry], fallback_last_printed: int) -> List[Tuple[int, int, str]]:
        usable = [e for e in entries if e.page is not None]
        top = [e for e in usable if e.level == 1]
        if len(top) < 2:
            top = usable
        top = sorted(top, key=lambda e: e.page)

        ranges = []
        for i, e in enumerate(top):
            start = e.page
            end = (top[i + 1].page - 1) if i + 1 < len(top) else max(start, fallback_last_printed)
            if end < start:
                end = start
            ranges.append((start, end, e.title))
        return ranges

    def _guess_last_printed(self, pages: List[PageInfo]) -> int:
        nums = []
        for p in pages[int(len(pages) * 0.6):]:
            for ln in p.lines[-5:]:
                if _is_page_number(ln):
                    nums.append(int(ln.strip()))
        return max(nums) if nums else max(1, len(pages))

    def _map_printed_to_pdf_pages(self, pages: List[PageInfo], toc_entries: List[TocEntry]) -> Dict[int, int]:
        """
        Build printed_page -> pdf_page_index mapping.
        Handles books with blank early pages and reversed TOC format.
        """

        # ── Step 1: Try offset from chapter title text matching ──
        offset = None
        for e in sorted(toc_entries, key=lambda x: x.page or 0):
            if not e.page:
                continue
            token = re.sub(
                r"^\s*(chapter|unit|module|section)\s+[\d\.]+[:.\s\-]*",
                "", e.title, flags=re.I
            ).strip()
            token_full = e.title.strip()

            for search_token in [token, token_full]:
                search_token = search_token[:50].lower().strip()
                if len(search_token) < 4:
                    continue
                for p in pages:
                    if search_token in p.text.lower():
                        candidate_offset = p.page_number - e.page
                        if 0 <= candidate_offset <= len(pages):
                            offset = candidate_offset
                            break
                if offset is not None:
                    break
            if offset is not None:
                break

        # ── Step 2: If no offset found, try footer number correlation ──
        if offset is None:
            offsets_found = []
            for p in pages[5:min(50, len(pages))]:   # skip first few blank pages
                probe = p.lines[:2] + p.lines[-2:]
                for ln in probe:
                    if _is_page_number(ln):
                        printed = int(ln.strip())
                        if printed > 0:
                            candidate = p.page_number - printed
                            if 0 <= candidate <= len(pages):
                                offsets_found.append(candidate)
            if offsets_found:
                from collections import Counter
                offset = Counter(offsets_found).most_common(1)[0][0]

        # ── Step 3: Build the mapping ──
        mapping: Dict[int, int] = {}

        if offset is not None:
            last_printed = self._guess_last_printed(pages)
            max_toc_page = max([e.page for e in toc_entries if e.page] + [1])
            upper = max(last_printed, max_toc_page, len(pages) - offset)

            for printed in range(1, upper + 1):
                pdf_p = printed + offset
                if 1 <= pdf_p <= len(pages):
                    mapping[printed] = pdf_p
        else:
            for p in pages:
                mapping[p.page_number] = p.page_number

        return mapping

    def _detect_repeated_header_footer(self, pages: List[PageInfo]) -> set:
        counter = Counter()
        for p in pages:
            probe = p.lines[:2] + p.lines[-2:]
            for ln in probe:
                n = _norm(ln)
                if len(n) < 4 or _is_page_number(n):
                    continue
                counter[n] += 1
        ban = set()
        thresh = max(3, len(pages) // 4)
        for k, v in counter.items():
            if v >= thresh:
                ban.add(k)
        return ban

    def _filter_lines(self, lines: List[str], ban_lines: set) -> List[str]:
        out = []
        for ln in lines:
            s = ln.strip()

            if len(s) < self.min_body_line_len:
                continue
            if _is_page_number(s):
                continue
            if _norm(s) in ban_lines:
                continue
            if _matches_any(s, BOILERPLATE_LINES):
                continue
            if _matches_any(s, AUTHOR_PUBLISHER_PATTERNS):
                continue
            if re.fullmatch(r"(chapter|unit|module)\s+\d+", s, flags=re.I):
                continue
            if not self.keep_exercises and re.match(r"^\s*(exercises|review questions|practice problems)\b", s, re.I):
                break

            # ── Code / path / config line filters ──
            if re.match(r"^\s*(import|from)\s+\w+", s):
                continue
            if re.match(r"^\s*\w+[/\\]\w+[/\\]?\w*\s*$", s):
                continue
            if re.match(r"^\s*[A-Z_]{4,}\s*=\s*", s):
                continue
            if re.match(r"^\s*\w+\s*=\s*[\[\{\"\'\w<]", s) and len(s) < 80:
                continue
            if re.match(r"^\s*<[a-zA-Z][^>]{0,60}>", s):
                continue
            if re.match(r"^\s*https?://\S+\s*$", s):
                continue
            if re.match(r"^\s*[\w_\-]+\.(py|html|js|css|json|yaml|txt|md|cfg|ini)\s*$", s):
                continue
            if re.search(r"\{[{%].*?[%}]\}", s):
                continue
            if len(s) < 25 and not re.match(r"^[A-Z]", s):
                continue

            out.append(ln)

        merged = []
        for ln in out:
            if merged and merged[-1].endswith("-") and ln[:1].islower():
                merged[-1] = merged[-1][:-1] + ln
            else:
                merged.append(ln)
        return merged


def extract_academic_content(pdf_path: str, material_type: str = "textbook") -> Tuple[str, dict]:
    """
    material_type: "textbook" | "notes" | "question_bank" | "slides"
    (case-insensitive)
    """
    mt = material_type.lower()

    filt = AcademicContentFilter(
        keep_appendices=False,
        keep_exercises=(mt != "textbook"),
        max_toc_scan_pages=50 if mt == "textbook" else 10,
    )

    if mt == "textbook":
        text, report = filt.filter_pdf(pdf_path)
    else:
        doc = fitz.open(pdf_path)
        pages = [(page.get_text("text") or "") for page in doc]
        doc.close()
        text, report = filt.filter_text_pages(pages)

    return text, asdict(report)
