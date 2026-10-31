"""
=============================================================
  LOCAL FILE SEARCH ENGINE  —  BM25 + TF-IDF  (No AI)
=============================================================
  How it works:
    1. Indexes every .md / .txt / .pdf / .py file in a folder
    2. For each query, ranks documents using BM25 scoring
    3. Returns top-N results with highlighted snippet excerpts
  
  Dependencies (all stdlib except one optional):
    - No external libs required for core BM25
    - 'colorama' for coloured terminal output  (pip install colorama)
    - 'PyPDF2'   for PDF support               (pip install PyPDF2)
  
  Usage:
    python search_engine.py                         # interactive mode
    python search_engine.py --folder ./docs         # custom folder
    python search_engine.py --query "reset password" --top 5
=============================================================
"""

from __future__ import annotations

import os
import re
import sys
import math
import json
import argparse
import textwrap
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

# --- Optional pretty-print deps ---
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

    class _Dummy:
        def __getattr__(self, _): return ""

    Fore = Style = _Dummy()

# --- Optional PDF support ---
try:
    # pyrefly: ignore [missing-import]
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False


# ==============================================================================
#  SECTION 1 - TEXT UTILITIES
# ==============================================================================

STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "this", "that", "be",
    "as", "are", "was", "were", "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "if", "so", "up", "out", "then", "than", "into", "also",
    "i", "we", "you", "he", "she", "they", "me", "us", "him", "her", "them",
    "my", "your", "his", "our", "their", "what", "which", "who", "when",
    "where", "how", "about", "after", "before", "more", "most", "other",
    "some", "such", "any", "all", "each", "been", "being",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, strip markdown/HTML, split, remove stop words."""
    text = re.sub(r"[#*_`~>|]", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    tokens = re.findall(r"[a-z][a-z0-9']*", text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def _finish_stem(word: str) -> str:
    """Make a stem independent of the inflection it was cut from.

    Without this, 'running' -> 'runn' never meets 'run', and 'escalated'
    -> 'escalat' never meets 'escalate'.
    """
    # undo a doubled final consonant ('runn' -> 'run', 'resett' -> 'reset'),
    # but keep 'll', 'ss', 'zz' ('bill', 'access')
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeioulsz":
        word = word[:-1]
    # drop a trailing 'e' so 'escalate' and 'escalat(ed/ion)' share one stem
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def stem(word: str) -> str:
    """Lightweight suffix-stripping stemmer (no external libs).

    Different inflections of one word should reduce to the same stem, so a
    query like "refunded" can find a document that says "refund":

    >>> stem("refund") == stem("refunded") == stem("refunds")
    True
    >>> stem("escalate") == stem("escalated") == stem("escalation")
    True
    >>> stem("delete") == stem("deleted") == stem("deleting") == stem("deletion")
    True
    >>> stem("key") == stem("keys")
    True
    >>> stem("reset") == stem("resetting")
    True
    >>> stem("setting") == stem("settings")
    True

    Words that merely look inflected are left alone:

    >>> stem("access"), stem("status"), stem("speed")
    ('access', 'status', 'speed')
    """
    rules = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("ising", "ise"),
        ("izing", "ize"), ("bles", "ble"), ("ness", ""), ("ment", ""),
        ("ful", ""), ("less", ""), ("ously", ""), ("ings", ""),
        ("ers", "er"), ("ing", ""), ("ies", "y"), ("ied", "y"),
        ("tion", "te"), ("tions", "te"), ("edly", ""), ("ably", ""),
        ("ibly", ""), ("ations", "ate"), ("ators", "ate"),
        ("alism", "al"), ("alists", "al"), ("alize", "al"),
        ("alities", "al"), ("ality", "al"), ("fulness", "ful"),
        ("ousness", "ous"), ("iveness", "ive"), ("ication", "y"),
        ("izations", "ize"), ("ization", "ize"), ("ed", ""),
    ]
    for suffix, replacement in rules:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            if suffix == "ed" and word.endswith("eed"):
                continue  # 'speed', 'need' are not past tenses
            return _finish_stem(word[: len(word) - len(suffix)] + replacement)
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return _finish_stem(word[:-1])
    return _finish_stem(word)


def normalize_tokens(tokens: list[str]) -> list[str]:
    """Apply stemming to every token."""
    return [stem(t) for t in tokens]


def extract_snippet(text: str, query_tokens: list[str], window: int = 200) -> str:
    """Find the best window in text that contains the most query terms."""
    lower = text.lower()
    best_pos, best_score = 0, 0

    for token in query_tokens:
        pos = lower.find(token)
        while pos != -1:
            region = lower[max(0, pos - 50): pos + window]
            score = sum(1 for t in query_tokens if t in region)
            if score > best_score:
                best_score, best_pos = score, max(0, pos - 30)
            pos = lower.find(token, pos + 1)

    snippet = text[best_pos: best_pos + window].strip()
    if len(snippet) >= window - 10:
        last_period = max(snippet.rfind("."), snippet.rfind("!"), snippet.rfind("?"))
        if last_period > window // 2:
            snippet = snippet[: last_period + 1]

    return "..." + snippet + "..." if best_pos > 0 else snippet + "..."


def highlight(text: str, terms: list[str]) -> str:
    """Wrap matched terms in ANSI colour codes."""
    if not HAS_COLOR:
        return text
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(
            lambda m: Fore.YELLOW + Style.BRIGHT + m.group() + Style.RESET_ALL,
            text,
        )
    return text


# ==============================================================================
#  SECTION 2 - DOCUMENT LOADING
# ==============================================================================

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst", ".py", ".json", ".csv", ".pdf"}


def load_file(path: Path) -> str | None:
    """Read a file and return its raw text content."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        if not HAS_PDF:
            print(f"[WARN] Skipping {path.name} - install PyPDF2 for PDF support.")
            return None
        try:
            # pyrefly: ignore [missing-import]
            import PyPDF2
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return " ".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"[ERROR] Could not read PDF {path.name}: {e}")
            return None

    if ext == ".json":
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except Exception:
            pass

    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] Could not read {path.name}: {e}")
        return None


def discover_files(folder: str | Path, recursive: bool = True) -> list[Path]:
    """Walk a folder and collect all supported files."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files: list[Path] = []
    glob = folder.rglob("*") if recursive else folder.glob("*")
    for p in glob:
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            if not any(part.startswith(".") for part in p.parts):
                if p.name != Path(__file__).name:
                    files.append(p)

    return sorted(files)


# ==============================================================================
#  SECTION 3 - BM25 INDEX
# ==============================================================================

# BM25 hyperparameters (Okapi BM25 defaults)
K1 = 1.5   # term frequency saturation
B  = 0.75  # length normalisation factor


class Document:
    """Represents one indexed file."""
    __slots__ = ("path", "raw_text", "tokens", "stem_tokens", "term_freq", "length")

    def __init__(self, path: Path, raw_text: str):
        self.path = path
        self.raw_text = raw_text
        tokens = tokenize(raw_text)
        self.tokens = tokens
        self.stem_tokens = normalize_tokens(tokens)
        self.term_freq: Counter[str] = Counter(self.stem_tokens)
        self.length: int = len(self.stem_tokens)


class BM25Index:
    """
    Inverted index with Okapi BM25 scoring.

    BM25 score for term t in document d:
        IDF(t)  = log((N - df + 0.5) / (df + 0.5) + 1)
        score   = IDF(t) * (tf * (k1+1)) / (tf + k1*(1-b + b*|d|/avgdl))
    """

    def __init__(self):
        self.documents: list[Document] = []
        self.inverted_index: dict[str, list[int]] = defaultdict(list)
        self.df: Counter[str] = Counter()
        self.avgdl: float = 0.0
        self.N: int = 0

    def add_document(self, doc: Document) -> None:
        doc_id = len(self.documents)
        self.documents.append(doc)
        for term in set(doc.stem_tokens):
            self.inverted_index[term].append(doc_id)
            self.df[term] += 1

    def build(self, folder: str | Path, recursive: bool = True) -> None:
        """Discover, load, tokenize and index every file in folder."""
        files = discover_files(folder, recursive)
        if not files:
            print(f"[WARN] No supported files found in {folder}.")
            return

        print(f"\nIndexing {len(files)} file(s)...")
        for path in files:
            raw = load_file(path)
            if raw and raw.strip():
                doc = Document(path, raw)
                self.add_document(doc)
                print(f"  OK  {path.name}  ({doc.length} tokens)")
            else:
                print(f"  --  {path.name}  (empty / unreadable)")

        self.N = len(self.documents)
        self.avgdl = (
            sum(d.length for d in self.documents) / self.N if self.N else 1.0
        )
        print(
            f"\nIndex ready:  {self.N} documents  |  "
            f"{len(self.inverted_index)} unique terms  |  "
            f"avg length {self.avgdl:.1f} tokens\n"
        )

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _bm25_score(self, doc: Document, query_stems: list[str]) -> float:
        score = 0.0
        for term in query_stems:
            tf = doc.term_freq.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            denom = tf + K1 * (1 - B + B * doc.length / self.avgdl)
            score += idf * (tf * (K1 + 1)) / denom
        return score

    def search(self, query: str, top_n: int = 5) -> list[dict[str, Any]]:
        """Run a BM25 query and return top-N results."""
        if self.N == 0:
            return []

        query_tokens = tokenize(query)
        query_stems  = normalize_tokens(query_tokens)

        if not query_stems:
            return []

        # Candidate docs: union of posting lists
        candidate_ids: set[int] = set()
        for term in query_stems:
            candidate_ids.update(self.inverted_index.get(term, []))

        if not candidate_ids:
            return []

        # Score every candidate
        scored: list[tuple[float, int]] = []
        for doc_id in candidate_ids:
            doc   = self.documents[doc_id]
            score = self._bm25_score(doc, query_stems)
            if score > 0:
                scored.append((score, doc_id))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc_id in scored[:top_n]:
            doc     = self.documents[doc_id]
            snippet = extract_snippet(doc.raw_text, query_tokens)
            results.append({
                "filename": doc.path.name,
                "path": str(doc.path),
                "score": round(score, 4),
                "snippet": snippet,
                "rank": len(results) + 1,
            })

        return results


# ==============================================================================
#  SECTION 4 - OUTPUT / DISPLAY
# ==============================================================================

SEPARATOR = "-" * 70


def print_banner() -> None:
    print(Fore.CYAN + Style.BRIGHT + r"""
  +----------------------------------------------------------+
  |          LOCAL FILE SEARCH ENGINE  (BM25)                |
  |          No AI . No Cloud . Pure Keyword Matching        |
  +----------------------------------------------------------+
""" + Style.RESET_ALL)


def print_results(results: list[dict], query: str, query_tokens: list[str]) -> None:
    if not results:
        print(f"\n{Fore.RED}No results found for: \"{query}\"{Style.RESET_ALL}\n")
        return

    print(f"\n{Fore.CYAN}{SEPARATOR}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  Results for:{Style.RESET_ALL} {Style.BRIGHT}\"{query}\"{Style.RESET_ALL}   ({len(results)} match{'es' if len(results) != 1 else ''})")
    print(f"{Fore.CYAN}{SEPARATOR}{Style.RESET_ALL}\n")

    for r in results:
        print(f"  {Fore.GREEN}[#{r['rank']}]{Style.RESET_ALL}  "
              f"{Fore.WHITE}{Style.BRIGHT}{r['filename']}{Style.RESET_ALL}  "
              f"{Fore.YELLOW}score={r['score']:.4f}{Style.RESET_ALL}")
        print(f"         {Fore.CYAN}path:{Style.RESET_ALL} {r['path']}")

        snippet = highlight(r["snippet"], query_tokens)
        wrapped = textwrap.fill(
            snippet, width=66,
            initial_indent="         ",
            subsequent_indent="         "
        )
        print(f"\n{wrapped}\n")
        print(f"  {Fore.CYAN}{SEPARATOR}{Style.RESET_ALL}")

    print()


def print_help() -> None:
    print(f"""
  {Fore.CYAN}Commands:{Style.RESET_ALL}
    <your query>          - search the indexed files
    {Fore.YELLOW}:top <N>{Style.RESET_ALL}              - change number of results shown  (default: 5)
    {Fore.YELLOW}:files{Style.RESET_ALL}                - list all indexed files
    {Fore.YELLOW}:stats{Style.RESET_ALL}                - show index statistics
    {Fore.YELLOW}:reload{Style.RESET_ALL}               - re-index all files
    {Fore.YELLOW}:help{Style.RESET_ALL}                 - show this help message
    {Fore.YELLOW}:quit{Style.RESET_ALL} / {Fore.YELLOW}:exit{Style.RESET_ALL}        - quit
""")


def print_files(index: BM25Index) -> None:
    print(f"\n  {Fore.CYAN}Indexed files ({index.N}):{Style.RESET_ALL}")
    for i, doc in enumerate(index.documents, 1):
        print(f"    {Fore.GREEN}{i:>3}.{Style.RESET_ALL}  {doc.path.name}  "
              f"{Fore.CYAN}({doc.length} tokens){Style.RESET_ALL}")
    print()


def print_stats(index: BM25Index) -> None:
    print(f"""
  {Fore.CYAN}Index Statistics:{Style.RESET_ALL}
    Documents   : {index.N}
    Unique terms: {len(index.inverted_index)}
    Avg length  : {index.avgdl:.1f} tokens
    BM25 k1     : {K1}
    BM25 b      : {B}
""")


# ==============================================================================
#  SECTION 5 - CLI ENTRY POINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local BM25 file search engine - no AI required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python search_engine.py
              python search_engine.py --folder ./docs
              python search_engine.py --query "password reset" --top 3
              python search_engine.py --no-recursive
        """),
    )
    parser.add_argument("--folder", "-f", default=".",
                        help="Folder to index (default: current directory)")
    parser.add_argument("--query", "-q", default=None,
                        help="Run a single query and exit (non-interactive)")
    parser.add_argument("--top", "-n", type=int, default=5,
                        help="Number of results to show (default: 5)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Do not recurse into sub-folders")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable coloured output")
    return parser.parse_args()


def interactive_loop(index: BM25Index, folder: str, top_n: int, recursive: bool) -> None:
    """REPL: read query -> search -> print results -> repeat."""
    print_help()
    current_top = top_n

    while True:
        try:
            raw = input(f"{Fore.GREEN}search> {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}\n")
            break

        if not raw:
            continue

        if raw.startswith(":"):
            cmd = raw.lower()

            if cmd in (":quit", ":exit", ":q"):
                print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}\n")
                break
            elif cmd == ":help":
                print_help()
            elif cmd == ":files":
                print_files(index)
            elif cmd == ":stats":
                print_stats(index)
            elif cmd == ":reload":
                print(f"\n{Fore.CYAN}Re-indexing...{Style.RESET_ALL}")
                index.__init__()
                index.build(folder, recursive)
            elif cmd.startswith(":top"):
                parts = cmd.split()
                if len(parts) == 2 and parts[1].isdigit():
                    current_top = int(parts[1])
                    print(f"  {Fore.CYAN}Top-N set to {current_top}{Style.RESET_ALL}\n")
                else:
                    print(f"  {Fore.RED}Usage: :top <number>{Style.RESET_ALL}\n")
            else:
                print(f"  {Fore.RED}Unknown command '{raw}'. Type :help for help.{Style.RESET_ALL}\n")
            continue

        results = index.search(raw, top_n=current_top)
        query_tokens = tokenize(raw)
        print_results(results, raw, query_tokens)


def main() -> None:
    args = parse_args()

    global HAS_COLOR
    if args.no_color:
        HAS_COLOR = False

    print_banner()

    index = BM25Index()
    folder = args.folder
    recursive = not args.no_recursive

    try:
        index.build(folder, recursive)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if index.N == 0:
        print("[ERROR] No documents were indexed. Nothing to search.")
        sys.exit(1)

    if args.query:
        results = index.search(args.query, top_n=args.top)
        query_tokens = tokenize(args.query)
        print_results(results, args.query, query_tokens)
        return

    interactive_loop(index, folder, args.top, recursive)


if __name__ == "__main__":
    main()
