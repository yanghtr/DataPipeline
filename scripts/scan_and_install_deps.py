"""
Scan one or more JSONL files for Python imports in code blocks,
report which packages are missing, and optionally install them.

Usage:
    python scan_and_install_deps.py <jsonl_or_dir> [<jsonl_or_dir> ...]
    python scan_and_install_deps.py path/to/file.jsonl --install
    python scan_and_install_deps.py path/to/dir/ --install --glob "*.jsonl"
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# import-name  →  pip install name
# ---------------------------------------------------------------------------
IMPORT_TO_PIP = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "wx": "wxPython",
    "gi": "PyGObject",
    "gtk": "PyGObject",
    "Image": "Pillow",
    "Crypto": "pycryptodome",
    "OpenGL": "PyOpenGL",
    "mpl_toolkits": "matplotlib",
    "mpl": "matplotlib",
    "plotly": "plotly",
    "bokeh": "bokeh",
    "altair": "altair",
    "squarify": "squarify",
    "wordcloud": "wordcloud",
    "cartopy": "Cartopy",
    "geopandas": "geopandas",
    "shapely": "shapely",
    "folium": "folium",
    "statsmodels": "statsmodels",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "flask": "Flask",
    "django": "Django",
    "fastapi": "fastapi",
    "requests": "requests",
    "aiohttp": "aiohttp",
    "httpx": "httpx",
    "sqlalchemy": "SQLAlchemy",
    "pymongo": "pymongo",
    "redis": "redis",
    "celery": "celery",
    "pydantic": "pydantic",
    "arrow": "arrow",
    "pendulum": "pendulum",
    "dateutil": "python-dateutil",
    "tqdm": "tqdm",
    "rich": "rich",
    "click": "click",
    "typer": "typer",
    "loguru": "loguru",
    "prettytable": "PrettyTable",
    "tabulate": "tabulate",
    "colorama": "colorama",
    "termcolor": "termcolor",
    "pygments": "Pygments",
    "lxml": "lxml",
    "openpyxl": "openpyxl",
    "xlrd": "xlrd",
    "xlwt": "xlwt",
    "paramiko": "paramiko",
    "fabric": "fabric",
    "psutil": "psutil",
    "attr": "attrs",
    "attrs": "attrs",
    "six": "six",
    "pkg_resources": "setuptools",
    "setuptools": "setuptools",
}

# Python standard library top-level module names (3.x)
STDLIB_MODULES = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else {
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio", "asyncore",
    "atexit", "audioop", "base64", "bdb", "binascii", "binhex", "bisect", "builtins",
    "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd", "code", "codecs",
    "codeop", "colorsys", "compileall", "concurrent", "configparser", "contextlib",
    "contextvars", "copy", "copyreg", "cProfile", "csv", "ctypes", "curses", "dataclasses",
    "datetime", "dbm", "decimal", "difflib", "dis", "distutils", "doctest", "email",
    "encodings", "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
    "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt", "getpass", "gettext",
    "glob", "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http", "idlelib",
    "imaplib", "imghdr", "imp", "importlib", "inspect", "io", "ipaddress", "itertools",
    "json", "keyword", "lib2to3", "linecache", "locale", "logging", "lzma", "mailbox",
    "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder", "multiprocessing",
    "netrc", "nis", "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform", "plistlib",
    "poplib", "posix", "posixpath", "pprint", "profile", "pstats", "pty", "pwd", "py_compile",
    "pyclbr", "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib", "resource",
    "rlcompleter", "runpy", "sched", "secrets", "select", "selectors", "shelve", "shlex",
    "shutil", "signal", "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver",
    "spwd", "sqlite3", "sre_compile", "sre_constants", "sre_parse", "ssl", "stat",
    "statistics", "string", "stringprep", "struct", "subprocess", "sunau", "symtable",
    "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile",
    "termios", "test", "textwrap", "threading", "time", "timeit", "tkinter", "token",
    "tokenize", "trace", "traceback", "tracemalloc", "tty", "turtle", "turtledemo",
    "types", "typing", "unicodedata", "unittest", "urllib", "uu", "uuid", "venv",
    "warnings", "wave", "weakref", "webbrowser", "winreg", "winsound", "wsgiref",
    "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
    "_thread", "__future__",
}

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_IMPORT_RE = re.compile(r"^(?:import|from)\s+([\w]+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def collect_jsonl_files(paths: list[str], glob_pat: str) -> list[Path]:
    files = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob(glob_pat)))
        else:
            print(f"[WARN] Not found: {p}")
    return files


def scan_file(path: Path) -> dict[str, int]:
    """Return {import_name: occurrence_count} for all imports in code blocks."""
    counts: dict[str, int] = defaultdict(int)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for turn in record.get("data", []):
                if turn.get("role") != "assistant":
                    continue
                for item in turn.get("content", []):
                    if item.get("type") != "text":
                        continue
                    raw = item["text"]
                    if isinstance(raw, dict):
                        raw = raw.get("string", "")
                    for m in _CODE_FENCE.finditer(raw):
                        for im in _IMPORT_RE.finditer(m.group(1)):
                            counts[im.group(1)] += 1
    return counts


def is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def pip_install(packages: list[str], dry_run: bool = False) -> dict[str, bool]:
    results = {}
    for pkg in packages:
        if dry_run:
            print(f"  [dry-run] pip install {pkg}")
            results[pkg] = True
            continue
        print(f"  Installing {pkg} ...", end=" ", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True, text=True,
        )
        ok = proc.returncode == 0
        results[pkg] = ok
        print("OK" if ok else f"FAILED\n{proc.stderr.strip()}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scan JSONL code blocks for imports, report and install missing packages."
    )
    parser.add_argument("paths", nargs="+", help="JSONL files or directories")
    parser.add_argument("--glob", default="*.jsonl", help="Glob pattern when scanning a directory (default: *.jsonl)")
    parser.add_argument("--install", action="store_true", help="Install missing packages via pip")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be installed without actually installing")
    args = parser.parse_args()

    files = collect_jsonl_files(args.paths, args.glob)
    if not files:
        sys.exit("No JSONL files found.")

    print(f"Scanning {len(files)} file(s)...\n")

    total_counts: dict[str, int] = defaultdict(int)
    for f in files:
        for name, cnt in scan_file(f).items():
            total_counts[name] += cnt

    if not total_counts:
        print("No import statements found in any code block.")
        return

    # Separate stdlib / third-party / unknown status
    stdlib, present, missing = [], [], []
    for name in sorted(total_counts):
        if name in STDLIB_MODULES:
            stdlib.append(name)
        elif is_installed(name):
            present.append(name)
        else:
            missing.append(name)

    # ---------- Report ----------
    col = 28
    print(f"{'MODULE':<{col}} {'OCCURRENCES':>12}  STATUS")
    print("-" * (col + 26))
    for name in stdlib:
        print(f"  {name:<{col-2}} {total_counts[name]:>12}  stdlib")
    for name in present:
        print(f"  {name:<{col-2}} {total_counts[name]:>12}  installed")
    for name in missing:
        pip_name = IMPORT_TO_PIP.get(name, name)
        note = f"  (pip: {pip_name})" if pip_name != name else ""
        print(f"  {name:<{col-2}} {total_counts[name]:>12}  MISSING{note}")

    print()
    print(f"  stdlib   : {len(stdlib)}")
    print(f"  installed: {len(present)}")
    print(f"  missing  : {len(missing)}")

    if not missing:
        print("\nAll third-party packages are already installed.")
        return

    if not args.install and not args.dry_run:
        print("\nRun with --install to install missing packages.")
        return

    print("\nInstalling missing packages:")
    pip_targets = [IMPORT_TO_PIP.get(m, m) for m in missing]
    pip_targets = list(dict.fromkeys(pip_targets))  # deduplicate, preserve order
    results = pip_install(pip_targets, dry_run=args.dry_run)

    failed = [pkg for pkg, ok in results.items() if not ok]
    if failed:
        print(f"\n[ERROR] Failed to install: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll packages installed successfully.")


if __name__ == "__main__":
    main()
