#!/usr/bin/env python3
"""Generate root-level redirect stubs for docs pages that moved to /docs/.

The docs used to be published at the site root. This deploy moves them to
/docs/ and puts the landing page at the root instead. Anyone with an old
bookmark or external link to a root-level docs page (e.g. .../getting_started/)
would otherwise hit a 404.

This reads the sitemap mkdocs just built (_site/docs/sitemap.xml) and, for
every page except the docs' own home page, writes a tiny HTML redirect at
the pre-move root-level path (e.g. _site/getting_started/index.html)
pointing at its new /docs/... location. Driven by the sitemap rather than a
hand-maintained list, so it covers mkdocstrings' auto-generated API pages
too, and stays correct as docs pages are added, renamed, or removed.

Run after both "mkdocs build" and the landing page copy, so _site/ already
has its final root-level contents and this can safely skip any path that's
already occupied (e.g. the landing page's own index.html).
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlsplit

# The sitemap we parse below is one mkdocs just generated from our own docs
# source in this same job, not untrusted input - but defusedxml (a drop-in
# replacement for xml.etree.ElementTree that disables external entity
# resolution and entity-expansion bombs) costs nothing here, so use it.
import defusedxml.ElementTree as ET

SITE_ROOT = Path("_site")
SITEMAP = SITE_ROOT / "docs" / "sitemap.xml"
DOCS_PREFIX = "/docs/"

REDIRECT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Redirecting…</title>
<link rel="canonical" href="{target}">
<meta http-equiv="refresh" content="0; url={target}">
</head>
<body>
<p>This page has moved to <a href="{target}">{target}</a>.</p>
</body>
</html>
"""


def old_path_for(loc: str) -> Path | None:
    """Map a sitemap <loc> URL to the pre-move root-relative file path.

    Returns None for the docs home page itself, which isn't redirected —
    the landing page intentionally now owns the root.
    """
    path = urlsplit(loc).path  # e.g. /mattext/docs/getting_started/

    idx = path.find(DOCS_PREFIX)
    if idx == -1:
        return None

    after = path[idx + len(DOCS_PREFIX) :]  # e.g. "getting_started/" or ""
    if after in ("", "index.html"):
        return None

    old_rel = after if after.endswith(".html") else f"{after.rstrip('/')}/index.html"
    return SITE_ROOT / old_rel


def main() -> int:
    if not SITEMAP.exists():
        print(f"no sitemap at {SITEMAP}, skipping redirect generation")
        return 0

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    tree = ET.parse(SITEMAP)

    written, skipped = 0, 0
    for url in tree.getroot().findall("sm:url/sm:loc", ns):
        loc = (url.text or "").strip()
        if not loc:
            continue

        old_path = old_path_for(loc)
        if old_path is None:
            continue

        if old_path.exists():
            # something (usually the landing page itself) already owns this
            # path — never overwrite it
            print(f"skip (already exists): {old_path}")
            skipped += 1
            continue

        target_path = urlsplit(loc).path
        old_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.write_text(REDIRECT_TEMPLATE.format(target=target_path))
        written += 1

    print(f"wrote {written} redirect stub(s), skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
