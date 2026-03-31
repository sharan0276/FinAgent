from collections import Counter
from pathlib import Path

import importlib.util


_UTILS_SPEC = importlib.util.spec_from_file_location(
    "sec_parser_utils", Path(__file__).with_name("sec_parser_utils.py")
)
_UTILS = importlib.util.module_from_spec(_UTILS_SPEC)
assert _UTILS_SPEC.loader is not None
_UTILS_SPEC.loader.exec_module(_UTILS)


TICKER = "META"
FORM = "10-K"
OUTPUT_DIR = _UTILS.OUTPUT_DIR
OUTPUT_FILE = OUTPUT_DIR / "meta_sec_parser_spike_output.txt"
TREE_FILE = OUTPUT_DIR / "meta_sec_parser_tree.txt"


def build_report() -> str:
    parser_name = "Edgar10QParser with Method 2 (10-Q top-section steps removed)"
    html = _UTILS.fetch_filing_html(TICKER, FORM)
    elements, tree, rendered_tree = _UTILS.parse_filing_html(html)
    top_nodes = list(tree.nodes)

    element_counts = Counter(type(element).__name__ for element in elements)
    root_counts = Counter(type(node).__name__ for node in top_nodes)
    top_titles = [node.text for node in top_nodes[:10] if getattr(node, "text", "").strip()]

    lines: list[str] = []
    lines.append(f"{TICKER} latest {FORM} fetched successfully.")
    lines.append(f"Parser used: {parser_name}")
    lines.append("Method: skipped 10-Q-specific top-section parsing steps per sec-parser docs.")
    lines.append(f"HTML length: {len(html):,} characters")
    lines.append(f"Semantic elements parsed: {len(elements):,}")
    lines.append(f"Top-level tree nodes: {len(top_nodes):,}")

    lines.append("")
    lines.append("Element type counts:")
    for name, count in element_counts.most_common(10):
        lines.append(f"- {name}: {count}")

    lines.append("")
    lines.append("Top-level node type counts:")
    for name, count in root_counts.most_common(10):
        lines.append(f"- {name}: {count}")

    lines.append("")
    lines.append("Top-level node preview:")
    for title in top_titles:
        lines.append(f"- {title}")

    lines.append("")
    lines.append("Rendered semantic tree preview:")
    lines.append(_UTILS.safe_console(_UTILS.first_lines(rendered_tree, count=40)))

    return "\n".join(lines) + "\n", rendered_tree


def main() -> None:
    report, rendered_tree = build_report()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    TREE_FILE.write_text(rendered_tree, encoding="utf-8")
    print(_UTILS.safe_console(report), end="")
    print(f"\nSaved run output to: {OUTPUT_FILE}")
    print(f"Saved full tree to: {TREE_FILE}")


if __name__ == "__main__":
    main()
