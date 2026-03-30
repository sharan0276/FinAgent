import os
from pathlib import Path

from dotenv import load_dotenv
from sec_downloader import Downloader
import sec_parser as sp
from sec_parser.processing_steps import (
    IndividualSemanticElementExtractor,
    TopSectionManagerFor10Q,
    TopSectionTitleCheck,
)


OUTPUT_DIR = Path("data-ingestion") / "outputs"


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def first_lines(text: str, count: int = 20) -> str:
    return "\n".join(text.splitlines()[:count])


def safe_console(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def get_non_10q_steps():
    all_steps = sp.Edgar10QParser().get_default_steps()
    steps_without_top_section_manager = [
        step for step in all_steps if not isinstance(step, TopSectionManagerFor10Q)
    ]

    def get_checks_without_top_section_title_check():
        all_checks = sp.Edgar10QParser().get_default_single_element_checks()
        return [
            check
            for check in all_checks
            if not isinstance(check, TopSectionTitleCheck)
        ]

    return [
        IndividualSemanticElementExtractor(
            get_checks=get_checks_without_top_section_title_check
        )
        if isinstance(step, IndividualSemanticElementExtractor)
        else step
        for step in steps_without_top_section_manager
    ]


def build_non_10q_parser():
    return sp.Edgar10QParser(get_steps=get_non_10q_steps)


def fetch_filing_html(ticker: str, form: str) -> str:
    load_dotenv()
    downloader = Downloader(require_env("SEC_APP_NAME"), require_env("SEC_EMAIL"))
    return downloader.get_filing_html(ticker=ticker, form=form)


def parse_filing_html(html: str):
    parser = build_non_10q_parser()
    elements = parser.parse(html)
    tree = sp.TreeBuilder().build(elements)
    rendered_tree = sp.render(tree)
    return elements, tree, rendered_tree
