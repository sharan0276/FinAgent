# Data Ingestion Notes

## Goal

This project is building a company financial analyzer based on SEC EDGAR `10-K` filings.

For the current iteration, the focus is deliberately narrow:

- work with only `10-K` annual reports
- validate the ingestion and parsing flow on `Meta` first
- keep parsing deterministic and code-based
- delay LLM usage until after extraction and validation are reliable

## Current Approach

The current `Meta` spike follows this flow:

1. Fetch the latest `10-K` HTML from SEC EDGAR.
2. Parse the HTML into semantic elements and a tree.
3. Validate the parse deterministically.
4. Export reviewable artifacts to files.

This work lives mainly in:

- `data-ingestion/sec_parser_utils.py`
- `data-ingestion/meta_sec_parser_spike.py`
- `data-ingestion/meta_sec_parser_validate.py`

## Libraries Used

### `sec-downloader`

Used to fetch SEC filing HTML from EDGAR using a proper SEC identity header.

Purpose in this project:

- download the target company's latest `10-K`
- keep retrieval simple and repeatable

### `sec-parser`

Used to parse the filing HTML into semantic elements such as:

- `TitleElement`
- `TextElement`
- `SupplementaryText`
- `TableElement`

It also builds a semantic tree that is easier to inspect than raw HTML.

## Important Library Decision

The installed `sec-parser` version does not expose a dedicated `Edgar10KParser`.
It exposes `Edgar10QParser`, and the documentation suggests two ways to use it on other forms such as `10-K`.

We chose the documentation's "Method 2":

- remove `TopSectionManagerFor10Q`
- remove `TopSectionTitleCheck`
- reuse the rest of the parsing pipeline

Reason for choosing Method 2:

- it avoids repeated `10-Q`-specific section warnings
- it produces a fuller and more useful `10-K` tree
- it works better for our current `Meta` experiment than leaving the 10-Q logic in place

Tradeoff:

- there is still some noisy metadata near the top of the filing
- the parser is usable for prototype extraction, but not yet perfect as a `10-K`-native parser

## Deterministic Parsing and Validation

No LLM is used in the parsing or validation stage.

The current pipeline is entirely code-based:

- filing download
- HTML parsing
- section detection
- ordering checks
- keyword checks
- noise checks
- output generation

This is important because it makes the extraction:

- reproducible
- cheaper
- easier to debug
- easier to compare across companies later

## Chosen Core Sections

The current validator checks for the following core `10-K` sections:

- `PART I`
- `Item 1. Business`
- `Item 1A. Risk Factors`
- `Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations`
- `Item 7A. Quantitative and Qualitative Disclosures About Market Risk`
- `Item 8. Financial Statements and Supplementary Data`
- `PART III`
- `Item 10`

`Item 15` was intentionally removed from the required list.

Why these sections were chosen:

- `Item 1` gives company and business context
- `Item 1A` gives explicit risk disclosures
- `Item 7` gives management narrative on performance, liquidity, and operations
- `Item 7A` gives market-risk exposure
- `Item 8` gives the formal audited financial statements and notes
- `PART III` and `Item 10` help confirm broader structure and continuity in the filing

## Validation Logic

The validator is designed to be deterministic.

### 1. Filing identity checks

The validator checks whether the parsed document appears to match the intended filing:

- company name found near the top
- `FORM 10-K` found near the top
- filing year found near the top
- HTML size above a minimum threshold

### 2. Core section coverage

The validator confirms whether each required section is found.

### 3. Section ordering

The validator checks whether the required sections occur in the expected filing order.

### 4. Section quality

For key sections, it calculates:

- start and end positions
- word count
- character count
- table node count
- expected keyword hits

### 5. Noise checks

The validator also checks for:

- metadata-heavy content near the top
- replacement-character issues
- XBRL-like token patterns

## Current Output Files

Current outputs are written under `data-ingestion/outputs/`.

Important files include:

- `meta_sec_parser_spike_output.txt`
- `meta_sec_parser_tree.txt`
- `meta_sec_parser_validation.json`
- `meta_sec_parser_heading_index.txt`
- `meta_sections/item_1_business.txt`
- `meta_sections/item_1a_risk_factors.txt`
- `meta_sections/item_7_mda.txt`
- `meta_sections/item_7a_market_risk.txt`
- `meta_sections/item_8_financial_statements.txt`

These files support both:

- deterministic validation
- human review

## Table Handling Status

Tables are currently being detected by the parser.

In particular, `Item 8` includes many `TableElement` nodes and the validator counts them.

Current status:

- yes, tables are being fetched
- yes, table content is being included in exported section text
- no, tables are not yet preserved as clean row-column structured outputs

This means the current pipeline is good for:

- proving table presence
- reviewing table-related content

But it is not yet ideal for:

- robust numerical extraction
- direct ratio computation from structured financial statements

## Current Status for Meta

The latest deterministic validation result for `Meta` is effectively:

- core sections found
- section ordering correct
- key section exports produced
- parser output usable for a prototype
- cleanup still needed for metadata noise near the top

The current overall status is best described as:

- `usable_with_cleanup`

## Practical Conclusion

At this stage, the project has a working deterministic prototype for:

- fetching a `10-K`
- parsing it with `sec-parser`
- exporting meaningful sections
- validating parse quality without an LLM

This is a strong base for the next step, which will likely be one of:

- preamble cleanup before downstream usage
- cleaner table extraction from `Item 8`
- extending the same flow from `Meta` to additional companies

## Next Likely Steps

Recommended near-term steps:

1. Clean noisy metadata before the main filing body.
2. Export `Item 8` tables in a more structured form.
3. Repeat the same deterministic validation for a small set of additional companies.
4. Only after extraction quality is stable, introduce embeddings and LLM-based comparison.
