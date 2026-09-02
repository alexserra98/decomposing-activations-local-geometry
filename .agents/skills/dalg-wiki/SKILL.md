---
name: dalg-wiki
description: Navigate, read, create, update, or reorganize the DALG Markdown wiki under docs/. Use when answering from project documentation, recording durable research knowledge, maintaining the documentation index and cross-links, or deciding where a document belongs.
---

# Maintain the DALG Wiki

The wiki is plain Markdown tracked with the repository. Its canonical entry
point is `docs/README.md`; treat that page as the navigation source rather than
maintaining a second document catalog in this skill.

## Read from the wiki

1. Read `docs/README.md` completely and use its task-oriented routes to select
   the smallest relevant set of pages.
2. Follow related-document links only when they are needed for the request. Do
   not load every page by default.
3. Distinguish what the documentation says from what the implementation does.
   For a question explicitly about documented intent, answer from the wiki. For
   a claim about current runtime behavior or an update that asserts technical
   facts, verify the relevant code, config, tests, or artifacts and report any
   disagreement with the docs.
4. For read-only requests, do not edit the wiki. Cite the relevant local pages
   and line numbers in the answer.

## Write to the wiki

Before editing, read the hub, the target page, and any directly related page.
Search the whole repository with hidden files included when changing paths or
terminology; workflow skills under `.agents/` can contain documentation links.

Classify a new page by its primary use:

- `research/`: current direction, open questions, and research snapshots
- `models/`: durable model behavior and invariants
- `workflows/`: task-oriented operational guides
- `reference/`: exact configuration, schema, and interface contracts
- `evaluation/`: metric definitions and evaluator output contracts
- `experiments/`: temporary, run-specific, or attachable context
- `archive/`: superseded material retained for history; create it only when used

Make the smallest update that captures the requested knowledge. Preserve
unrelated edits and avoid copying the same explanation across multiple pages.
Link to the canonical explanation instead. Keep generated outputs and large
artifacts outside `docs/`.

Every new page must begin with one H1 followed by a compact routing block:

```markdown
# Page title

> **Kind:** Workflow · **Status:** Current · **Use when:** Running or inspecting ...
```

Choose truthful status language. Change a page's date only when its substantive
content has been reviewed, not merely because formatting or links changed.

Update `docs/README.md` when a page is added, moved, removed, archived, or
changes role. For moves, repair every inbound link across the repository,
including hidden agent and Claude files. Update `AGENTS.md` only when the
high-level documentation map or agent routing changes.

Keep workflow skills and wiki pages complementary: skills contain recurring
agent execution procedures, while the wiki contains durable concepts,
contracts, research context, and human-readable guides. Cross-link them when
useful; do not duplicate an entire procedure in both places.

## Validate writes

Before finishing:

- confirm every local Markdown link resolves relative to its containing file;
- search hidden and ordinary files for stale paths after a move or rename;
- check that the hub contains exactly one useful entry for every active page;
- run `git diff --check` and inspect the documentation diff;
- report pages created, moved, archived, or materially changed.

Documentation-only changes do not require the code test suite. Run targeted
tests when documentation edits accompany or assert a code behavior change.
