"""Render the canonical requirement register without overwriting evidence statuses."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    register = json.loads((ROOT / 'docs/traceability.json').read_text())
    rows = register['requirements']
    identifiers = [row['id'] for row in rows]
    if len(rows) != 117 or len(set(identifiers)) != len(rows):
        raise ValueError('REQUIREMENT_REGISTER_MEMBERSHIP')
    lines = ['# Requirement traceability matrix', '', register['status_definition'] + '.', '',
             'Separate submission attachments cover the recording guide and tooling statement.', '',
             '| ID / PDF page | Requirement | Implementation | Test / evidence | Status |',
             '| --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['id']} / {row['source_page']} | {row['requirement']} | "
                     f"{row['implementation']} | {row['test_or_manual_check']} | {row['status']} |")
    (ROOT / 'docs/traceability.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'requirements': len(rows), 'statuses': {
        status: sum(row['status'] == status for row in rows)
        for status in sorted({row['status'] for row in rows})}}))


if __name__ == '__main__':
    main()
