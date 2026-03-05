"""Skill Editor — applies targeted edits to SKILL.md based on trace diagnosis.

The editor is the UPDATE step of the ABSTRAL meta-loop. For each finding
in the diagnosis, it asks Claude to make a minimal, targeted edit to the
skill document. Every edit is cited to a specific trace.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import anthropic

from runner.agent_system import Diagnosis


EDITOR_SYSTEM_PROMPT = """You are a precision document editor for an Agent Builder Skill (SKILL.md).

You will receive a skill document and one targeted edit instruction.

RULES:
1. Make ONLY the minimal change described in the edit instruction.
2. Do NOT touch other sections or rewrite existing rules unless the edit instruction specifically says to.
3. Append a citation comment to every new rule you add, in this format:
   <!-- [Evidence: trace TRACE_ID, iter N] -->
4. Preserve all existing content, formatting, and structure.
5. If adding a new entry to a list, add it at the end of the relevant section.
6. If correcting an existing rule, mark the old version with [CORRECTED] and add the new version.
7. Output the COMPLETE updated document text — not just the changed section.

Output the complete updated skill document."""


async def apply_diagnosis(
    diagnosis: Diagnosis,
    skill_path: str,
    output_path: str,
    model: str = "claude-sonnet-4-20250514"
) -> list[dict]:
    """Apply each finding as a targeted edit to the skill document.

    Returns a list of diffs applied.
    """
    skill_text = Path(skill_path).read_text()
    diff_log = []

    if not diagnosis.findings:
        return diff_log

    client = anthropic.AsyncAnthropic()

    for finding in diagnosis.findings:
        update = finding.get("proposed_update", {})
        if not update:
            continue

        # Build citation
        evidence = finding.get("evidence_trace", {})
        trace_id = evidence.get("trace_id", "unknown")
        citation = f"[Evidence: trace {trace_id}, iter {diagnosis.iteration}]"

        operation = update.get("operation", "UPDATE_1")
        target_section = update.get("target_section", "K")
        description = update.get("description", finding.get("description", ""))

        response = await client.messages.create(
            model=model,
            max_tokens=4000,
            system=EDITOR_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"""SKILL DOCUMENT:
{skill_text}

EDIT INSTRUCTION:
Operation: {operation}
Target section: {target_section} (look for the ### {target_section}: heading)
Evidence class: {finding.get('evidence_class', 'unknown')}
Description: {description}

CITATION TO APPEND: <!-- {citation} -->

Output the complete updated skill document.
"""
            }]
        )

        updated_text = response.content[0].text

        # Validate: the response should still look like a skill document
        if "### K:" in updated_text or "### K " in updated_text or "## K:" in updated_text:
            skill_text = updated_text
            diff_log.append({
                "finding": finding.get("description", "")[:100],
                "operation": operation,
                "target_section": target_section,
                "evidence_class": finding.get("evidence_class", ""),
                "citation": citation
            })
        else:
            # Claude may have wrapped the document in code blocks
            # Try extracting from code blocks
            code_block = re.search(r'```(?:markdown)?\s*\n?(.*?)\n?```', updated_text, re.DOTALL)
            if code_block:
                extracted = code_block.group(1)
                if "### K:" in extracted or "### K " in extracted or "## K:" in extracted:
                    skill_text = extracted
                    diff_log.append({
                        "finding": finding.get("description", "")[:100],
                        "operation": operation,
                        "target_section": target_section,
                        "evidence_class": finding.get("evidence_class", ""),
                        "citation": citation
                    })

    # Save updated skill
    Path(output_path).write_text(skill_text)

    # Save version snapshot
    versions_dir = Path(skill_path).parent / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    version_path = versions_dir / f"ABS_v{diagnosis.iteration + 1}.md"
    shutil.copy(output_path, version_path)

    return diff_log


def compute_skill_diff(skill_path_a: str, skill_path_b: str) -> list[str]:
    """Compute a simple diff between two skill versions."""
    lines_a = Path(skill_path_a).read_text().splitlines()
    lines_b = Path(skill_path_b).read_text().splitlines()

    diffs = []
    set_a = set(lines_a)
    set_b = set(lines_b)

    added = set_b - set_a
    removed = set_a - set_b

    for line in added:
        if line.strip():
            diffs.append(f"+ {line.strip()}")
    for line in removed:
        if line.strip():
            diffs.append(f"- {line.strip()}")

    return diffs
