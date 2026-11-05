"""Local wording proposals for the first completion-loss experiment.

The proposal vocabulary is intentionally explicit: discourse connectives and
metalinguistic instructions, not arithmetic relations, quantities or entities.
Every proposal is a single span replacement. The five-example run exposes the
actual changes for human review before the 100-example run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


POLICY_VERSION = "local_wording_v1"
# Connectives only match at the beginning of a sentence/line and before a comma.
CONNECTIVES = (
    ("therefore", "thus", "hence", "consequently"),
    ("first", "firstly"),
    ("next", "then", "subsequently"),
    ("finally", "lastly"),
    ("additionally", "furthermore", "moreover"),
)
WORDING = (
    ("calculate", "compute"),
    ("calculating", "computing"),
    ("calculated", "computed"),
    ("calculation", "computation"),
    ("calculations", "computations"),
    ("determine", "find out"),
    ("determining", "finding out"),
    ("let's", "let us"),
    ("step by step", "one step at a time"),
)
MATH_OR_MARKUP = re.compile(
    r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|(?<!\\)\$[^\n$]+(?<!\\)\$"
    r"|```.*?```|`[^`\n]+`|<[^>\n]+>", re.DOTALL
)
FINAL_HEADING = re.compile(r"(?im)^\s*(?:[#*>\s]*)(?:final answer|answer)\b")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    before: str
    after: str
    rule: str

    def apply(self, text: str) -> str:
        if text[self.start:self.end] != self.before:
            raise ValueError("Edit offsets no longer match the current text")
        return text[:self.start] + self.after + text[self.end:]

    def to_dict(self) -> dict:
        return asdict(self)


def answer_start(text: str) -> int | None:
    """Protect the final answer paragraph, including surrounding answer prose."""
    headings = list(FINAL_HEADING.finditer(text))
    if headings:
        return headings[-1].start()
    boxes = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if boxes:
        paragraph_break = text.rfind("\n\n", 0, boxes[-1].start())
        return paragraph_break + 2 if paragraph_break >= 0 else 0
    markers = list(re.finditer(r"(?im)^\s*####\s|\b(?:the )?final answer is\b", text))
    if markers:
        return markers[-1].start()
    return None


def protected_spans(text: str, problem: str = "") -> list[tuple[int, int]]:
    spans = [(m.start(), m.end()) for m in MATH_OR_MARKUP.finditer(text)]
    # Inline equations are kept as complete lines, including their verbal labels.
    spans.extend((m.start(), m.end()) for m in re.finditer(r"(?m)^.*(?:[=×÷]|\\[A-Za-z]+).*$", text))
    start = answer_start(text)
    if start is not None:
        spans.append((start, len(text)))
    if problem:
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(problem), text))
    return spans


def preserve_case(before: str, after: str) -> str:
    if before.isupper():
        return after.upper()
    if before[0].isupper():
        return after[0].upper() + after[1:]
    return after


def propose_edits(text: str, problem: str = "") -> list[Edit]:
    if answer_start(text) is None:
        return []
    protected = protected_spans(text, problem)
    proposals = []
    for kind, groups in (("connective", CONNECTIVES), ("wording", WORDING)):
        for group in groups:
            pattern = r"(?<![\w'])" + "(?:" + "|".join(map(re.escape, group)) + r")(?![\w'])"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                if any(start < b and end > a for a, b in protected):
                    continue
                if kind == "connective":
                    prefix = text[:start].rstrip(" \t*")
                    if prefix and prefix[-1] not in ".!?\n":
                        continue
                    if not re.match(r"\s*,", text[end:]):
                        continue
                for replacement in group:
                    if replacement == match.group().lower():
                        continue
                    proposals.append(Edit(start, end, match.group(), preserve_case(match.group(), replacement), kind))
    return sorted(proposals, key=lambda edit: (edit.start, edit.after))


def validate_edit(text: str, edit: Edit, problem: str = "") -> list[str]:
    """Check an edit against the explicit vocabulary and protected text."""
    if edit not in propose_edits(text, problem):
        return ["outside_wording_policy_or_protected_span"]
    changed = edit.apply(text)
    old_start, new_start = answer_start(text), answer_start(changed)
    reasons = []
    if old_start is None or new_start is None or text[old_start:] != changed[new_start:]:
        reasons.append("final_answer_changed")
    if NUMBER.findall(text) != NUMBER.findall(changed):
        reasons.append("numbers_changed")
    if MATH_OR_MARKUP.findall(text) != MATH_OR_MARKUP.findall(changed):
        reasons.append("math_or_markup_changed")
    return reasons
