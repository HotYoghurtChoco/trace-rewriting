"""Sentence-level MaxLoss on a frozen student's completion loss."""

from collections import Counter
from difflib import SequenceMatcher
import re

from experiments.maxloss_sentence.data import ROUNDS, MIN_GAIN
from experiments.maxloss_sentence.rewriter import BadReply, round_seeds
from experiments.maxloss_v1.optimizer import UnscorableTrace
from experiments.maxloss_v1.text_edits import NUMBER, answer_start, protected_spans


END = re.compile(r"""[.!?]+["'”’)]*(?=\s|$)""")
ABBREVIATION = re.compile(r"(?:\b(?:Mr|Ms|Mrs|Dr|St|vs|e\.g|i\.e)|\b[A-Z])\.$")
PREFIX = re.compile(r"^\s*(?:(?:[-*•]|\d+[.)])\s+)?")
STRUCTURE = re.compile(r"[<>{}\\=×÷\x00-\x1f\x60]|(?:^|\s)#{1,6}\s")
ARITHMETIC = re.compile(r"\d[\d.,]*\s*[-+*/×÷−=^]\s*[-+]?\d")
QUANTITY = re.compile(r"[$£€%]|(?<!\w)[+-](?=\d)|[*/×÷=]")


def prose_spans(text):
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        body = line.group()
        start = PREFIX.match(body).end()
        if body[start:].startswith("#"):
            continue
        for boundary in END.finditer(body, start):
            if ABBREVIATION.search(body[start:boundary.end()]):
                continue
            while start < boundary.end() and body[start].isspace():
                start += 1
            a, b = line.start() + start, line.start() + boundary.end()
            sentence = text[a:b]
            if (len(re.findall(r"[A-Za-z]+", sentence)) >= 4
                    and not STRUCTURE.search(sentence) and not ARITHMETIC.search(sentence)):
                spans.append((a, b))
            start = boundary.end()
    return spans


def sentence_spans(text, problem=""):
    """Return prose sentence offsets, leaving equations, markup and answer text in place."""
    limit = answer_start(text)
    if limit is None:
        return []
    protected = protected_spans(text, problem)
    return [(a, b) for a, b in prose_spans(text[:limit])
            if not any(a < hi and lo < b for lo, hi in protected)]


def eligible_wording(source, new):
    if not isinstance(new, str) or not new or new == source or STRUCTURE.search(new):
        return False
    if Counter(NUMBER.findall(source)) != Counter(NUMBER.findall(new)):
        return False
    if Counter(QUANTITY.findall(source)) != Counter(QUANTITY.findall(new)):
        return False
    # Also rejects labels, bullet lists and candidates containing two sentences.
    return prose_spans(new) == [(0, len(new))]


def token_distance(old, new):
    """Exact Levenshtein distance on body token IDs. Trim common ends before the DP."""
    start, left, right = 0, len(old), len(new)
    while start < min(left, right) and old[start] == new[start]:
        start += 1
    while left > start and right > start and old[left - 1] == new[right - 1]:
        left -= 1
        right -= 1
    old, new = old[start:left], new[start:right]
    if len(new) > len(old):
        old, new = new, old
    previous = list(range(len(new) + 1))
    for i, token in enumerate(old, 1):
        current = [i]
        for j, other in enumerate(new, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (token != other)))
        previous = current
    return previous[-1]


def rank_sentences(tokenizer, instruction, problem, trace, spans, feature, gradient):
    """Map the actual scored chat IDs to text offsets before pooling input gradients."""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": problem.strip()},
        {"role": "assistant", "content": trace.strip()},
    ]
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != feature.ids:
        raise ValueError("Gradient offsets do not match the scored chat input")
    norms = gradient[0].float().norm(dim=-1).tolist()
    body = trace.strip()
    leading = len(trace) - len(trace.lstrip())
    # The scorer already marks the first assistant token in the same token sequence.
    assistant_start = encoded["offset_mapping"][feature.mask.index(1)][0]
    assistant = full[assistant_start:]
    pos = assistant.find(body)
    # Some chat templates normalize whitespace around the thinking delimiter.
    blocks = ([(0, pos, len(body))] if pos >= 0 else
              SequenceMatcher(None, body, assistant, autojunk=False).get_matching_blocks())
    ranked = []
    for start, end in spans:
        for src, dst, length in blocks:
            if src <= start - leading and end - leading <= src + length:
                offset = assistant_start + dst - src - leading
                values = [norm for (a, b), norm, mask in zip(encoded["offset_mapping"], norms, feature.mask)
                          if mask and a < offset + end and b > offset + start and a < b]
                if values:
                    ranked.append((sum(values) / len(values), (start, end)))
                break
    return sorted(ranked, key=lambda item: (-item[0], item[1][0]))


def optimize_sentence(student, tokenizer, rewriter, instruction, row, index, max_percent=None):
    from experiments.maxloss_v1.run_optimization import answer_status

    problem, original = row["problem"], row["original_trace"]

    def body_ids(text):
        return tokenizer.encode(text.strip(), add_special_tokens=False)

    original_ids = body_ids(original)
    current = original
    feature = student.feature(problem, current)
    initial = current_loss = student.loss(feature)
    remaining = sentence_spans(current, problem)
    history, ranked, accepted = [], [], 0
    stop = "round_budget"
    for turn in range(ROUNDS):
        if not ranked:
            if not remaining:
                stop = "no_remaining_sentence"
                break
            _, gradient = student.loss_and_gradient(feature)
            ranked = rank_sentences(tokenizer, instruction, problem, current, remaining, feature, gradient)
            if not ranked:
                stop = "no_aligned_sentence"
                break
        strength, span = ranked.pop(0)
        remaining.remove(span)
        a, b = span
        source = current[a:b]
        proposal_seed, judge_seed = round_seeds(index, turn + 1)
        trials, pending = [], []
        record = {"round": turn + 1, "source": source, "start": a, "mean_gradient_norm": strength,
                  "proposals": trials, "accepted": None, "loss_before": current_loss, "loss_after": current_loss}
        history.append(record)
        try:
            alternatives = rewriter.propose(problem, current, source, proposal_seed)
        except BadReply as error:
            record["format_error"] = str(error)
            continue
        seen = {source}
        for new in alternatives:
            trial = {"sentence": new}
            trials.append(trial)
            if new in seen or not eligible_wording(source, new):
                trial["status"] = "duplicate_or_structure"
                continue
            seen.add(new)
            candidate = current[:a] + new + current[b:]
            try:
                encoded = student.feature(problem, candidate)
            except UnscorableTrace:
                trial["status"] = "input_cap"
                continue
            if max_percent is not None:
                distance = token_distance(original_ids, body_ids(candidate))
                if 100 * distance > max_percent * len(original_ids):
                    trial["status"] = "edit_budget"
                    continue
            pending.append((trial, candidate, encoded))
        try:
            judgments = (rewriter.assess(problem, current, source, [t["sentence"] for t, _, _ in pending],
                                         judge_seed)["judgments"] if pending else [])
        except BadReply as error:
            record["format_error"] = str(error)
            continue
        best = None
        for (trial, candidate, encoded), judgment in zip(pending, judgments, strict=True):
            trial["semantic_rating"] = judgment["rating"]
            if judgment["rejection"] is not None:
                trial["status"] = judgment["rejection"]
                continue
            if not answer_status(row["solution"], candidate)["correct"]:
                trial["status"] = "answer_check"
                continue
            loss = student.loss(encoded)
            trial.update(status="scored", loss=loss)
            if loss - current_loss >= MIN_GAIN and (best is None or loss > best[0]):
                best = (loss, trial["sentence"], candidate, encoded)
        if best is not None:
            current_loss, new, current, feature = best
            delta = len(new) - len(source)
            remaining = [(x + delta, y + delta) if x >= b else (x, y) for x, y in remaining]
            ranked = []  # The next gradient is evaluated on the accepted trace.
            record["accepted"] = new
            accepted += 1
        record["loss_after"] = current_loss
    return {"text": current, "loss_before": initial, "loss_after": current_loss,
            "accepted_edits": accepted, "stop": stop, "rounds": history}
