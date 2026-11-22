"""One gradient-selected mathematical passage per compact teacher solution."""

import re

from experiments.maxloss_sentence.data import MIN_GAIN
from experiments.maxloss_sentence.method import rank_sentences
from experiments.maxloss_sentence.rewriter import BadReply
from experiments.maxloss_step.rewriter import ANSWER_MARKER, JudgeOutputTruncated
from experiments.maxloss_v1.optimizer import UnscorableTrace
from experiments.maxloss_v1.run_optimization import answer_status
from experiments.maxloss_v1.text_edits import answer_start


BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")
MATH = re.compile(r"\d|[=+×÷*/%]|\\(?:frac|times|div)|\b(?:half|twice|double|triple|"
                  r"quarter|sum|difference|product|ratio|remainder|per|subtract|multiply|divide)\b", re.I)


def protected_answer(trace):
    boxes = list(BOX.finditer(trace))
    if boxes:
        match = boxes[-1]
        depth = 1
        for i in range(match.end(), len(trace)):
            if i and trace[i - 1] == "\\":
                continue
            if trace[i] == "{":
                depth += 1
            elif trace[i] == "}":
                depth -= 1
                if not depth:
                    return "\\[\n" + trace[match.start():i + 1] + "\n\\]"
        return None
    start = answer_start(trace)
    return trace[start:].strip() if start is not None else None


def compose(steps, answer):
    """Locate each mathematical step in the actual scored string."""
    cursor, spans = 0, []
    for step in steps:
        end = cursor + len(step)
        spans.append((cursor, end) if MATH.search(step) else None)
        cursor = end + 1
    return "\n".join(steps) + "\n\n" + answer, spans


def selected_passage(ranked, spans):
    """Choose the highest-gradient anchor and its highest-ranked immediate math neighbor."""
    norm, anchor = ranked[0]
    position = spans.index(anchor)
    neighbors = [spans[i] for i in (position - 1, position + 1) if 0 <= i < len(spans)]
    neighbor = next((span for _, span in ranked if span in neighbors), anchor)
    return norm, anchor, (min(anchor[0], neighbor[0]), max(anchor[1], neighbor[1]))


def rewrite_row(student, tokenizer, rewriter, instruction, row, index):
    original, question = row["original_trace"], row["problem"]
    seed = 888 + 10 * index

    def token_count(text):
        return len(tokenizer.encode(text.strip(), add_special_tokens=False))

    original_tokens = token_count(original)
    info = {"selected": "original", "original_tokens": original_tokens, "final_tokens": original_tokens,
            "original_loss": None, "base_loss": None, "final_loss": None, "proposals": {}}

    def finish(reason, text=original):
        info.update(reason=reason, final_tokens=token_count(text))
        return {"index": index, **row, "rewrite_trace": text, "step": info}

    answer = protected_answer(original)
    if not answer or not answer_status(row["solution"], answer)["correct"]:
        return finish("unverified_original_answer")
    try:
        orig_feature = student.feature(question, original)
    except UnscorableTrace:
        return finish("unscorable_original")
    info["original_loss"] = info["final_loss"] = student.loss(orig_feature)
    try:
        steps = rewriter.base(question, original, answer, seed)
    except BadReply as error:
        info["base_error"] = str(error)
        return finish("malformed_base")
    if any(ANSWER_MARKER.search(s) for s in steps):
        return finish("answer_marker_in_steps")
    base, spans = compose(steps, answer)
    info.update(base=base, base_tokens=token_count(base))
    if info["base_tokens"] >= original_tokens:
        return finish("base_did_not_compress")
    if not answer_status(row["solution"], base)["correct"]:
        return finish("base_answer_mismatch")
    try:
        feature = student.feature(question, base)
    except UnscorableTrace:
        return finish("unscorable_base")
    base_loss, gradient = student.loss_and_gradient(feature)
    info["base_loss"] = base_loss
    ranked = rank_sentences(tokenizer, instruction, question, base, [s for s in spans if s], feature, gradient)
    del gradient
    solutions = {"base": base}
    features = {}
    if ranked:
        norm, anchor, (start, end) = selected_passage(ranked, spans)
        info["gradient_anchor"] = {"start": anchor[0], "end": anchor[1], "text": base[slice(*anchor)],
                                   "mean_token_gradient_norm": norm, "ranked_steps": len(ranked)}
        info["selected_span"] = {"start": start, "end": end, "text": base[start:end]}
        try:
            fragments = rewriter.propose(question, base, (start, end), anchor, seed + 2)
        except BadReply as error:
            fragments = []
            info["proposal_error"] = str(error)
        for label, fragment in zip(("A", "B"), fragments):
            item = info["proposals"][label] = {"replacement": fragment}
            text = base[:start] + fragment + base[end:]
            if text in solutions.values() or ANSWER_MARKER.search(fragment):
                item["rejected"] = "duplicate_or_answer_marker"
                continue
            # Allow a few extra words for an equivalent operation or its unit; every arm stays compact.
            item["body_tokens"] = token_count(text)
            if item["body_tokens"] >= original_tokens:
                item["rejected"] = "not_shorter_than_original"
                continue
            if not answer_status(row["solution"], text)["correct"]:
                item["rejected"] = "answer_mismatch"
                continue
            try:
                features[label] = student.feature(question, text)
            except UnscorableTrace:
                item["rejected"] = "unscorable_candidate"
                continue
            solutions[label] = text
    # Fresh judge call: do not reveal the loss, the gradient or the preferred candidate.
    info["judge_stats"] = {}
    try:
        verdicts = rewriter.judge(row, solutions, seed + 4, stats=info["judge_stats"])
    except JudgeOutputTruncated as error:
        info["judge_error"] = str(error)
        return finish("judge_output_truncated")
    except BadReply as error:
        info["judge_error"] = str(error)
        return finish("malformed_judge")
    info["verdicts"] = verdicts
    if not verdicts["base"]["valid"]:
        return finish("invalid_base_derivation")
    winner, highest = "base", base_loss
    for label, candidate_feature in features.items():
        item = info["proposals"][label]
        if not verdicts[label]["valid"]:
            item["rejected"] = "failed_math_step_check"
            continue
        loss = item["loss"] = student.loss(candidate_feature)
        if loss >= base_loss + MIN_GAIN and loss > highest:
            winner, highest = label, loss
    info.update(selected=winner, final_loss=highest)
    if winner != "base":
        reason = "maxloss_accepted"
    elif not ranked:
        reason = "no_scoreable_math_passage"
    elif "proposal_error" in info:
        reason = "malformed_proposal"
    else:
        reason = "no_qualifying_gain"
    return finish(reason, solutions[winner])
