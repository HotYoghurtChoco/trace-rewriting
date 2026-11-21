"""Sentence proposals and a separate semantic check on the local 6C endpoint."""

import json
import re

MODEL = "openai/gpt-oss-120b"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"

PROPOSE = (
    'Rephrase only the specified sentence of the supplied solution in three different ways. '
    'Return a JSON object {"sentences":["...","...","..."]}. Write one complete sentence per item. '
    'Vary the sentence structure or clause order without adding generic lead-ins. '
    'Preserve who did what, all quantities, numeric spellings, units, time, comparisons and '
    'arithmetic. Keep each sentence natural in its surrounding text. Do not add a step, a '
    'new claim, a label or commentary.'
)
JUDGE = (
    'Interpret the source sentence and each replacement independently at the same point in '
    'the solution. Use the problem and surrounding sentences to resolve references. Read '
    'what each sentence actually says, without repairing it to fit the intended answer. '
    'First inventory the reference groups or amounts relevant to any proportional share in '
    'these sentences. Include the original population and any restricted population that '
    'could supply the base. Assign a short ID to each distinct group. Describe its membership '
    'and give its size and unit if derivable. Reuse an ID for the same group under other wording. '
    'For each proportional share, including one expressed in words, ask what would count '
    'as 100% of the base of this expression, holding prior facts fixed. Distinguish the group '
    'being described by the share from the group the share is taken OF. Identify the base '
    'from that sentence, without back-solving from the reported result. Express the claim '
    'as rate times that specific base when applicable. If the wording admits different bases '
    'in context, record null for its base. '
    'Preserve who did what, time, operations and the direction of comparisons. '
    'Check the entity, unit and scope of every quantity. Compare facts in both directions. '
    'Rate 5 only when all facts and relations are equivalent, 4 for a minor omission or '
    'uncertainty, 3 for a material change, 2 for a contradiction and 1 for an unrelated claim. '
    'Return JSON. groups is an object mapping group IDs to description strings. source_claim '
    'is a string. source_bases maps one slot ID per source share to its base group ID or null. '
    'In judgments, return one object per replacement, in order, with claim (string), '
    'bases (object) and rating (integer). '
    'In bases reuse each source slot ID for the corresponding share even if clauses move. '
    'Omit a removed share and give an added share a new slot ID. Use {} for a sentence '
    'with no shares, and {} for groups if there are none. Keep descriptions brief.'
)


def round_seeds(index, round_number):
    """Return the existing proposal and judge seeds for a one-based editing round."""
    proposal = 888 + 100 * index + 4 * (round_number - 1)
    return proposal, proposal + 2


class BadReply(ValueError):
    """The endpoint replied, but did not supply the requested structured output."""


def _read_bases(bindings, groups):
    if not isinstance(bindings, dict):
        raise ValueError("Expected a mapping from share IDs to base group IDs")
    for group in bindings.values():
        if group is not None and (not isinstance(group, str) or group not in groups):
            raise ValueError("Base refers to an undefined group")
    return bindings


def _assess_judgments(result, data):
    if not isinstance(result["source_claim"], str) or not result["source_claim"].strip():
        raise ValueError("Missing source claim")
    groups = result["groups"]
    if not isinstance(groups, dict) or not all(isinstance(v, str) and v.strip() for v in groups.values()):
        raise ValueError("Expected reference groups with descriptions")
    source = _read_bases(result["source_bases"], groups)
    # A literal percent sign requires a base even if the model omits that share.
    source_unknown = None in source.values() or len(source) < data["source_sentence"].count("%")
    for sentence, item in zip(data["replacements"], result["judgments"], strict=True):
        rating, claim = item["rating"], item["claim"]
        if type(rating) is not int or not 1 <= rating <= 5:
            raise ValueError("Invalid semantic rating")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Missing replacement claim")
        bases = _read_bases(item["bases"], groups)
        if source_unknown or None in bases.values() or len(bases) < sentence.count("%"):
            rejection = "reference_base_uncertain"
        elif bases != source:
            rejection = "reference_base_changed"
        elif rating != 5:
            rejection = "meaning_changed_or_uncertain"
        else:
            rejection = None
        item["rejection"] = rejection
    return result


class Rewriter:
    def __init__(self, endpoint, request=None):
        if request is None:
            from experiments.stage6c_cosrewrite_v1.run_single_problem_closed_loop import request_json
            request = request_json
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.request = request

    def _ask(self, instruction, data, key, size, seed):
        last_error = None
        for retry in range(2):
            payload = {
                "model": MODEL,
                "messages": [{"role": "system", "content": instruction},
                             {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
                "temperature": 0.6 if key == "sentences" else 0.0,
                "seed": seed + retry, "reasoning_effort": "medium",
                "max_tokens": 4096 if key == "judgments" else 1024, "stream": False,
            }
            # Transport failures stop the job. Completed rows remain reusable.
            response = self.request(self.endpoint, payload)
            try:
                choice = response["choices"][0]
                if choice["finish_reason"] != "stop":
                    raise ValueError(f"finish_reason={choice['finish_reason']}")
                content = choice["message"]["content"].strip()
                content = re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$", "", content)
                result = json.loads(content)
                values = result[key]
                if not isinstance(values, list) or len(values) != size:
                    raise ValueError(f"Expected {size} {key}")
                if key == "sentences" and all(isinstance(v, str) and v.strip() for v in values):
                    return [v.strip() for v in values]
                if key == "judgments":
                    return _assess_judgments(result, data)
                raise ValueError(f"Invalid {key}")
            except (KeyError, IndexError, TypeError, AttributeError, ValueError) as error:
                last_error = error
        raise BadReply(f"{key}, seed={seed}: {last_error}")

    def propose(self, problem, trace, sentence, seed):
        data = {"problem": problem, "solution": trace, "source_sentence": sentence}
        return self._ask(PROPOSE, data, "sentences", 3, seed)

    def assess(self, problem, trace, sentence, candidates, seed):
        data = {"problem": problem, "solution": trace, "source_sentence": sentence, "replacements": candidates}
        return self._ask(JUDGE, data, "judgments", len(candidates), seed)

    def judge(self, problem, trace, sentence, candidates, seed):
        """Return raw model ratings. assess() also returns each eligibility decision."""
        assessment = self.assess(problem, trace, sentence, candidates, seed)
        return [item["rating"] for item in assessment["judgments"]]
