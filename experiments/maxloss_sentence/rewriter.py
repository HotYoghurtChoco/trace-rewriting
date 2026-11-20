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
    'State the source claim once and each replacement claim in one short clause or equation. '
    'Name the entities involved. For a quantity, specify what it measures and its unit. '
    'For a proportion or rate, name the reference group or amount that supplies its denominator. '
    'Preserve who did what, time, operations and the direction of comparisons. '
    'Compare each claim to the source in both directions. Different wording can express the '
    'same relation. Matching numbers or a matching final answer do not establish equivalence. '
    'Rate 5 only when all facts and relations are equivalent, 4 for a minor omission or '
    'uncertainty, 3 for a material change (including a different reference amount), '
    '2 for a contradiction and 1 for an unrelated claim. '
    'Return one JSON object with source_claim (string) and judgments (an array in the '
    'original replacement order). Each judgment has claim (string) and rating (integer).'
)


def round_seeds(index, round_number):
    """Return the existing proposal and judge seeds for a one-based editing round."""
    proposal = 888 + 100 * index + 4 * (round_number - 1)
    return proposal, proposal + 2


class BadReply(ValueError):
    """The endpoint replied, but did not supply the requested structured output."""


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
                "max_tokens": 1024, "stream": False,
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
                    source_claim = result["source_claim"]
                    if not isinstance(source_claim, str) or not source_claim.strip():
                        raise ValueError("Missing source claim")
                    for judgment in values:
                        claim, rating = judgment["claim"], judgment["rating"]
                        if not isinstance(claim, str) or not claim.strip():
                            raise ValueError("Missing replacement claim")
                        if type(rating) is not int or not 1 <= rating <= 5:
                            raise ValueError("Invalid semantic rating")
                    return result
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
        assessment = self.assess(problem, trace, sentence, candidates, seed)
        return [item["rating"] for item in assessment["judgments"]]
