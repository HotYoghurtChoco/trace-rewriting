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
    'For each replacement in order, compare it to the specified source sentence in context. '
    'Check both directions: all original facts must survive and no new claim may be introduced. '
    'Check the entity, owner, unit, time, operation, comparison direction and scope of every '
    'quantity. The same final answer alone is insufficient. Rate the meaning from 1 to 5: '
    '5 = fully equivalent, 4 = minor fact missing or uncertain, 3 = material change, '
    '2 = contradicts an important fact, 1 = unrelated. Return {"ratings":[...]} with one '
    'integer per replacement in the original order.'
)


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
                values = json.loads(content)[key]
                if not isinstance(values, list) or len(values) != size:
                    raise ValueError(f"Expected {size} {key}")
                if key == "sentences" and all(isinstance(v, str) and v.strip() for v in values):
                    return [v.strip() for v in values]
                if key == "ratings" and all(type(v) is int and 1 <= v <= 5 for v in values):
                    return values
                raise ValueError(f"Invalid {key}")
            except (KeyError, IndexError, TypeError, AttributeError, ValueError) as error:
                last_error = error
        raise BadReply(f"{key}, seed={seed}: {last_error}")

    def propose(self, problem, trace, sentence, seed):
        data = {"problem": problem, "solution": trace, "source_sentence": sentence}
        return self._ask(PROPOSE, data, "sentences", 3, seed)

    def judge(self, problem, trace, sentence, candidates, seed):
        data = {"problem": problem, "solution": trace, "source_sentence": sentence, "replacements": candidates}
        return self._ask(JUDGE, data, "ratings", len(candidates), seed)
