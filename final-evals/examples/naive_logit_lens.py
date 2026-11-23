"""Example participant: a naive logit lens on the competition model, run end to end.

This is what a participant's code looks like. It writes ONE function,

    lens_fn(question) -> {"id": ..., "readouts": [{"layer": L, "tokens": [top tokens, best first]}, ...]}

and calls it once per public question (what `collect` does; this script loops itself so that it
can save as it goes and resume). The function sees only the
public part of a question: "id", "prompt", "model_input", "read_position", "read_token"
(never the intermediate, the answer or the options). The organisers then score the dicts with
`evaluate` (lens_eval/run.py).

The lens. thinkingmachines/Inkling-2-Mini is a looped model: 24 stored layers, of which layers 6-17 run
three times, so one token passes through 48 "layer visits" (numbered 0-47 in execution order; the
"layer" of a readout below is the visit number). It has no single residual stream; each visit
receives `hidden_states`, the 6-layer block in progress. The naive logit lens reads that directly:

    tokens(visit) = top-10 of unembed(norm(hidden_states entering that visit))   at the read position

What it does per question:
  1. tokenize question["model_input"] with no special tokens added;
  2. check that the token at question["read_position"] is question["read_token"];
  3. one forward pass, keeping `hidden_states` at that position for the 10 chosen visits;
  4. unembed each and return the top 10 token strings per visit.

Usage (from final-evals/, needs a GPU and the model in the HF cache):
    python examples/naive_logit_lens.py OUT_DIR [--limit N] [--device cuda] [--extra FILE.jsonl]
Writes OUT_DIR/submissions.jsonl (appended per question, so a stopped run resumes) and
OUT_DIR/output_read.jsonl, a control that is NOT a lens: the model's own next-token top 10 at
the same position, as a one-layer submission. Score either with
    python -m lens_eval.run OUT_DIR/submissions.jsonl --name my-run
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lens_eval import public_questions  # noqa: E402

REPO = "thinkingmachines/Inkling-2-Mini"
REV = "b85c3cfa615fc413cc9af3379f491b963f3b586a"
# The participant's choice of 10 layer visits (the same for every question): layer 12 in each of
# the three passes (visits 12, 24, 36), two visits before the last of those, and the last five.
VISITS = [12, 24, 34, 35, 36, 43, 44, 45, 46, 47]
TOP_K = 10


class NaiveLogitLens:
    def __init__(self, device="cuda", visits=VISITS, k=TOP_K):
        from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration
        torch.set_grad_enabled(False)
        self.visits, self.k = list(visits), k
        self.tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
        self.model = Inkling2LoopForConditionalGeneration.from_pretrained(
            REPO, revision=REV, dtype=torch.bfloat16, device_map=device).eval()
        self.llm = self.model.model.language_model
        self.cfg = self.llm.config
        # Every stored layer is entered once, and the looped ones once more per extra pass.
        self.num_visits = self.cfg.num_hidden_layers + (self.cfg.loop_num_passes - 1) * (
            self.cfg.loop_layer_index_end - self.cfg.loop_layer_index_start)
        self.unembed = self.model.lm_head.weight.float()    # covers the real vocabulary only
        self.device = self.unembed.device
        self.output_read = {}                         # question id -> the model's own next-token top k

    def _strings(self, logits):
        return [self.tok.decode([i]) or self.tok.convert_ids_to_tokens(i) for i in logits.topk(self.k).indices.tolist()]

    def _unembed(self, state):
        """norm -> unembed -> the model's logit soft cap."""
        logits = self.llm.norm(state.to(torch.bfloat16)).float() @ self.unembed.T
        return self.model.logit_softcap(logits)

    def __call__(self, question):
        ids = self.tok(question["model_input"], return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        pos = question["read_position"]
        here = self.tok.decode([ids[0, pos].item()]) or self.tok.convert_ids_to_tokens(ids[0, pos].item())
        assert here == question["read_token"], (question["id"], here, question["read_token"])

        partials = []                                 # `hidden_states` entering each of the 48 layer visits, at pos

        def keep(module, args, kwargs):
            state = args[0] if args else kwargs["hidden_states"]
            partials.append(state[0, pos].clone())

        hooks = [layer.register_forward_pre_hook(keep, with_kwargs=True) for layer in self.llm.layers]
        try:
            out = self.model(input_ids=ids)
        finally:
            for h in hooks:
                h.remove()
        assert len(partials) == self.num_visits, len(partials)

        logits = self._unembed(torch.stack([partials[v] for v in self.visits]))
        self.output_read[question["id"]] = self._strings(out.logits[0, pos].float())
        return {"id": question["id"],
                "readouts": [{"layer": v, "tokens": self._strings(logits[i])} for i, v in enumerate(self.visits)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int, default=0, help="how many questions to read, spread over the set (0 = all); for a smoke test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extra", help="JSONL of extra public questions to read as well")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    questions = public_questions()
    if a.limit:
        questions = questions[:: max(1, len(questions) // a.limit)][: a.limit]
    if a.extra:
        hidden = {"intermediate", "answer", "options", "gate"}
        questions += [{k: v for k, v in json.loads(l).items() if k not in hidden}
                      for l in Path(a.extra).read_text().splitlines() if l.strip()]

    sub_path, ctl_path = out / "submissions.jsonl", out / "output_read.jsonl"
    done = {json.loads(l)["id"] for l in sub_path.read_text().splitlines()} if sub_path.exists() else set()
    t0 = time.time()
    lens = NaiveLogitLens(a.device)
    print(f"model loaded in {time.time() - t0:.0f}s on {lens.device}; visits {lens.visits}", flush=True)
    t0 = time.time()
    for n, q in enumerate(questions, 1):
        if q["id"] in done:
            continue
        sub = lens(q)                                  # <- the participant's function
        with sub_path.open("a") as f:
            f.write(json.dumps(sub, ensure_ascii=False) + "\n")
        with ctl_path.open("a") as f:
            f.write(json.dumps({"id": q["id"], "readouts": [{"layer": "output", "tokens": lens.output_read[q["id"]]}]},
                               ensure_ascii=False) + "\n")
        if n <= 3 or n % 25 == 0 or n == len(questions):
            print(f"{n}/{len(questions)} {q['id']} {time.time() - t0:.0f}s  e.g. visit {sub['readouts'][-1]['layer']}: "
                  f"{sub['readouts'][-1]['tokens'][:5]}", flush=True)
    print("wrote", sub_path, "and", ctl_path)


if __name__ == "__main__":
    main()
