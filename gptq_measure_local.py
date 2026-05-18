"""gptq_measure_local_v4.py — Path B, attempt 3

Fixes a tiny bug in v3: auto_gptq's BaseGPTQForCausalLM.generate() doesn't
accept positional args the way transformers' generate() does. v3 was so
close — both models LOADED successfully and we were measuring perplexity
when throughput's m.generate(prompt, ...) hit the wall. v4 passes the
prompt as input_ids=prompt instead.

Reuses the fp16 numbers from v2/v3 runs.
"""
import os, time, math, csv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from auto_gptq import AutoGPTQForCausalLM

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.get_device_name(0) if dev.type=='cuda' else '(cpu)'}")
print(f"Torch: {torch.__version__}\n")

LOCAL_SMOLLM2_DIR = os.path.abspath("./SmolLM2-1.7B-GPTQ-Int4")

# fp16 baselines from previous runs (no need to remeasure)
FP16_KNOWN = {
    "SmolLM2-1.7B": {"weight_GB_fp16": 3.42, "ppl_fp16": 6.200, "tok_per_s_fp16": 39.7},
    "Qwen2-0.5B":   {"weight_GB_fp16": 0.99, "ppl_fp16": 11.978, "tok_per_s_fp16": 44.2},
}

GPTQ_CONFIGS = [
    ("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B",       LOCAL_SMOLLM2_DIR),
    ("Qwen2-0.5B",   "Qwen/Qwen2-0.5B-Instruct",         "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4"),
]


def ppl(m, ids, start):
    """Perplexity on labels[start:]. Works on both transformers and auto_gptq models."""
    with torch.no_grad():
        out = m(ids, use_cache=False)
    # auto_gptq's __call__ returns a tuple in some versions; handle both.
    logits = out.logits if hasattr(out, "logits") else out[0]
    sl = logits[:, :-1].contiguous().view(-1, logits.size(-1))
    lb = ids[:, 1:].contiguous().clone()
    lb[:, :max(0, start - 1)] = -100
    loss = F.cross_entropy(sl, lb.view(-1), ignore_index=-100)
    return math.exp(loss.item())


def throughput(m, tok, prompt, n_new=50):
    """Token/sec for a 50-token generate. Uses keyword args so it works on
    both transformers (positional ok) and auto_gptq (kwargs required)."""
    with torch.no_grad():
        m.generate(input_ids=prompt, max_new_tokens=10,
                   pad_token_id=tok.eos_token_id)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        m.generate(input_ids=prompt, max_new_tokens=n_new,
                   pad_token_id=tok.eos_token_id, use_cache=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return n_new / (time.perf_counter() - t0)


def weight_GB(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e9


print("Loading wikitext-2 test split...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
EVAL_TEXT = "\n\n".join(t for t in ds["text"] if t.strip())

rows = []
for mname, fp16_id, gptq_src in GPTQ_CONFIGS:
    print(f"\n=== {mname} ===")
    known = FP16_KNOWN[mname]
    row = {
        "model": mname,
        "weight_GB_fp16": known["weight_GB_fp16"],
        "ppl_fp16":       known["ppl_fp16"],
        "tok_per_s_fp16": known["tok_per_s_fp16"],
        "weight_GB_4bit": None,
        "ppl_gptq":       None,
        "tok_per_s_gptq": None,
        "compression":    None,
        "delta_ppl":      None,
        "note": "",
    }
    try:
        tok = AutoTokenizer.from_pretrained(fp16_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        ids = tok.encode(EVAL_TEXT, return_tensors="pt")[:, :2048].to(dev)
        prompt = ids[:, :512]

        print(f"  Loading GPTQ via auto_gptq direct ({gptq_src})...")
        m_gp = AutoGPTQForCausalLM.from_quantized(
            gptq_src, device_map="auto",
        ).eval()

        w_gp = weight_GB(m_gp)
        print(f"  Measuring GPTQ perplexity...")
        p_gp = ppl(m_gp, ids, 1024)
        print(f"  Measuring GPTQ throughput...")
        tps_gp = throughput(m_gp, tok, prompt)

        row.update({
            "weight_GB_4bit": round(w_gp, 3),
            "ppl_gptq":       round(p_gp, 3),
            "tok_per_s_gptq": round(tps_gp, 1),
            "compression":    round(known["weight_GB_fp16"] / w_gp, 2) if w_gp > 0 else None,
            "delta_ppl":      round(p_gp - known["ppl_fp16"], 3),
        })
        print(f"  GPTQ:  {w_gp:.2f}GB | ppl={p_gp:.3f} | {tps_gp:.1f} tok/s")
        if row["compression"]:
            print(f"   -> {row['compression']}x compression | delta_ppl={row['delta_ppl']:+.3f}")

        del m_gp
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        row["note"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"  FAILED: {row['note']}")

    rows.append(row)

out_csv = "results_gptq_local.csv"
with open(out_csv, "w", newline="") as f:
    fieldnames = ["model", "weight_GB_fp16", "weight_GB_4bit", "compression",
                  "ppl_fp16", "ppl_gptq", "delta_ppl",
                  "tok_per_s_fp16", "tok_per_s_gptq", "note"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"\nSaved -> {out_csv}")

print("\nFinal results:")
for r in rows:
    note = r["note"][:80] if r["note"] else "OK"
    print(f"  {r['model']:15s} | "
          f"fp16: ppl={r['ppl_fp16']} {r['weight_GB_fp16']}GB {r['tok_per_s_fp16']}t/s | "
          f"gptq: ppl={r['ppl_gptq']} {r['weight_GB_4bit']}GB {r['tok_per_s_gptq']}t/s | "
          f"compression={r['compression']}x delta_ppl={r['delta_ppl']} | {note}")
