# LSO-Eval: Formal Verification & Benchmarking Suite

The **LSO-Eval Suite** provides formal mathematical verification, Z3 SMT solver invariant proofs, 2D trajectory convergence tests, and IEEE numerical parity audits for the **LuminaV Optimizer** and **LuminaMoon DataLoader** under **PROJECT LUMI**, developed by **Lumina Moon ([@cloverx-id](https://github.com/cloverx-id))** and **Silver Moon ([@cloverxion](https://github.com/cloverxion))**.

[Versi Bahasa Indonesia (README.id.md)](README.id.md)

---

## Suite Components

| Component | Path | Description |
| :--- | :--- | :--- |
| **Formal Specification** | [`lso_eval_specification.tex`](lso_eval_specification.tex) | Complete mathematical formulation, proof lemmas, and LaTeX manuscript (IEEE Two-Column layout). |
| **PDF Manuscript** | [`LSO-Eval.pdf`](LSO-Eval.pdf) | Compiled formal specification paper and proof records. |
| **Stage 1 (SMT & Precision)** | [`stage1_smt_and_precision.py`](stage1_smt_and_precision.py) | Z3 SMT solver proofs verifying bounded step size invariants, CLM non-divergence, and FP16/BF16 stochastic rounding guarantees. |
| **Execution Harness** | [`run_all.py`](run_all.py) | Master evaluation orchestrator running formal verification stages sequentially. |

---

## Running Verification

```bash
# Execute full formal verification suite
python -m eval

# Or run Stage 1 SMT proofs directly
python eval/stage1_smt_and_precision.py
```
