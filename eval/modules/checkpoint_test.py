try:
    from .common import LuminaV
except (ImportError, ValueError):
    from common import LuminaV

import copy
import io
import torch
import torch.nn as nn


def test_checkpoint_roundtrip_and_parity() -> bool:
    print("\nStage II.4 & II.5: Package Trajectory Parity & Checkpoint Serialization Audit")
    torch.manual_seed(42)
    forbidden_keys = (
        "p_contig",
        "exp_avg_contig",
        "exp_avg_sq_contig",
        "mp_contig",
        "_plan",
        "_contig_step",
    )

    for dtype in [torch.float32, torch.bfloat16]:
        for buf in [1, 2]:
            for mw in [False, "semi", "full"]:
                model_cont = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 2),
                ).to(dtype=dtype)
                model_res = copy.deepcopy(model_cont)

                opt_cont = LuminaV(model_cont.parameters(), lr=1e-3, buffer=buf, bound=True, master_weights=mw)
                opt_res = LuminaV(model_res.parameters(), lr=1e-3, buffer=buf, bound=True, master_weights=mw)

                batches = [
                    (torch.randn(4, 10, dtype=dtype), torch.tensor([0, 1, 0, 1], dtype=torch.long))
                    for _ in range(6)
                ]

                for i in range(5):
                    x, y = batches[i]
                    for m, opt in ((model_cont, opt_cont), (model_res, opt_res)):
                        loss = nn.CrossEntropyLoss()(m(x).float(), y)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                buffer_io = io.BytesIO()
                torch.save(opt_res.state_dict(), buffer_io)
                buffer_io.seek(0)

                opt_loaded = LuminaV(model_res.parameters(), lr=1e-3, buffer=buf, bound=True, master_weights=mw)
                loaded_dict = torch.load(buffer_io, weights_only=False)
                opt_loaded.load_state_dict(loaded_dict)

                for p_state in loaded_dict["state"].values():
                    for forbidden_key in forbidden_keys:
                        assert forbidden_key not in p_state, f"Memory leak: {forbidden_key} in state_dict"
                    if "scratch" in p_state:
                        sc = p_state["scratch"]
                        assert float(sc[0]) == 0.0 and float(sc[1]) == 0.0
                        assert float(sc[2]) == 0.0 and float(sc[5]) == 0.0

                for p in model_res.parameters():
                    st = opt_loaded.state[p]
                    if "scratch" in st:
                        assert st["scratch"].dtype == torch.float32
                    if "master_param" in st:
                        assert st["master_param"].dtype == torch.float32

                x_last, y_last = batches[5]
                for m, opt in ((model_cont, opt_cont), (model_res, opt_loaded)):
                    loss = nn.CrossEntropyLoss()(m(x_last).float(), y_last)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                for p_c, p_r in zip(model_cont.parameters(), model_res.parameters()):
                    torch.testing.assert_close(p_c, p_r, rtol=0.0, atol=0.0)

    print("  1. Forbidden staging keys leak check     : PASSED")
    print("  2. 6-slot scratchpad reset (s0,1,2,5=0)  : PASSED")
    print("  3. FP32 state & master_param preservation: PASSED")
    print("  4. Bit-exact resume trajectory (FP32/BF16): PASSED")
    return True


def run_checkpoint_suite() -> bool:
    return test_checkpoint_roundtrip_and_parity()


if __name__ == "__main__":
    run_checkpoint_suite()
