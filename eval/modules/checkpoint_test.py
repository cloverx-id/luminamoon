try:
    from .common import *
except (ImportError, ValueError):
    from common import *

import io
import torch
import torch.nn as nn

def test_checkpoint_roundtrip():
    torch.manual_seed(42)

    for buf in [1, 2]:
        for mw in [False, "semi", "full"]:
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 2)
            )
            opt = LuminaV(model.parameters(), lr=1e-3, buffer=buf, bound=True, master_weights=mw)

            for _ in range(5):
                x = torch.randn(4, 10)
                y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
                loss = nn.CrossEntropyLoss()(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            buffer_io = io.BytesIO()
            torch.save(opt.state_dict(), buffer_io)
            buffer_io.seek(0)

            opt_loaded = LuminaV(model.parameters(), lr=1e-3, buffer=buf, bound=True, master_weights=mw)
            loaded_dict = torch.load(buffer_io, weights_only=True)
            opt_loaded.load_state_dict(loaded_dict)

            for p_state in loaded_dict["state"].values():
                for forbidden_key in ("p_contig", "exp_avg_contig", "exp_avg_sq_contig", "mp_contig", "_plan"):
                    assert forbidden_key not in p_state, f"Memory leak detected: {forbidden_key} found in checkpoint!"

            x = torch.randn(4, 10)
            y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
            loss = nn.CrossEntropyLoss()(model(x), y)
            opt_loaded.zero_grad()
            loss.backward()
            opt_loaded.step()

    return True

def run_checkpoint_suite():
    print("\n--- [LSO-Eval] Software Integrity: Checkpoint Serialization Roundtrip ---")
    success = test_checkpoint_roundtrip()
    print(f"Checkpoint Load/Save & State Cleanliness: {'PASSED' if success else 'FAILED'}")

if __name__ == "__main__":
    run_checkpoint_suite()
