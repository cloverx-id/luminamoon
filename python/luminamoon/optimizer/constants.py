from __future__ import annotations

DEFAULT_LR = 8.0e-4
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1.0e-8
DEFAULT_WEIGHT_DECAY = 8.0e-2
DEFAULT_TAU = 0.8
DEFAULT_LAM = 0.3
DEFAULT_CAUTIOUS = True
DEFAULT_CAUTIOUS_CLAMP_MIN = 0.5
DEFAULT_BUFFER = 2
DEFAULT_STOCHASTIC_ROUNDING = True
DEFAULT_BOUND = True
DEFAULT_BOUND_TYPE = "coordinate"
DEFAULT_BOUND_RATIO = 0.05
DEFAULT_MASTER_WEIGHTS = "none"
DEFAULT_EXECUTION = "auto"
DEFAULT_FUSED_SINGLE_PASS = True
DEFAULT_EMA_DECAY = "auto"
DEFAULT_FSDP_SYNC = False
DEFAULT_OPTIMIZER_SEED = 1337
DEFAULT_ALGORITHM_SPEC = "1.5.4"
DEFAULT_GPU_DISPATCH_THRESHOLD = 16384
SR_QUANTIZE_CHUNK_SIZE = 1_048_576

VALID_BUFFER_CANDIDATES = ("1", "2", "single", "dual", "tdclm")
VALID_BOUND_TYPE_CANDIDATES = ("coordinate", "radial")
VALID_MASTER_WEIGHTS_CANDIDATES = ("none", "semi", "full", "false")
VALID_EXECUTION_MODE_CANDIDATES = (
    "auto",
    "vulkan",
    "simd",
    "scalar",
    "triton",
    "single",
    "foreach",
)

TEMP_STATE_DICT_KEYS = frozenset(
    {
        "p_contig",
        "exp_avg_contig",
        "exp_avg_sq_contig",
        "mp_contig",
        "_plan",
        "_contig_step",
    }
)
