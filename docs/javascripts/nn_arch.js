(function () {
  var MODULES = {
    sin3d: {
      title: "Sinusoidal3DEmbedding",
      module: "luminamoon.nn.position",
      shape: "(p_z, p_y, p_x) [...] \u2192 [..., dim]",
      desc: "Sinusoidal positional embedding for 3D (z, y, x) coordinate tensors using log-spaced frequency bands. Requires even dim >= 6 and accepts an optional output dtype.",
      options: "Sinusoidal3DEmbedding(dim, theta=10000.0). Splits dim into even channel counts across (z, y, x), assigning remaining channels to x.",
      note: "Useful for volumetric, spatial, or multimodal coordinate inputs prior to entering the backbone."
    },
    hub: {
      title: "OrthogonalHub",
      module: "luminamoon.nn.memory",
      shape: "batch_size B \u2192 [B, H, D]",
      desc: "Prefix hub tokens produced by applying an SO(H) orthogonal transform (float32 Cayley transform of a skew-symmetric matrix) to the learned hub basis.",
      options: "OrthogonalHub(hub_size=128, dim=512, use_context_gate=False). Caches the Cayley orthogonal matrix in eval mode.",
      note: "Keep use_context_gate=False for causal autoregressive language models because the optional context gate pools across the sequence dimension."
    },
    dkd: {
      title: "ReversibleDKDStack / ReversibleDKDBlock",
      module: "luminamoon.nn.blocks",
      shape: "(q, p): [B, L, D/2] \u2192 (q_next, p_next, kv_cache)",
      desc: "Volume-preserving Drift-Kick-Drift (DKD) split-stream block and multi-layer stack. Operates on split hidden-state halves (q, p) of width D/2 and alternates FFN drift steps with attention kick steps.",
      options: "Supports selective reversibility ('none', 'all', 'every', 'even', 'odd', 'first_half', 'second_half', 'indices'), custom kick_attn_fn / drift_ffn_fn factories (such as inserting RaMoE every N layers), whole-stack torch.compile when all layers are non-reversible, and CUDA Graph boundary guards for modes such as mode='reduce-overhead'.",
      note: "state_dict() keys remain identical between eager and compiled stacks. Total sequence length L = H + S when prefixed by OrthogonalHub; slice [:, H:, :] after concatenating (q_next, p_next)."
    },
    lkv: {
      title: "LatentKVAttention",
      module: "luminamoon.nn.attention",
      shape: "x [B, L, D/2] \u2192 (out [B, L, D/2], cache)",
      desc: "Low-rank Key-Value (KV) latent attention with pre-transpose QK-RaNorm, QRoPE, optional DistanceDecayBias, and incremental decoding cache (width dim, equal to D/2 inside DKD blocks).",
      options: "Supports cache_mode='latent' or 'kv' and attention_backend='auto' (inherits use_attention_backend(...) or PyTorch SDPA), 'sdpa', 'ra', 'flash', 'cudnn', 'fa3', 'flex', or a custom attention_fn.",
      note: "Computes non-learnable distance decay and hub causal masking in GPU shared memory (SRAM) via Triton Flash-RaAttention ('ra', with torch.library.custom_op for torch.compile) or compiled CUDA flex_attention ('flex')."
    },
    qrope: {
      title: "QRoPE / qrope",
      module: "luminamoon.nn.position",
      shape: "x [B, num_heads, L, head_dim] \u2192 [B, num_heads, L, head_dim]",
      desc: "Double-Quaternion Rotary Position Embedding applying Spin(4) left-right rotations on 4-element head blocks (head_dim % 4 == 0), preserving L2 norms and relative-position inner products.",
      options: "QRoPE(head_dim=32, max_seq_len=4096, base=10000.0, freq_ratio=0.6180339887498949). Supports forward(x, offset=0) for incremental KV-cache decoding.",
      note: "Uses a fused Triton CUDA kernel (python/luminamoon/nn/kernels/position.py) with analytic inverse-rotation backward on CUDA, falling back to eager PyTorch otherwise."
    },
    ddb: {
      title: "DistanceDecayBias",
      module: "luminamoon.nn.position",
      shape: "(seq_len, num_all, device, dtype) \u2192 [1, num_heads, seq_len, num_all]",
      desc: "Per-head additive distance decay bias (-s * d in 'linear' mode or -log(1 + s * d) in 'rational' mode, clamped to [-10.0, 0.0]) with optional prefix hub-token damping.",
      options: "DistanceDecayBias(num_heads=8, hub_size=0, ..., mode='linear' | 'rational', learnable=False). Keeps slopes positive via softplus when learnable=True.",
      note: "When learnable=False inside LatentKVAttention with attention_backend='ra' or 'flex', the bias is fused on the fly in SRAM instead of allocating an [N, N] matrix."
    },
    slot: {
      title: "SlotCrossAttention",
      module: "luminamoon.nn.attention",
      shape: "(terminal_base [B, T, D], context_states [B, S, D]) \u2192 [B, T\u00b7K, D]",
      desc: "Compresses context_states into a fixed set of summary slots using orthogonally initialized per-slot queries and Feature-wise Linear Modulation (FiLM) from terminal_base.",
      options: "SlotCrossAttention(dim, num_heads=8, num_terminals=16, slots_per_terminal=4, attention_backend='auto', attention_fn=None).",
      note: "For chunked long-context pipelines, choose T terminals and K = 4 slots per terminal so compression ratio L_chunk / (T * K) stays roughly in [4, 16]."
    },
    twotier: {
      title: "TwoTierMemory",
      module: "luminamoon.nn.memory",
      shape: "chunks [B, M, D] + query [B, Q, D] \u2192 (l1 [B, min(M, max_l1), D], l2 [B, min(M, max_l2), D])",
      desc: "Differentiable two-tier slot memory combining Tier-1 (L1, leading fixed slots) and Tier-2 (L2, query-gated selector using multi-head token-max LogSumExp scoring).",
      options: "TwoTierMemory(dim, max_l1=16, max_l2=32, num_heads=4, use_projections=True, use_topk=True, threshold=0.0, temperature=0.25). Supports soft-only mode (use_topk=False, returning full M slots in l2) during warmup.",
      note: "Call encode_keys(chunks) once per stored bank and pass cached_keys across repeated chunk queries to avoid redundant key projections."
    },
    dgate: {
      title: "DirectiveGate",
      module: "luminamoon.nn.memory",
      shape: "(chunks [B, N, D], directive_tokens [B, M, D]) \u2192 [B, N, D]",
      desc: "Pre-norm cosine gating and Feature-wise Linear Modulation (FiLM) modulating chunks from mean-pooled directive_tokens.",
      options: "DirectiveGate(dim, init_logit_scale=4.0, eps=1e-6). Bounds the cosine gate multiplier to (0.25, 2.0) and initializes FiLM weights with std=0.02.",
      note: "Pairs downstream of TwoTierMemory to gate and modulate stored chunk summaries."
    },
    rollout: {
      title: "ResidualRollout",
      module: "luminamoon.nn.blocks",
      shape: "x [B, S, D] \u2192 [B, S, D]",
      desc: "Weight-tied iterative residual refinement block repeated for num_steps passes before final normalization and readout.",
      options: "ResidualRollout(dim, num_steps=2, activation='ra_tanh' | 'tanh'). Defaults to rational ra_tanh activation.",
      note: "Adds refinement depth at readout without multiplying parameter count across steps."
    }
  };

  function initNnArch() {
    var shells = document.querySelectorAll("[data-nn-arch]");
    shells.forEach(function (shell) {
      if (shell.dataset.nnArchBound === "1") return;
      shell.dataset.nnArchBound = "1";

      var tabs = shell.querySelectorAll("[data-arch-tab]");
      var views = shell.querySelectorAll("[data-arch-view]");
      var hint = shell.querySelector("[data-arch-hint]");
      var nodes = shell.querySelectorAll("[data-mod]");
      var dirBtns = shell.querySelectorAll("[data-dkd-dir]");
      var dkdSvg = shell.querySelector("[data-dkd-svg]");
      var dkdFwd = shell.querySelector('[data-dkd-edges="fwd"]');
      var dkdBwd = shell.querySelector('[data-dkd-edges="bwd"]');
      var dkdOps = shell.querySelectorAll("[data-dkd-op]");
      var dkdSummary = shell.querySelector("[data-dkd-summary]");

      var insTitle = shell.querySelector('[data-ins="title"]');
      var insMod = shell.querySelector('[data-ins="module"]');
      var insShape = shell.querySelector('[data-ins="shape"]');
      var insDesc = shell.querySelector('[data-ins="desc"]');
      var insOptions = shell.querySelector('[data-ins="options"]');
      var insNote = shell.querySelector('[data-ins="note"]');

      function selectTab(tabId) {
        tabs.forEach(function (btn) {
          var active = btn.getAttribute("data-arch-tab") === tabId;
          btn.classList.toggle("is-active", active);
          btn.setAttribute("aria-selected", active ? "true" : "false");
        });
        views.forEach(function (view) {
          view.classList.toggle("is-active", view.getAttribute("data-arch-view") === tabId);
        });
        if (hint) {
          hint.style.display = tabId === "pipeline" ? "" : "none";
        }
      }

      function selectModule(modId) {
        var info = MODULES[modId];
        if (!info) return;
        nodes.forEach(function (node) {
          var selected = node.getAttribute("data-mod") === modId;
          node.classList.toggle("is-selected", selected);
          node.setAttribute("aria-pressed", selected ? "true" : "false");
        });
        if (insTitle) insTitle.textContent = info.title;
        if (insMod) insMod.textContent = info.module;
        if (insShape) insShape.textContent = info.shape;
        if (insDesc) insDesc.textContent = info.desc;
        if (insOptions) insOptions.textContent = info.options;
        if (insNote) insNote.textContent = info.note;
      }

      function setDkdDirection(dir) {
        var isFwd = dir !== "bwd";
        dirBtns.forEach(function (btn) {
          var active = btn.getAttribute("data-dkd-dir") === (isFwd ? "fwd" : "bwd");
          btn.classList.toggle("is-active", active);
          btn.setAttribute("aria-pressed", active ? "true" : "false");
        });
        if (dkdSvg) dkdSvg.classList.toggle("is-inverse", !isFwd);
        if (dkdFwd) dkdFwd.style.display = isFwd ? "" : "none";
        if (dkdBwd) dkdBwd.style.display = isFwd ? "none" : "";
        dkdOps.forEach(function (op) {
          op.textContent = isFwd ? "+" : "\u2212";
        });
        if (dkdSummary) {
          dkdSummary.textContent = isFwd
            ? "Forward: Drift 1 \u2192 Kick \u2192 Drift 2 on split streams (q, p)"
            : "Analytic Inverse (reversible=True): skips internal FFN/attention intermediates by reconstructing (q, p) via Drift 2 \u2192 Kick \u2192 Drift 1";
        }
      }

      tabs.forEach(function (btn) {
        btn.addEventListener("click", function () {
          selectTab(btn.getAttribute("data-arch-tab"));
        });
      });

      nodes.forEach(function (node) {
        node.addEventListener("click", function (ev) {
          ev.stopPropagation();
          selectModule(node.getAttribute("data-mod"));
        });
        node.addEventListener("keydown", function (ev) {
          if (ev.key === "Enter" || ev.key === " ") {
            ev.preventDefault();
            selectModule(node.getAttribute("data-mod"));
          }
        });
      });

      dirBtns.forEach(function (btn) {
        btn.addEventListener("click", function () {
          setDkdDirection(btn.getAttribute("data-dkd-dir"));
        });
      });

      selectModule("dkd");
      setDkdDirection("fwd");
    });
  }

  if (window.__lmNnArchInitialized) return;
  window.__lmNnArchInitialized = true;

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(initNnArch);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initNnArch);
  } else {
    initNnArch();
  }
})();
