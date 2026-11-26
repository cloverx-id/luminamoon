---
search:
  boost: 1.5
hide:
  - toc
---

# Interactive Architecture Explorer (`luminamoon.nn`)

Interactive architecture diagrams for the PyTorch position, latent attention, reversible Drift-Kick-Drift (DKD) block, and two-tier memory modules in `luminamoon.nn`. Select any module block to inspect its tensor shapes, computation, and options.

<div class="nn-arch-shell" data-nn-arch>
  <div class="nn-arch-toolbar">
    <div class="nn-arch-tabs" role="tablist" aria-label="Architecture diagram views">
      <button type="button" class="nn-arch-tab is-active" role="tab" aria-selected="true" data-arch-tab="pipeline">
        1. Full Pipeline Diagram
      </button>
      <button type="button" class="nn-arch-tab" role="tab" aria-selected="false" data-arch-tab="dkd">
        2. Reversible DKD (q, p) Diagram
      </button>
      <button type="button" class="nn-arch-tab" role="tab" aria-selected="false" data-arch-tab="glossary">
        3. Acronyms &amp; Tensor Axes
      </button>
    </div>
    <div class="nn-arch-legend" data-arch-hint>
      <span class="nn-legend-chip"><span class="nn-legend-swatch nn-swatch-mod"></span>Clickable module</span>
      <span class="nn-legend-chip"><span class="nn-legend-swatch nn-swatch-op"></span>Static tensor op</span>
      <span class="nn-arch-hint">Click any module block to inspect details below</span>
    </div>
  </div>

  <!-- VIEW 1: TRUE SVG PIPELINE ARCHITECTURE DIAGRAM -->
  <div class="nn-arch-view is-active" role="tabpanel" data-arch-view="pipeline">
    <div class="nn-svg-frame">
      <svg class="nn-svg-diagram" viewBox="0 0 1240 580" role="img" aria-label="Full luminamoon.nn chunked sequence architecture diagram">
        <defs>
          <marker id="nn-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1.2 L 9 5 L 0 8.8 z" class="nn-svg-arrow-fill"></path>
          </marker>
          <marker id="nn-arrow-input" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1.2 L 9 5 L 0 8.8 z" class="nn-fill-input"></path>
          </marker>
          <marker id="nn-arrow-accent" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1.2 L 9 5 L 0 8.8 z" class="nn-svg-arrow-accent"></path>
          </marker>
          <marker id="nn-arrow-mem" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1.2 L 9 5 L 0 8.8 z" class="nn-svg-arrow-mem"></path>
          </marker>
        </defs>

        <!-- Stage Background Zones (84px clear channels between stages) -->
        <rect x="12" y="16" width="284" height="548" rx="14" class="nn-zone nn-zone-input"></rect>
        <text x="30" y="42" class="nn-zone-title nn-fill-input">1. INPUT &amp; PREFIX HUB</text>

        <rect x="380" y="16" width="466" height="548" rx="14" class="nn-zone nn-zone-core"></rect>
        <text x="398" y="42" class="nn-zone-title nn-fill-core">2. REVERSIBLE DKD CORE (&times; N LAYERS)</text>

        <rect x="930" y="16" width="298" height="548" rx="14" class="nn-zone nn-zone-mem"></rect>
        <text x="948" y="42" class="nn-zone-title nn-fill-mem">3. CHUNK MEMORY &amp; READOUT</text>

        <!-- ===== CONNECTORS & ARROWS (drawn behind nodes) ===== -->
        <!-- Input Tokens -> Concat (centered x=174) -->
        <path d="M 174 146 L 174 218" class="nn-edge" marker-end="url(#nn-arrow)"></path>
        <text x="184" y="186" class="nn-edge-label">[B, S, D]</text>

        <!-- Sinusoidal3D (optional dashed) -> Input Tokens via dedicated left rail at x=40 -->
        <path d="M 66 492 L 40 492 L 40 115 L 64 115" class="nn-edge nn-edge-dashed" marker-end="url(#nn-arrow)"></path>
        <text x="27" y="304" text-anchor="middle" class="nn-edge-label" transform="rotate(-90 27 304)">optional 3D coords</text>

        <!-- OrthogonalHub -> Concat (teal edge + matching teal arrowhead) -->
        <path d="M 174 348 L 174 282" class="nn-edge nn-edge-input" marker-end="url(#nn-arrow-input)"></path>
        <text x="184" y="319" class="nn-edge-label">[B, H, D]</text>

        <!-- Concat -> Split (q, p) across 84px channel (x=296..380) -->
        <path d="M 282 250 L 398 250" class="nn-edge nn-edge-accent" marker-end="url(#nn-arrow-accent)"></path>
        <rect x="298" y="222" width="80" height="20" rx="5" class="nn-label-pill"></rect>
        <text x="338" y="236" text-anchor="middle" class="nn-edge-label">[B, H+S, D]</text>

        <!-- Split -> Stack (centered x=444; labels cleanly above the elbow in open Zone 2 space) -->
        <path d="M 444 210 L 444 148 L 510 148" class="nn-edge nn-edge-accent" marker-end="url(#nn-arrow-accent)"></path>
        <text x="444" y="116" text-anchor="middle" class="nn-edge-label">(q, p)</text>
        <text x="444" y="132" text-anchor="middle" class="nn-edge-label">[B, L, D/2]</text>

        <!-- Stack -> Merge & Slice (centered x=668) -->
        <path d="M 668 404 L 668 462" class="nn-edge nn-edge-accent" marker-end="url(#nn-arrow-accent)"></path>
        <text x="678" y="437" class="nn-edge-label">(q_next, p_next)</text>

        <!-- Merge & Slice -> Branch Up to SlotCrossAttention (dedicated 84px channel x=846..930, rail x=888) -->
        <path d="M 818 496 L 888 496 L 888 114 L 948 114" class="nn-edge nn-edge-mem" marker-end="url(#nn-arrow-mem)"></path>
        <rect x="854" y="264" width="68" height="20" rx="5" class="nn-label-pill"></rect>
        <text x="888" y="278" text-anchor="middle" class="nn-edge-label">[B, S, D]</text>

        <!-- Merge & Slice -> Branch Direct to ResidualRollout (pill placed cleanly BELOW the horizontal arrow) -->
        <path d="M 818 496 L 948 496" class="nn-edge nn-edge-accent" marker-end="url(#nn-arrow-accent)"></path>
        <rect x="854" y="506" width="68" height="20" rx="5" class="nn-label-pill"></rect>
        <text x="888" y="520" text-anchor="middle" class="nn-edge-label">[B, S, D]</text>

        <!-- SlotCrossAttention -> TwoTierMemory (centered x=1080) -->
        <path d="M 1080 150 L 1080 194" class="nn-edge nn-edge-mem" marker-end="url(#nn-arrow-mem)"></path>
        <text x="1090" y="176" class="nn-edge-label">slots [B, T&middot;K, D]</text>

        <!-- TwoTierMemory -> DirectiveGate (centered x=1080) -->
        <path d="M 1080 268 L 1080 312" class="nn-edge nn-edge-mem" marker-end="url(#nn-arrow-mem)"></path>
        <text x="1090" y="294" class="nn-edge-label">(l1, l2)</text>

        <!-- DirectiveGate -> ResidualRollout (gated conditioning) -->
        <path d="M 1080 386 L 1080 458" class="nn-edge nn-edge-mem nn-edge-dashed" marker-end="url(#nn-arrow-mem)"></path>
        <text x="1090" y="426" class="nn-edge-label">gated context</text>

        <!-- ===== STAGE 1 NODES (x=66, width=216, balanced vertical rhythm) ===== -->
        <g class="nn-svg-static">
          <rect x="66" y="84" width="216" height="62" rx="10" class="nn-node-rect nn-node-op"></rect>
          <text x="174" y="111" text-anchor="middle" class="nn-svg-title">Token Embeddings</text>
          <text x="174" y="131" text-anchor="middle" class="nn-svg-sub">[B, S, D]</text>
        </g>

        <g class="nn-svg-static">
          <rect x="66" y="218" width="216" height="64" rx="10" class="nn-node-rect nn-node-op"></rect>
          <text x="174" y="245" text-anchor="middle" class="nn-svg-title">Concat Prefix Hub</text>
          <text x="174" y="265" text-anchor="middle" class="nn-svg-sub">cat(hub, tok) &rarr; [B, H+S, D]</text>
        </g>

        <g class="nn-svg-node" data-mod="hub" tabindex="0" role="button" aria-pressed="false" aria-label="Inspect OrthogonalHub">
          <rect x="66" y="348" width="216" height="68" rx="10" class="nn-node-rect"></rect>
          <circle cx="268" cy="362" r="3.5" class="nn-node-dot"></circle>
          <text x="174" y="377" text-anchor="middle" class="nn-svg-title">OrthogonalHub</text>
          <text x="174" y="397" text-anchor="middle" class="nn-svg-sub">SO(H) Cayley [B, H, D]</text>
        </g>

        <g class="nn-svg-node" data-mod="sin3d" tabindex="0" role="button" aria-pressed="false" aria-label="Inspect Sinusoidal3DEmbedding">
          <rect x="66" y="458" width="216" height="68" rx="10" class="nn-node-rect"></rect>
          <circle cx="268" cy="472" r="3.5" class="nn-node-dot"></circle>
          <text x="174" y="487" text-anchor="middle" class="nn-svg-title">Sinusoidal3DEmbedding</text>
          <text x="174" y="507" text-anchor="middle" class="nn-svg-sub">3D (z, y, x) coords</text>
        </g>

        <!-- ===== STAGE 2 NODES ===== -->
        <g class="nn-svg-static">
          <rect x="398" y="210" width="92" height="80" rx="10" class="nn-node-rect nn-node-op"></rect>
          <text x="444" y="241" text-anchor="middle" class="nn-svg-title">Split</text>
          <text x="444" y="260" text-anchor="middle" class="nn-svg-sub">(q, p)</text>
          <text x="444" y="277" text-anchor="middle" class="nn-svg-sub">D &rarr; D/2</text>
        </g>

        <!-- ReversibleDKDStack Outer Container (x=510, width=316) -->
        <g class="nn-svg-node is-selected" data-mod="dkd" tabindex="0" role="button" aria-pressed="true" aria-label="Inspect ReversibleDKDStack">
          <rect x="510" y="64" width="316" height="340" rx="12" class="nn-node-rect nn-node-container"></rect>
          <circle cx="810" cy="80" r="3.8" class="nn-node-dot"></circle>
          <text x="668" y="92" text-anchor="middle" class="nn-svg-title">ReversibleDKDStack (&times; N)</text>
          <text x="668" y="112" text-anchor="middle" class="nn-svg-sub">Drift &rarr; Kick &rarr; Drift &bull; Whole-Stack Compile</text>
        </g>

        <g class="nn-svg-node" data-mod="lkv" tabindex="0" role="button" aria-pressed="false" aria-label="Inspect LatentKVAttention">
          <rect x="528" y="130" width="280" height="64" rx="9" class="nn-node-rect nn-node-inner"></rect>
          <circle cx="794" cy="144" r="3.5" class="nn-node-dot"></circle>
          <text x="668" y="157" text-anchor="middle" class="nn-svg-title">LatentKVAttention (Kick)</text>
          <text x="668" y="177" text-anchor="middle" class="nn-svg-sub">Low-rank KV &bull; SDPA / Triton RA / Flex</text>
        </g>

        <!-- 32px clean arrow stems from QRoPE (x=595) & DistanceDecay (x=741) -->
        <path d="M 595 226 L 595 194" class="nn-edge" marker-end="url(#nn-arrow)"></path>
        <path d="M 741 226 L 741 194" class="nn-edge" marker-end="url(#nn-arrow)"></path>

        <g class="nn-svg-node" data-mod="qrope" tabindex="0" role="button" aria-pressed="false" aria-label="Inspect QRoPE">
          <rect x="528" y="226" width="134" height="66" rx="9" class="nn-node-rect nn-node-inner"></rect>
          <circle cx="649" cy="240" r="3.2" class="nn-node-dot"></circle>
          <text x="595" y="255" text-anchor="middle" class="nn-svg-title">QRoPE</text>
          <text x="595" y="275" text-anchor="middle" class="nn-svg-sub">Spin(4) 4D Rotary</text>
        </g>

        <g class="nn-svg-node" data-mod="ddb" tabindex="0" role="button" aria-pressed="false" aria-label="Inspect DistanceDecayBias">
          <rect x="674" y="226" width="134" height="66" rx="9" class="nn-node-rect nn-node-inner"></rect>
          <circle cx="795" cy="240" r="3.2" class="nn-node-dot"></circle>
          <text x="741" y="255" text-anchor="middle" class="nn-svg-title">DistanceDecay</text>
          <text x="741" y="275" text-anchor="middle" class="nn-svg-sub">SRAM Bias + Hub</text>
        </g>

        <g class="nn-svg-static">
          <rect x="528" y="316" width="280" height="64" rx="9" class="nn-node-rect nn-node-op"></rect>
          <text x="668" y="344" text-anchor="middle" class="nn-svg-title">Drift FFN (RaSwiGLU / RaMoE)</text>
          <text x="668" y="364" text-anchor="middle" class="nn-svg-sub">Custom drift_ffn_fn per layer</text>
        </g>

        <!-- Merge & Slice Hub -->
        <g class="nn-svg-static">
          <rect x="518" y="462" width="300" height="68" rx="10" class="nn-node-rect nn-node-op"></rect>
          <text x="668" y="491" text-anchor="middle" class="nn-svg-title">Concat (q, p) &amp; Slice Hub</text>
          <text x="668" y="511" text-anchor="middle" class="nn-svg-sub">cat(q, p)[:, H:, :] &rarr; [B, S, D]</text>
        </g>

        <!-- ===== STAGE 3 NODES (x=948, width=264, centered x=1080) ===== -->
        <g class="nn-svg-node" data-mod="slot" tabindex="0" role="button" aria-pressed="false" aria-label="Inspect SlotCrossAttention">
          <rect x="948" y="76" width="264" height="74" rx="10" class="nn-node-rect"></rect>
          <circle cx="1198" cy="90" r="3.5" class="nn-node-dot"></circle>
          <text x="1080" y="108" text-anchor="middle" class="nn-svg-title">SlotCrossAttention</text>
          <text x="1080" y="129" text-anchor="middle" class="nn-svg-sub">Chunk [B, S, D] &rarr; Slots [B, T&middot;K, D]</text>
        </g>

        <g class="nn-svg-node" data-mod="twotier" tabindex="0" role="button" aria-pressed="false" aria-label="Inspect TwoTierMemory">
          <rect x="948" y="194" width="264" height="74" rx="10" class="nn-node-rect"></rect>
          <circle cx="1198" cy="208" r="3.5" class="nn-node-dot"></circle>
          <text x="1080" y="226" text-anchor="middle" class="nn-svg-title">TwoTierMemory</text>
          <text x="1080" y="247" text-anchor="middle" class="nn-svg-sub">Tier-1 (L1) Fixed + Tier-2 (L2) Gated</text>
        </g>

        <g class="nn-svg-node" data-mod="dgate" tabindex="0" role="button" aria-pressed="false" aria-label="Inspect DirectiveGate">
          <rect x="948" y="312" width="264" height="74" rx="10" class="nn-node-rect"></rect>
          <circle cx="1198" cy="326" r="3.5" class="nn-node-dot"></circle>
          <text x="1080" y="344" text-anchor="middle" class="nn-svg-title">DirectiveGate</text>
          <text x="1080" y="365" text-anchor="middle" class="nn-svg-sub">Cosine Gating + FiLM Conditioning</text>
        </g>

        <g class="nn-svg-node" data-mod="rollout" tabindex="0" role="button" aria-pressed="false" aria-label="Inspect ResidualRollout">
          <rect x="948" y="458" width="264" height="74" rx="10" class="nn-node-rect"></rect>
          <circle cx="1198" cy="472" r="3.5" class="nn-node-dot"></circle>
          <text x="1080" y="490" text-anchor="middle" class="nn-svg-title">ResidualRollout &rarr; RaNorm</text>
          <text x="1080" y="511" text-anchor="middle" class="nn-svg-sub">Weight-tied N-step &rarr; Readout</text>
        </g>
      </svg>
    </div>

    <!-- Inspector Panel -->
    <div class="nn-inspector" aria-live="polite">
      <div class="nn-inspector-head">
        <div class="nn-inspector-title-group">
          <strong class="nn-inspector-title" data-ins="title">ReversibleDKDStack / ReversibleDKDBlock</strong>
          <code class="nn-inspector-mod" data-ins="module">luminamoon.nn.blocks</code>
        </div>
        <code class="nn-inspector-shape" data-ins="shape">(q, p): [B, L, D/2] &rarr; (q_next, p_next, kv_cache)</code>
      </div>
      <p class="nn-inspector-desc" data-ins="desc">
        Volume-preserving Drift-Kick-Drift (DKD) split-stream block and multi-layer stack. Operates on split hidden-state halves (q, p) of width D/2 and alternates FFN drift steps with attention kick steps.
      </p>
      <div class="nn-inspector-meta">
        <div class="nn-inspector-box">
          <span class="nn-inspector-label">Options &amp; Compilation</span>
          <p class="nn-inspector-text" data-ins="options">
            Supports selective reversibility ("none", "all", "every", "even", "odd", "first_half", "second_half", "indices"), custom kick_attn_fn / drift_ffn_fn factories (such as inserting RaMoE every N layers), whole-stack torch.compile when all layers are non-reversible, and CUDA Graph boundary guards for modes such as mode="reduce-overhead".
          </p>
        </div>
        <div class="nn-inspector-box">
          <span class="nn-inspector-label">Integration Note</span>
          <p class="nn-inspector-text" data-ins="note">
            state_dict() keys remain identical between eager and compiled stacks. Total sequence length L = H + S when prefixed by OrthogonalHub; slice [:, H:, :] after concatenating (q_next, p_next).
          </p>
        </div>
      </div>
    </div>
  </div>

  <!-- VIEW 2: TRUE SVG SYMPLECTIC DKD LADDER DIAGRAM -->
  <div class="nn-arch-view" role="tabpanel" data-arch-view="dkd">
    <div class="nn-dkd-bar">
      <div class="nn-dkd-dir-group" role="group" aria-label="DKD pass direction">
        <button type="button" class="nn-dir-btn is-active" aria-pressed="true" data-dkd-dir="fwd">Forward Pass (+)</button>
        <button type="button" class="nn-dir-btn" aria-pressed="false" data-dkd-dir="bwd">Analytic Inverse (&minus;)</button>
      </div>
      <span class="nn-dkd-subtitle" data-dkd-summary>Forward: Drift 1 &rarr; Kick &rarr; Drift 2 on split streams (q, p)</span>
    </div>

    <div class="nn-svg-frame">
      <svg class="nn-svg-diagram" data-dkd-svg viewBox="0 0 1240 364" role="img" aria-label="Reversible Drift-Kick-Drift dual-stream schematic diagram">
        <defs>
          <marker id="dkd-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 1.2 L 9 5 L 0 8.8 z" class="nn-svg-arrow-accent"></path>
          </marker>
          <marker id="dkd-arrow-rev" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 1.2 L 9 5 L 0 8.8 z" class="nn-svg-arrow-mem"></path>
          </marker>
        </defs>

        <!-- Rail Terminals -->
        <rect x="18" y="54" width="132" height="56" rx="10" class="nn-node-rect nn-node-op"></rect>
        <text x="84" y="79" text-anchor="middle" class="nn-svg-title">q (Position)</text>
        <text x="84" y="98" text-anchor="middle" class="nn-svg-sub">[B, L, D/2]</text>

        <rect x="18" y="254" width="132" height="56" rx="10" class="nn-node-rect nn-node-op"></rect>
        <text x="84" y="279" text-anchor="middle" class="nn-svg-title">p (Momentum)</text>
        <text x="84" y="298" text-anchor="middle" class="nn-svg-sub">[B, L, D/2]</text>

        <rect x="1090" y="54" width="132" height="56" rx="10" class="nn-node-rect nn-node-op"></rect>
        <text x="1156" y="79" text-anchor="middle" class="nn-svg-title">q_next</text>
        <text x="1156" y="98" text-anchor="middle" class="nn-svg-sub">[B, L, D/2]</text>

        <rect x="1090" y="254" width="132" height="56" rx="10" class="nn-node-rect nn-node-op"></rect>
        <text x="1156" y="279" text-anchor="middle" class="nn-svg-title">p_next</text>
        <text x="1156" y="298" text-anchor="middle" class="nn-svg-sub">[B, L, D/2]</text>

        <!-- Forward rails -->
        <g data-dkd-edges="fwd">
          <!-- q rail (y=82) -->
          <path d="M 150 82 L 298 82" class="nn-edge nn-edge-accent" marker-end="url(#dkd-arrow)"></path>
          <path d="M 336 82 L 904 82" class="nn-edge nn-edge-accent" marker-end="url(#dkd-arrow)"></path>
          <path d="M 942 82 L 1090 82" class="nn-edge nn-edge-accent" marker-end="url(#dkd-arrow)"></path>

          <!-- p rail (y=282) -->
          <path d="M 150 282 L 601 282" class="nn-edge nn-edge-accent" marker-end="url(#dkd-arrow)"></path>
          <path d="M 639 282 L 1090 282" class="nn-edge nn-edge-accent" marker-end="url(#dkd-arrow)"></path>

          <!-- Cross taps: Drift 1 (x=317), Kick (x=620), Drift 2 (x=923) -->
          <path d="M 317 282 L 317 222" class="nn-edge nn-edge-accent" marker-end="url(#dkd-arrow)"></path>
          <path d="M 317 142 L 317 101" class="nn-edge nn-edge-accent" marker-end="url(#dkd-arrow)"></path>

          <path d="M 620 82 L 620 142" class="nn-edge nn-edge-accent" marker-end="url(#dkd-arrow)"></path>
          <path d="M 620 222 L 620 263" class="nn-edge nn-edge-accent" marker-end="url(#dkd-arrow)"></path>

          <path d="M 923 282 L 923 222" class="nn-edge nn-edge-accent" marker-end="url(#dkd-arrow)"></path>
          <path d="M 923 142 L 923 101" class="nn-edge nn-edge-accent" marker-end="url(#dkd-arrow)"></path>
        </g>

        <!-- Inverse rails -->
        <g data-dkd-edges="bwd" style="display: none;">
          <!-- q rail reversed -->
          <path d="M 1090 82 L 942 82" class="nn-edge nn-edge-mem" marker-end="url(#dkd-arrow-rev)"></path>
          <path d="M 904 82 L 336 82" class="nn-edge nn-edge-mem" marker-end="url(#dkd-arrow-rev)"></path>
          <path d="M 298 82 L 150 82" class="nn-edge nn-edge-mem" marker-end="url(#dkd-arrow-rev)"></path>

          <!-- p rail reversed -->
          <path d="M 1090 282 L 639 282" class="nn-edge nn-edge-mem" marker-end="url(#dkd-arrow-rev)"></path>
          <path d="M 601 282 L 150 282" class="nn-edge nn-edge-mem" marker-end="url(#dkd-arrow-rev)"></path>

          <!-- Cross taps -->
          <path d="M 923 282 L 923 222" class="nn-edge nn-edge-mem nn-edge-dashed" marker-end="url(#dkd-arrow-rev)"></path>
          <path d="M 923 142 L 923 101" class="nn-edge nn-edge-mem nn-edge-dashed" marker-end="url(#dkd-arrow-rev)"></path>

          <path d="M 620 82 L 620 142" class="nn-edge nn-edge-mem nn-edge-dashed" marker-end="url(#dkd-arrow-rev)"></path>
          <path d="M 620 222 L 620 263" class="nn-edge nn-edge-mem nn-edge-dashed" marker-end="url(#dkd-arrow-rev)"></path>

          <path d="M 317 282 L 317 222" class="nn-edge nn-edge-mem nn-edge-dashed" marker-end="url(#dkd-arrow-rev)"></path>
          <path d="M 317 142 L 317 101" class="nn-edge nn-edge-mem nn-edge-dashed" marker-end="url(#dkd-arrow-rev)"></path>
        </g>

        <!-- Symmetric midpoint rail labels -->
        <text x="468" y="68" text-anchor="middle" class="nn-edge-label">q_half</text>
        <text x="772" y="68" text-anchor="middle" class="nn-edge-label">q_half</text>
        <text x="468" y="305" text-anchor="middle" class="nn-edge-label">p</text>
        <text x="772" y="305" text-anchor="middle" class="nn-edge-label">p_next</text>

        <!-- Rail tap junction dots -->
        <circle cx="317" cy="282" r="3.5" class="nn-tap-dot"></circle>
        <circle cx="620" cy="82" r="3.5" class="nn-tap-dot"></circle>
        <circle cx="923" cy="282" r="3.5" class="nn-tap-dot"></circle>

        <!-- Step 1: Drift 1 (width=278) -->
        <rect x="178" y="142" width="278" height="80" rx="10" class="nn-node-rect"></rect>
        <text x="317" y="173" text-anchor="middle" class="nn-svg-title">Drift 1 (FFN)</text>
        <text x="317" y="196" text-anchor="middle" class="nn-svg-sub">0.5 &middot; dt &middot; drift_ffn(norm_p(p))</text>

        <!-- Step 2: Kick (width=278) -->
        <rect x="481" y="142" width="278" height="80" rx="10" class="nn-node-rect nn-dkd-kick"></rect>
        <text x="620" y="173" text-anchor="middle" class="nn-svg-title">Kick (Attention)</text>
        <text x="620" y="196" text-anchor="middle" class="nn-svg-sub">dt &middot; LatentKV(norm_q(q_half))</text>

        <!-- Step 3: Drift 2 (width=278) -->
        <rect x="784" y="142" width="278" height="80" rx="10" class="nn-node-rect"></rect>
        <text x="923" y="173" text-anchor="middle" class="nn-svg-title">Drift 2 (FFN)</text>
        <text x="923" y="196" text-anchor="middle" class="nn-svg-sub">0.5 &middot; dt &middot; drift_ffn(norm_p(p_next))</text>

        <!-- Residual Sum / Subtract Circles (centrally aligned via dominant-baseline) -->
        <circle cx="317" cy="82" r="18" class="nn-op-circle"></circle>
        <text x="317" y="82" text-anchor="middle" dominant-baseline="central" class="nn-op-symbol" data-dkd-op>+</text>

        <circle cx="620" cy="282" r="18" class="nn-op-circle"></circle>
        <text x="620" y="282" text-anchor="middle" dominant-baseline="central" class="nn-op-symbol" data-dkd-op>+</text>

        <circle cx="923" cy="82" r="18" class="nn-op-circle"></circle>
        <text x="923" y="82" text-anchor="middle" dominant-baseline="central" class="nn-op-symbol" data-dkd-op>+</text>
      </svg>
    </div>

    <div class="nn-mode-grid">
      <div class="nn-mode-card">
        <div class="nn-mode-head">
          <code>selective_type="none"</code>
          <span class="nn-mode-tag">Throughput</span>
        </div>
        <p>Standard autograd on all layers. Enables whole-stack <code>torch.compile</code> and automatic CUDA Graph boundary guards under <code>mode="reduce-overhead"</code>.</p>
      </div>
      <div class="nn-mode-card">
        <div class="nn-mode-head">
          <code>selective_type="every", every=4</code>
          <span class="nn-mode-tag">Balanced</span>
        </div>
        <p>Applies reversible recomputation on every 4th layer while keeping standard autograd on intermediate layers to balance VRAM and training speed.</p>
      </div>
      <div class="nn-mode-card">
        <div class="nn-mode-head">
          <code>selective_type="all"</code>
          <span class="nn-mode-tag">Low VRAM</span>
        </div>
        <p>Skips storing internal FFN and attention intermediates at every layer by reconstructing <code>(q, p)</code> analytically during backward recomputation.</p>
      </div>
    </div>
  </div>

  <!-- VIEW 3: Acronyms & Tensor Shapes -->
  <div class="nn-arch-view" role="tabpanel" data-arch-view="glossary">
    <div class="nn-glossary-grid">
      <div class="nn-glossary-item">
        <strong>DKD</strong>
        <span>Drift-Kick-Drift: volume-preserving split-stream update alternating FFN drift steps and attention kick steps on <code>(q, p)</code>.</span>
      </div>
      <div class="nn-glossary-item">
        <strong>QRoPE</strong>
        <span>Double-Quaternion Rotary Position Embedding: <code>Spin(4)</code> left-right rotation on 4-element head blocks (<code>head_dim % 4 == 0</code>).</span>
      </div>
      <div class="nn-glossary-item">
        <strong>KV / LatentKV</strong>
        <span>Key-Value / Low-Rank Latent Key-Value: projects keys and values through a shared low-rank latent space before head expansion.</span>
      </div>
      <div class="nn-glossary-item">
        <strong>SDPA</strong>
        <span>Scaled Dot-Product Attention: PyTorch's built-in attention dispatcher (<code>F.scaled_dot_product_attention</code>).</span>
      </div>
      <div class="nn-glossary-item">
        <strong>SRAM</strong>
        <span>GPU on-chip shared memory: used by Triton <code>Flash-RaAttention</code> to compute distance decay and hub masking without materializing an <code>[N, N]</code> bias matrix.</span>
      </div>
      <div class="nn-glossary-item">
        <strong>FiLM</strong>
        <span>Feature-wise Linear Modulation: applies learned per-channel scale and shift to condition features from a context summary.</span>
      </div>
      <div class="nn-glossary-item">
        <strong>Tier-1 (L1) / Tier-2 (L2)</strong>
        <span>The two slot tiers inside <code>TwoTierMemory</code>: leading fixed slots (Tier-1 <code>L1</code>) paired with query-gated multi-head selected slots (Tier-2 <code>L2</code>) &mdash; distinct from L<sub>1</sub> / L<sub>2</sub> vector norms.</span>
      </div>
      <div class="nn-glossary-item">
        <strong>Tensor Axes</strong>
        <span><code>B</code> = batch size, <code>S</code> = token sequence length, <code>H</code> = prefix hub tokens, <code>L = H + S</code>, <code>D</code> = model width, <code>T</code> = <code>num_terminals</code>, <code>K</code> = <code>slots_per_terminal</code> (total slots <code>T &middot; K</code>).</span>
      </div>
    </div>
  </div>
</div>

For API signatures, formulas, and runnable PyTorch examples, see the [Ra-Family Architecture Overview](README.md) and [Configuration & API Reference](../en/03_configuration_api.md).
