# MoonLight-Scarlett

An Xbox Series X|S (Scarlett) compatibility layer for Windows, forked from [avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX) / [Marvisak/VIctory](https://github.com/Marvisak/VIctory) and engineered for **8 GB VRAM NVIDIA GPUs & 16 GB RAM Windows laptops**.

> **Notice — Source Code Only:** This repository contains **no game files, executables, assets, icons, dumps, or keys**. Bring your own legally dumped console files (`victory.toml`). This project is not affiliated with, endorsed by, or associated with any game developer, publisher, or console manufacturer.

---

## Measured Performance & 30-Minute Endurance Validation (2026-11-26)

- **Test Hardware:** Windows 11 x64, Intel Core i9-13900H, 16 GB System RAM, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM, Wave32)
- **Profile:** `--console series-s`, native `1920x1080` output @ `0.50` internal scale (`960x540` -> `1080p` temporal upscaler), ray tracing off

| Configuration | Result | Notes |
|---|---|---|
| Unmodified upstream baseline on this laptop | ~5.4 FPS, ~40 GB RAM pressure | 61.9 s residency stall, black opaque wall tiles on NVIDIA Wave32 |
| Tuned `VICTORY_RENDER_SCALE=0.45` | **22.7 FPS max** | Peak world framerate |
| Tuned `VICTORY_RENDER_SCALE=0.50` (`1080p`) | **15.9 FPS median** | Baked default resolution/scale preset |
| **30-Minute Continuous Endurance (`1800 s`, baked defaults, zero overrides)** | **15.1 FPS median** (mean `14.76`, p99 `16.0`, max `16.3`) | **0 crashes, 0 errors, 27,038 frames rendered** |

### 30-Minute Steady-State Windows (`t = 180..1805 s`)

| Metric | Measured Range Across 5-Min Windows |
|---|---|
| Window Median FPS | `15.20 / 15.00 / 15.20 / 15.10 / 14.90 FPS` |
| Window 1% Low FPS | `14.50 / 13.80 / 13.90 / 14.30 / 13.90 FPS` |
| Median Frame Time | `65.6 – 66.9 ms` |
| Dedicated VRAM | `5,284 – 5,304 MiB` (flat, no leak) |
| Private System RAM | `16,144 – 16,282 MB` (flat, no spill) |
| Peak GPU Temperature | `82.0 °C` |

> **Cycle 10 integrated (awaiting next plugged-in live run):** Merged upstream `VICTORY_PORT_GUARD=1` (64 KiB block claims) & `VICTORY_PORT_JOURNAL=1` (64-shard `Notes` combined with our `Shard::coarse` 16 MiB buckets), plus `VICTORY_TABLE_STAGES=1` (`stage_spans`), `SHARED_SPARE` 48 MiB upload buffer pool, and `WIDE_JOB = 256 KiB` (`454/454` unit tests passing in `victory-gpu` + `victory-loader`).

---

## NVIDIA, 8 GB VRAM & 16 GB RAM Optimizations

Every optimization below lives in Rust source code and is enabled safely in default runs (`apply_runtime_defaults` in `apps/victory/src/main.rs` and the GPU/HLE/shader crates):

| Area | Implementation | Effect |
|---|---|---|
| **Write-Watch & Constant Echo** | Unwatched 4 KiB diffing (`shadow.rs`) + `placed_cb_pool` (`list.rs`, `host/place.rs`) + lock-free ranged `Shadow` seqlock | Eliminates ~21 MiB/frame unwatched card-echo copy loops & `BEHIND_PLACED` stalls |
| **Sharded Journal & Coarse Skips** | 64-shard `Notes` with `Shard::coarse` 16 MiB bitmasks + `VICTORY_PORT_GUARD=1` / `VICTORY_PORT_JOURNAL=1` (`watch.rs`, `host.rs`) | Combines upstream's lock-sharded guards/journals with O(1) coarse bucket skipping |
| **Submission & Descriptor Hot Paths** | Fused single-pass `Batch::refresh_front` (`CHUNK=128`), O(1) `stale_under_flights` skip, zero-alloc `card_steps::plan`, `u64` `port_descriptors::take`, `VICTORY_TABLE_STAGES=1` (`host.rs`, `host/card_steps.rs`, `port_descriptors.rs`, `draw.rs`) | Cuts per-frame CPU overhead across refresh, barrier, and descriptor scans |
| **NVIDIA Wave32 & Shader Fixes** | `VICTORY_LDS_FLAGS=0`, `VICTORY_RAW_STRUCTURED=0`, `VICTORY_WAIT_RESIDENT=0`, `VICTORY_SHADER_LOOP_LIMITS`, BVH miss short-circuit (`intersect.rs`), `clear_air` NaN scrub (`ambient.rs`), R8 video plane fix (`texture.rs`) | Fixes black wall tiles on Wave32, eliminates 61.9 s residency stall, reduces heavy pass GPU load (-21.1 pts) |
| **Native Temporal Upscaler** | Custom TAAU/reconstruction pipeline + `VICTORY_RENDER_SCALE` + `VICTORY_OUTPUT` (`upscale.rs`) | Renders at configurable internal scale (`0.50` default -> `1920x1080`) for 8 GB VRAM cards |
| **8 GB VRAM & 16 GB RAM Caps** | Dynamic `spares()` VRAM cap, `SHARED_SPARE` 48 MiB pool, `WIDE_JOB=256 KiB` (`host.rs`, `pool.rs`), bounded `HostList` retention (`BURY_LAG=12`, `LIST_POOL_MAX=512`), `VICTORY_POOL_SMALL=1`, `VICTORY_D3D12_DRED=0` | Keeps memory flat (~5.3 GiB VRAM / ~16.2 GB RAM) without OOM paging stalls |
| **Windows 11 Laptop Responsiveness** | Dedicated DXGI GPU selection (`high_performance_adapter`), `VICTORY_PROCESS_PRIORITY=normal`, capped low-priority DirectStorage workers (`dstorage.rs`, `lifecycle.rs`), low-priority pipeline workers, occluded telemetry in `Meter::record_present` (`queue.rs`) | Eliminates OS/cursor freezes during loading & shader compilation on 16 GB laptops |

---

## How to Build & Run

Requirements: Windows 10/11 x64, CPU with AVX2, Direct3D 12 GPU with Shader Model 6.6 and Mesh Shaders, Rust toolchain (pinned in `rust-toolchain.toml`), MSVC build tools, and your own legally obtained console title files.

```powershell
git clone https://github.com/MoonLightDEV3/MoonLight-Scarlett
cd MoonLight-Scarlett
copy victory.example.toml victory.toml      # set paths outside the repo
cargo build --release --locked -p victory
```

Pre-compile shaders once before launching, then run with `--execute`:

```powershell
target\release\victory.exe shaders <path-to-shader-archives>
target\release\victory.exe run <path-to-title-executable> --console series-s --execute
```

On hybrid/Optimus laptops, `high_performance_adapter()` automatically prefers the discrete DXGI GPU (overridable via `VICTORY_D3D12_ADAPTER`).

---

## Credits & Upstream Attribution

This repository builds directly upon the upstream open-source ecosystem and keeps full git commit attribution:

- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)** — Core loader, HLE runtime services, D3D12 translation backend, and DXIL shader recompiler
- **[avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX)** — Thin list parallel engine (`VICTORY_THIN`), async submit/present, AVX2 landing, renderer ports 1–7, `VICTORY_OUTPUT`, and `VICTORY_AMBIENT_SETTLE`
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)** — Community fork base and pull requests
- **[trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA)** — NVIDIA Wave32 compatibility findings and VRAM/descriptor fixes
- **[deynesdev/Protonida](https://github.com/deynesdev/Protonida)** — `stage_spans` table-stage filtering (`VICTORY_TABLE_STAGES`)
- **[lepotane/VIctory-leonida](https://github.com/lepotane/VIctory-leonida)** — `SHARED_SPARE` upload pool and `WIDE_JOB` improvements
- **Additional upstream contributors:** `DexNom/VIctory`, `Belobug/VIctory-Belobug-AMD`, `FQY55kai/ZJOU-VICTORY`, `myokex/victory-amd`, `Geckomacabre`, `cxy0072019-sketch`, and `ivall/xcl` (architectural reference only; no GPL code included).
