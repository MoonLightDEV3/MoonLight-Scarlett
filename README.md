# MoonLight-Scarlett

An Xbox Series X|S (Scarlett) compatibility layer for Windows, forked from [avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX) / [Marvisak/VIctory](https://github.com/Marvisak/VIctory) and engineered for **8 GB VRAM NVIDIA GPUs & 16 GB RAM Windows laptops**.

> **Notice — Source Code Only:** This repository contains **no game files, executables, assets, icons, dumps, or keys**. Bring your own legally dumped console files (`victory.toml`). This project is not affiliated with, endorsed by, or associated with any game developer, publisher, or console manufacturer.

---

## Measured Performance & Validation (2026-11-26)

- **Test Hardware:** Windows 11 x64, Intel Core i9-13900H, 16 GB System RAM, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM, Wave32)
- **Profile:** `--console series-s`, ray tracing off (`diag` profile for telemetry; release builds enable Thin LTO + `codegen-units = 1`)

| Configuration | Result | Notes |
|---|---|---|
| Unmodified upstream baseline on this laptop | ~5.4 FPS, ~40 GB RAM pressure | 61.9 s residency stall, black opaque wall tiles on NVIDIA Wave32 |
| **30-Min Endurance (`1080p @ 0.50`, Cycle 9 `diag`, zero overrides)** | **15.1 FPS median** (mean `14.76`, p99 `16.0`, max `16.3`) | **0 crashes, 0 errors, 27,038 frames**; 5-min windows `15.20 / 15.00 / 15.20 / 15.10 / 14.90` FPS (1% low `13.8–14.5`) |
| Tuned `1080p` (`VICTORY_RENDER_SCALE=0.45`, Cycle 7E) | **22.7 FPS max** (`15.3` median) | Peak `1080p` output framerate |
| **Wave 11 (`1600x900 @ 0.50`, `diag`, `lto=false`)** | **14.7 FPS median / 18.0 max / 7.19 p1_low** | Intro **49.2 FPS**; crisp `900p` output, 2.8x higher 1% low |
| **Wave 11 Default (`1280x720 @ 0.60`, `diag`, `lto=false`)** | **15.8 FPS median / 19.8 max / 16.0–17.1 steady** | **41.0% mean GPU, 5,242 MiB VRAM**; CPU `submit` (~41–60 ms) is now the sole ceiling toward 30 FPS |
| **Wave 11 Native Release (`180 s` intro/menu check)** | **~58.8 FPS** (`4096 -> 8192` frames in `69.7 s`) | `hits=2304 misses=4`, `loaded=1981` pipelines, 0 crashes/errors |

### 30-Minute Steady-State Windows (`1080p @ 0.50` Endurance, `t = 180..1805 s`)

| Metric | Measured Range Across 5-Min Windows |
|---|---|
| Window Median FPS | `15.20 / 15.00 / 15.20 / 15.10 / 14.90 FPS` |
| Window 1% Low FPS | `14.50 / 13.80 / 13.90 / 14.30 / 13.90 FPS` |
| Median Frame Time | `65.6 – 66.9 ms` |
| Dedicated VRAM | `5,284 – 5,304 MiB` (flat, no leak) |
| Private System RAM | `16,144 – 16,282 MB` (flat, no spill) |
| Peak GPU Temperature | `82.0 °C` |

---

## NVIDIA, 8 GB VRAM & 16 GB RAM Optimizations

Every optimization below lives in Rust source code and is enabled safely in default runs (`apply_runtime_defaults` in `apps/victory/src/main.rs` and the GPU/HLE/shader crates):

| Area | Implementation | Effect |
|---|---|---|
| **Submission-Prelude Textures** | `PreludeUpload` & `HostList::prelude(&uploads)` (`texture.rs`, `host.rs`) | Refreshes & uploads modified textures on the command list's prelude right before draw execution, fixing texture flicker/popping & stale video-ring frames |
| **Mip & VRAM Texture Capping** | `from_base(desc, base, own_mips)` (`descriptor/host.rs`) + `over_cap()` / `VICTORY_TEXTURE_MAX` (`resource.rs`) + `room_lately()` gating | Prevents out-of-bounds mip views on arrays/streamed textures and caps peak texture mips dynamically (`1024`/`512`/`256`) at high VRAM pressure |
| **Wave32 & Ambient Lighting Fixes** | `VICTORY_LDS_FLAGS=0`, `VICTORY_RAW_STRUCTURED=0`, `VICTORY_WAIT_RESIDENT=0`, magenta/NaN probe scrub in `ambient::raise` + `clear_air` (`ambient.rs`), tightened `VICTORY_SHADER_LOOP_LIMITS`, `BVH_MISS=1`, R8 video plane fix (`texture.rs`) | Fixes black cinderblock wall tiles on Wave32, cleans stale magenta/NaN light probes, removes 61.9 s residency stall, cuts GPU shader cost |
| **Capped-Heap Dynamic Slots** | `dynamic_slots_capped` honoring `VICTORY_DYNAMIC_SLOTS=499999` (`descriptor/host.rs`) | Expands dynamic table slots from `262k` to `499,999` on NVIDIA's 1M descriptor heap |
| **Write-Watch & Sharded Journal** | Unwatched 4 KiB diffing (`shadow.rs`), `placed_cb_pool` (`list.rs`, `host/place.rs`), lock-free ranged `Shadow` seqlock, 64-shard `Notes` + 16 MiB `Shard::coarse` + coalesced `VirtualProtect` in `unguard` (`watch.rs`), `VICTORY_PORT_GUARD=1`, `VICTORY_PORT_JOURNAL=1` | Eliminates ~21 MiB/f echo copies & `BEHIND_PLACED` stalls while cutting guard/journal waits and kernel protect calls |
| **Submission & Upload Hot Paths** | Fused `Batch::refresh_front` (`CHUNK=128`), `SHARED_SPARE` 48 MiB pool, `WIDE_JOB=256 KiB` (`word_runs_put_avx2`), `VICTORY_TABLE_STAGES=1` (`draw.rs`), zero-alloc `card_steps::plan`, `u64` `port_descriptors::take`, `codegen-units=1` + Thin LTO (`Cargo.toml`) | Trims submission thread overhead across refreshes, barriers, descriptor scans, and uploads |
| **Native Temporal Upscaler** | Custom TAAU/reconstruction pipeline (`upscale.rs`), default `1280x720 @ 0.60` + `VICTORY_FRAME_INTERVAL_MS=16.667` | Smooth 720p TAAU output with large GPU headroom (`41%` GPU, `5.2 GiB` VRAM) and unlocked 60 FPS menus/intro |
| **8 GB VRAM & 16 GB RAM Laptops** | Dedicated DXGI GPU selection (`high_performance_adapter`), `VICTORY_PROCESS_PRIORITY=normal`, capped low-priority DirectStorage workers (`dstorage.rs`, `lifecycle.rs`), `VICTORY_PIPELINE_WORKERS=6` lowest priority, `BURY_LAG=12`, `LIST_POOL_MAX=512`, `VICTORY_POOL_SMALL=1`, `VICTORY_D3D12_DRED=0` | Eliminates Windows 11 OS/mouse freezes and keeps commit/VRAM flat |

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
- **[Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD)** — Submission-prelude texture uploads (`5d6bd84`), streamed mip clamping, dynamic VRAM `over_cap`, magenta ambient probe filtering, and release `codegen-units = 1` (`e942896`)
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)** — Community fork base and pull requests
- **[trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA)** — NVIDIA Wave32 compatibility findings and VRAM/descriptor fixes
- **[deynesdev/Protonida](https://github.com/deynesdev/Protonida)** — `stage_spans` table-stage filtering (`VICTORY_TABLE_STAGES`) and coalesced `VirtualProtect` (`fab7fda`)
- **[lepotane/VIctory-leonida](https://github.com/lepotane/VIctory-leonida)** — Capped-heap `VICTORY_DYNAMIC_SLOTS`, `SHARED_SPARE` upload pool, and `WIDE_JOB` improvements (`c6c9da9`)
- **Additional upstream contributors:** `DexNom/VIctory`, `FQY55kai/ZJOU-VICTORY`, `myokex/victory-amd`, `Geckomacabre`, `cxy0072019-sketch`, and `ivall/xcl` (architectural reference only; no GPL code included).
