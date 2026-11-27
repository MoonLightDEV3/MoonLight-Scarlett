// `victory`: the command-line front end.
//
// Every command reads the user's own files and writes only to stdout or to
// the user's cache directory (outside the repository).

mod census;
mod inspect;
#[cfg(all(windows, feature = "diagnostics"))]
mod replay;
mod run;
mod shader;
mod translate;
#[cfg(windows)]
mod warm;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[cfg(windows)]
#[unsafe(no_mangle)]
pub static NvOptimusEnablement: u32 = 1;

#[cfg(windows)]
#[unsafe(no_mangle)]
pub static AmdPowerXpressRequestHighPerformance: i32 = 1;

#[derive(Parser)]
#[command(
    name = "victory",
    version,
    about = "Run an Xbox Series title natively on Windows"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Summarise one PE image: headers, sections, imports, TLS, build id.
    Inspect {
        file: PathBuf,
        /// List every imported function.
        #[arg(long)]
        imports: bool,
        /// List every exported function.
        #[arg(long)]
        exports: bool,
    },
    /// Aggregate the imports of every PE image under the given paths and
    /// show which ones our runtime covers. Defaults to the configured
    /// game and test-title directories.
    Census {
        paths: Vec<PathBuf>,
        /// Do not count host pass-through as covered.
        #[arg(long)]
        strict: bool,
        /// List every missing `dll!function`, most used first.
        #[arg(long)]
        missing: bool,
    },
    /// List the `dll!export` pairs an import library describes.
    LibExports {
        lib: PathBuf,
        /// List every export, not just per-DLL counts.
        #[arg(long)]
        list: bool,
    },
    /// Find shader containers in files or directories (also inside engine
    /// archives, if uncompressed) and report their layout.
    ShaderInfo {
        paths: Vec<PathBuf>,
        /// Print one line per container found and show why paths were skipped.
        #[arg(short, long)]
        verbose: bool,
        /// Also list the decoded native instructions (implies --verbose).
        #[arg(long)]
        disasm: bool,
    },
    /// Run every shader found (containers, SGD5 files) through the
    /// translator and report what it cannot handle yet.
    ///
    /// VICTORY_HOST_WAVE overrides the host wave width, including with --warm.
    /// VICTORY_HOST_MAX_WAVE is a compatibility alias. When both values are
    /// valid, VICTORY_HOST_WAVE takes precedence. Zero means unknown width.
    /// For ordinary translation, VICTORY_HOST_RDNA=1 selects RDNA float settings.
    Translate {
        paths: Vec<PathBuf>,
        /// Check emitted DXIL with dxil.dll's validator (`VICTORY_DXIL`,
        /// else the newest Windows SDK's).
        #[arg(long)]
        validate: bool,
        /// Also create a host D3D12 pipeline for each emitted shader.
        #[arg(long)]
        pipelines: bool,
        /// Print one line per program and outcome.
        #[arg(long)]
        list: bool,
        /// Worker threads (default: every logical core).
        #[arg(long)]
        jobs: Option<usize>,
        /// Warm the run's shader cache: a file of shader hashes (the 16
        /// hex digits a run's log names a shader by, anywhere in a line);
        /// the containers with those hashes are translated into the
        /// cache and nothing else is done. Set `VICTORY_HOST_WAVE` and
        /// `VICTORY_VIEW_LIMIT` to what the run's log says of the host.
        #[arg(long)]
        warm: Option<PathBuf>,
    },
    /// Translate every shader found (as `translate` finds them) into the
    /// shader cache, so that no run has to: a few minutes, once per version
    /// of the translator. Translations an earlier build of the same version
    /// cached are made again, and those that came out different counted.
    /// In diagnostics builds, VICTORY_LOG=debug shows skipped paths.
    Shaders {
        paths: Vec<PathBuf>,
        /// Worker threads (default: every logical core).
        #[arg(long)]
        jobs: Option<usize>,
        /// Also write every shader that does not translate to this file,
        /// one line each: hash, where it was found, why.
        #[arg(long)]
        list: Option<PathBuf>,
    },
    /// Make the host pipelines of every pipeline the title ever created
    /// (runs with `VICTORY_PIPELINE_BOOK=1` write them down), so that the
    /// driver's cache on disk has them before a run needs one: once per change
    /// of the translator or the driver. Goes on where it was stopped.
    #[cfg(windows)]
    Warm {
        /// Processes at once (default: two thirds of the logical cores).
        #[arg(long)]
        jobs: Option<usize>,
        /// At most this many pipelines.
        #[arg(long)]
        limit: Option<usize>,
        /// Those made before too.
        #[arg(long)]
        again: bool,
        /// Only the pipelines earlier runs drew with.
        #[arg(long)]
        used: bool,
        /// (What the processes it starts are given.)
        #[arg(long, hide = true)]
        these: Option<String>,
    },
    /// Prepare, map and bind a title executable; with --execute, run it.
    Run {
        exe: PathBuf,
        /// What unimplemented imports do when called.
        #[arg(long, value_enum, default_value_t = Unimplemented::Trap)]
        unimplemented: Unimplemented,
        /// Disable host pass-through (everything must be implemented by us).
        #[arg(long)]
        strict: bool,
        /// Actually start the title. Without it, stop after binding.
        #[arg(long)]
        execute: bool,
        /// The console model the title sees (device type, memory).
        #[arg(long, value_enum, default_value_t = Console::SeriesX)]
        console: Console,
        /// Arguments for the title, after `--` (it reads the whole command
        /// line; a title that relaunches itself passes some).
        #[arg(last = true)]
        title_args: Vec<String>,
    },
    /// Run a captured frame's GPU work again (diagnostics builds; see
    /// docs/replay.md).
    #[cfg(all(windows, feature = "diagnostics"))]
    Replay(replay::Replay),
    /// Show the resolved configuration and check it.
    Config,
}

#[derive(Clone, Copy, ValueEnum)]
enum Unimplemented {
    Trap,
    Zero,
}

#[derive(Clone, Copy, ValueEnum)]
enum Console {
    SeriesX,
    SeriesS,
}

fn main() -> Result<()> {
    victory_core::logging::init("info");
    // The audio recorder `run` starts for `VICTORY_AUDIO_CAPTURE`.
    #[cfg(all(windows, feature = "diagnostics"))]
    if victory_hle::modules::audio::capture::child() {
        return Ok(());
    }
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect {
            file,
            imports,
            exports,
        } => inspect::run(&file, imports, exports),
        Command::Census {
            paths,
            strict,
            missing,
        } => census::run(paths, strict, missing),
        Command::LibExports { lib, list } => inspect::lib_exports(&lib, list),
        Command::ShaderInfo {
            paths,
            verbose,
            disasm,
        } => shader::run(paths, verbose || disasm, disasm),
        Command::Translate {
            paths,
            validate,
            pipelines,
            list,
            jobs,
            warm,
        } => {
            // `VICTORY_VIEW_LIMIT=<null slot>`: translate as on a host whose view heap is capped
            if let Some(n) = std::env::var("VICTORY_VIEW_LIMIT").ok().and_then(|v| v.parse().ok()) {
                victory_shader::emit::set_view_heap_limit(n);
            }
            let wave = translate::host_wave::select(
                std::env::var("VICTORY_HOST_WAVE").ok().as_deref(),
                std::env::var("VICTORY_HOST_MAX_WAVE").ok().as_deref(),
                |message| eprintln!("{message}"),
            );
            if let Some(file) = warm {
                return translate::warm(paths, &file, jobs, wave);
            }
            translate::run(paths, validate, pipelines, list, jobs, wave)
        }
        Command::Shaders { paths, jobs, list } => translate::precompile(paths, jobs, list),
        #[cfg(windows)]
        Command::Warm { jobs, limit, again, used, these } => warm::run(jobs, limit, again, used, these),
        Command::Run {
            exe,
            unimplemented,
            strict,
            execute,
            console,
            title_args,
        } => {
            if execute {
                apply_runtime_defaults();
            }
            victory_hle::runtime::set_title_args(title_args);
            run::run(&exe, unimplemented, strict, execute, console)
        }
        #[cfg(all(windows, feature = "diagnostics"))]
        Command::Replay(options) => replay::run(options),
        Command::Config => show_config(),
    }
}

/// Applies the hybrid VIctory-7900XTX CPU/GI + RTX 5060 8 GB VRAM / 16 GB RAM
/// laptop-safe defaults for any environment variable the user has not explicitly overridden.
fn apply_runtime_defaults() {
    let set_default = |key: &str, val: &str| {
        if std::env::var_os(key).is_none() {
            // SAFETY: called during single-threaded CLI startup before worker threads spawn.
            unsafe { std::env::set_var(key, val) };
        }
    };
    // CPU Engine (7900XTX): thin list recording + parallel translation, async submit/present,
    // AVX2 landing, GPU card steps (without saturating laptop compute queue or DWM).
    set_default("VICTORY_THIN", "1");
    set_default("VICTORY_THIN_THREADS", "6");
    set_default("VICTORY_FAST_CORES", "2");
    set_default("VICTORY_LANE_CORE", "1");
    set_default("VICTORY_UP_QUEUE", "0");
    set_default("VICTORY_WAIT_RESIDENT", "0");
    set_default("VICTORY_JOIN_PARTS", "1");
    set_default("VICTORY_RELAY", "2");
    set_default("VICTORY_WIDE_BACK", "1");
    set_default("VICTORY_WIDE_SPIN", "80");
    set_default("VICTORY_LAND_SKIP", "1");
    set_default("VICTORY_LAND_THREADS", "6");
    set_default("VICTORY_RELAY_COPY", "1");
    set_default("VICTORY_CARD_STEPS", "2");
    set_default("VICTORY_PLACED_ORDER", "1");
    set_default("VICTORY_JOIN_BARRIERS", "1");
    set_default("VICTORY_CARD_REHASH_RUNS", "1");
    set_default("VICTORY_JOIN_COPIES", "1");
    set_default("VICTORY_UNDER_FUSED", "0");
    set_default("VICTORY_INDIRECT_WINDOW", "1");
    set_default("VICTORY_INDIRECT_BATCH", "2");
    set_default("VICTORY_MAP_THREADS", "1");
    // Renderer port #2 (hooks 6 descriptor table stores in the title executable;
    // cuts heap-sync wait 9.8 -> 4.7 ms) + unguarded memcpy/store bypass (cuts RO blocks & RAM).
    set_default("VICTORY_PORT_DESCRIPTORS", "1");
    set_default("VICTORY_SYNC_FINE", "1");
    set_default("VICTORY_HEAP_SYNC_EARLY", "2");
    set_default("VICTORY_LATE_WAIT_LISTS", "128");
    set_default("VICTORY_PORT_GUARD", "1");
    set_default("VICTORY_PORT_JOURNAL", "1");
    set_default("VICTORY_COPY_BESIDE", "1");
    set_default("VICTORY_STORE_BESIDE", "1");
    // Anti-freeze OS/thread responsiveness on 16 GB Windows 11 laptops:
    // keep process at Normal priority, boost title's idle streaming pager (1),
    // and cap DirectStorage decompression to 6 low-priority threads.
    set_default("VICTORY_PROCESS_PRIORITY", "normal");
    set_default("VICTORY_TITLE_PRIORITIES", "1");
    set_default("VICTORY_DSTORAGE_WORKERS", "6");
    set_default("VICTORY_DSTORAGE_LOW_PRIORITY", "1");
    set_default("VICTORY_CARD_AWAKE", "0");
    set_default("VICTORY_UPLOAD_AHEAD", "1");
    set_default("VICTORY_TEXTURE_NO_WAIT", "1");
    // Wave 13: async GPU flight queue (2 frames), async texback writer, early table reclaim,
    // NaN scrub + SH ambient stability, single-wait pipelines, port constants order, fair texture queue.
    set_default("VICTORY_SUBMIT_FRAMES", "2");
    set_default("VICTORY_TEXBACK_ASYNC", "1");
    set_default("VICTORY_EARLY_TABLES", "1");
    set_default("VICTORY_NAN_SCRUB", "1");
    set_default("VICTORY_PIPELINE_WAIT_ONCE", "1");
    set_default("VICTORY_PORT_CONSTANTS_ORDER", "1");
    set_default("VICTORY_FAIR_DUE", "1");
    // Visuals, lighting & non-black textures/skin: directional SH GI + keep-shown fallback.
    set_default("VICTORY_AMBIENT", "1.1");
    set_default("VICTORY_AMBIENT_AIR", "1.5");
    set_default("VICTORY_AMBIENT_FLOOR", "0.22");
    set_default("VICTORY_AMBIENT_DARK", "0.55");
    set_default("VICTORY_AMBIENT_MOST", "2");
    set_default("VICTORY_KEEP_SHOWN", "1");
    set_default("VICTORY_SHOW_AFTER", "1");
    set_default("VICTORY_SKIP_PIPELINES", "none");
    set_default("VICTORY_RAYMARCHED_LOCAL_LIGHTS", "0");
    set_default("VICTORY_OWN_UPSCALER", "1");
    set_default("VICTORY_BVH_MISS", "1");
    set_default("VICTORY_SHADER_LOOP_LIMITS", "39f0e9aeb156a61c:32,8e614708143ef917:64,34fb51b8a0312e82:64");
    // Disable Wave32 LDS flag-region pass (PR #4) on NVIDIA: compiling 2cc4c197ee29e291
    // without full Wave64 LDS sync blacks out opaque lighting tiles on cinderblock walls.
    set_default("VICTORY_LDS_FLAGS", "0");
    // 1080p@0.50 internal scale (960x540 -> 1080p TAAU) + unlocked frame interval for RTX 5060 8 GB laptop.
    set_default("VICTORY_OUTPUT", "1920x1080");
    set_default("VICTORY_RENDER_SCALE", "0.50");
    set_default("VICTORY_FRAME_INTERVAL_MS", "16.667");
    // Keep <= 262144 on NVIDIA 1M-capped view heaps to preserve bindless slots 500k..737k;
    // VICTORY_EARLY_TABLES=1 keeps 0 table allocation failures.
    set_default("VICTORY_DYNAMIC_SLOTS", "262144");
    set_default("VICTORY_PIPELINE_BOOK", "1");
    set_default("VICTORY_PIPELINE_WORKERS", "6");
    set_default("VICTORY_PIPELINE_LOWEST", "1");
    // 16 GB RAM / 8 GB VRAM Laptop Memory Caps (prevents shared PC RAM spill & OOM freeze,
    // bounds HostList retention via shorter bury lag + matched pool cap, disables DRED book bloat).
    set_default("VICTORY_KEEP_FREED", "0");
    set_default("VICTORY_RIPE_BLOCKS", "0");
    set_default("VICTORY_BURY_LAG", "12");
    set_default("VICTORY_LIST_POOL_MAX", "512");
    set_default("VICTORY_WINDOWS_BAR_MB", "1024");
    set_default("VICTORY_WINDOW_KB", "128");
    set_default("VICTORY_WINDOW_TAILS", "1");
    set_default("VICTORY_CARD_BUFFERS_MIN_KB", "64");
    set_default("VICTORY_CARD_BUFFERS_MB", "1280");
    set_default("VICTORY_CARD_ROOM_FOR_TARGETS_MB", "1024");
    set_default("VICTORY_TEXTURE_BUDGET_MB", "1024");
    // Keep 4 KiB sub-64 KiB texture packing to avoid 3.3x VRAM bloat on 8 GB GPUs.
    set_default("VICTORY_POOL_SMALL", "1");
    set_default("VICTORY_TEXTURE_ROOM_PERCENT", "88");
    set_default("VICTORY_TEXTURE_STAGE_MB", "32");
    set_default("VICTORY_TEXTURE_KEEP", "0");
    set_default("VICTORY_INPLACE", "0");
    set_default("VICTORY_SHADER_LOOP_LIMIT", "4096");
    set_default("VICTORY_DISPLAY_MODE", "1920x1080");
    set_default("VICTORY_PLANE_OUT", "base");
    set_default("VICTORY_D3D12_DRED", "0");
}

fn show_config() -> Result<()> {
    let cfg = victory_core::Config::load()?;
    let status = |p: &Option<PathBuf>| match p {
        Some(p) if p.exists() => format!("{}", p.display()),
        Some(p) => format!("{}  (does not exist)", p.display()),
        None => "(not set)".to_owned(),
    };
    println!(
        "config file : {}",
        cfg.source
            .as_ref()
            .map_or("(none)".into(), |p| p.display().to_string())
    );
    println!("game_dir    : {}", status(&cfg.game_dir));
    println!("cache_dir   : {}", cfg.cache_dir.display());
    println!("data_dir    : {}", cfg.data_dir.display());
    for t in &cfg.test_titles {
        println!("test_title  : {}", status(&Some(t.clone())));
    }
    for p in cfg
        .game_dir
        .iter()
        .chain(&cfg.test_titles)
        .chain([&cfg.cache_dir, &cfg.data_dir])
    {
        if let Err(e) = victory_core::ensure_outside_repo(p) {
            println!("POLICY VIOLATION: {e}");
        }
    }
    Ok(())
}

/// Paths given on the command line, or the configured game + test titles.
pub(crate) fn default_paths(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if !paths.is_empty() {
        return Ok(paths);
    }
    let cfg = victory_core::Config::load()?;
    let all: Vec<PathBuf> = cfg.game_dir.into_iter().chain(cfg.test_titles).collect();
    anyhow::ensure!(
        !all.is_empty(),
        "no paths given and none configured (see victory.example.toml)"
    );
    Ok(all)
}
