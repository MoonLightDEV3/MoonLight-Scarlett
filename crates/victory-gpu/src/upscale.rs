#![allow(dead_code)]

//! Built-in HDR-safe compute upscaler used as a fallback when an AMD ACO
//! (`ACCC`-only, no DXIL) upscaler pass (such as FidelityFX FSR4 Xbox) cannot
//! be translated.
//!
//! Single-pass 12-tap Edge-Adaptive Spatial Upscaling (directional Lanczos2 /
//! EASU) followed by HDR tonemap-aware adaptive RCAS sharpening reconstructed
//! from the same 12-tap stencil, clamped strictly to the local 2x2 source
//! neighborhood so negative lobes never ring or halo.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::Fxc::{D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCompile};
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, PCSTR, s};

use crate::host::{HostList, host};

/// Single-pass EASU (directional Lanczos2) + HDR-safe tonemap-aware RCAS
/// compute shader. Preserves the exact `t0` (SRV) / `u0` (UAV) / `s0` (static
/// linear clamp sampler) bindings of the fallback upscaler root signature.
pub(crate) const UPSCALE_HLSL: &str = r#"
Texture2D<float4> srcTex : register(t0);
RWTexture2D<float4> dstTex : register(u0);
SamplerState linSamp : register(s0);

// Max finite value representable in both R11G11B10_FLOAT (65024) and R16G16B16A16_FLOAT (65504).
static const float HDR_MAX = 65024.0;
// Adaptive RCAS sharpening strength (0.0 = off, 1.0 = max safe anti-ringing lobe).
static const float RCAS_SHARPNESS = 0.75;

float sanitize_f(float v, float fallback) {
    return isfinite(v) ? clamp(v, 0.0, HDR_MAX) : fallback;
}

float4 load_safe(int2 coord, int2 srcMax) {
    float4 c = srcTex.Load(int3(clamp(coord, int2(0, 0), srcMax), 0));
    return float4(
        sanitize_f(c.r, 0.0),
        sanitize_f(c.g, 0.0),
        sanitize_f(c.b, 0.0),
        sanitize_f(c.a, 1.0)
    );
}

float3 load_safe_rgb(int2 coord, int2 srcMax) {
    float3 c = srcTex.Load(int3(clamp(coord, int2(0, 0), srcMax), 0)).rgb;
    return float3(
        sanitize_f(c.r, 0.0),
        sanitize_f(c.g, 0.0),
        sanitize_f(c.b, 0.0)
    );
}

// Tonemap-aware perceptual Rec.709 luma: maps linear HDR [0, +inf) -> [0, 1) smoothly,
// keeping darks sensitive and preventing specular HDR peaks from skewing edge detection or RCAS.
float hdr_luma(float3 rgb) {
    float l = dot(rgb, float3(0.2126, 0.7152, 0.0722));
    return sqrt(l * rcp(1.0 + l));
}

// Accumulates edge gradient direction and edge coherence (anisotropy) for one 5-tap cross.
void easu_set(
    inout float2 dir,
    inout float len,
    float w,
    float lTop,
    float lLeft,
    float lCenter,
    float lRight,
    float lBot
) {
    float dc = lRight - lCenter;
    float cb = lCenter - lLeft;
    float lenX = max(abs(dc), abs(cb));
    float dirX = lRight - lLeft;
    dir.x += dirX * w;
    float normX = (lenX > 1e-5) ? saturate(abs(dirX) / lenX) : 0.0;
    len += (normX * normX) * w;

    float ec = lBot - lCenter;
    float ca = lCenter - lTop;
    float lenY = max(abs(ec), abs(ca));
    float dirY = lBot - lTop;
    dir.y += dirY * w;
    float normY = (lenY > 1e-5) ? saturate(abs(dirY) / lenY) : 0.0;
    len += (normY * normY) * w;
}

// Evaluates one tap of the edge-aligned elliptical Lanczos2 approximation kernel.
void easu_tap(
    inout float3 aC,
    inout float aW,
    float2 off,
    float2 dirX,
    float2 dirY,
    float lob,
    float clp,
    float3 c
) {
    // Project tap offset into pre-scaled edge-aligned frame (aX across edge, aY along edge).
    float aX = dot(off, dirX);
    float aY = dot(off, dirY);
    float d2 = min(aX * aX + aY * aY, clp);

    // Fast Lanczos2 approximation: exact 1.0 at d2=0, exact 0.0 at d2=1 and d2=clp,
    // smooth negative outer lobe for sharp reconstruction.
    float wB = (2.0 / 5.0) * d2 - 1.0;
    float wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = (25.0 / 16.0) * wB - (25.0 / 16.0 - 1.0);
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint srcW, srcH, dstW, dstH;
    srcTex.GetDimensions(srcW, srcH);
    dstTex.GetDimensions(dstW, dstH);
    if (tid.x >= dstW || tid.y >= dstH || srcW == 0u || srcH == 0u) {
        return;
    }

    float2 srcSize = float2(srcW, srcH);
    float2 dstSize = float2(dstW, dstH);
    float2 invScale = srcSize / dstSize;
    float2 pp = (float2(tid.xy) + 0.5) * invScale - 0.5;
    float2 fp = floor(pp);
    float2 f = pp - fp;
    int2 base = int2(fp);
    int2 srcMax = int2(int(srcW) - 1, int(srcH) - 1);

    // 12-tap cross stencil around base:
    //       b  c
    //    e  f  g  h
    //    i  j  k  l
    //       n  o
    float3 tb = load_safe_rgb(base + int2( 0, -1), srcMax);
    float3 tc = load_safe_rgb(base + int2( 1, -1), srcMax);
    float3 te = load_safe_rgb(base + int2(-1,  0), srcMax);
    float4 tf = load_safe(base + int2( 0,  0), srcMax);
    float4 tg = load_safe(base + int2( 1,  0), srcMax);
    float3 th = load_safe_rgb(base + int2( 2,  0), srcMax);
    float3 ti = load_safe_rgb(base + int2(-1,  1), srcMax);
    float4 tj = load_safe(base + int2( 0,  1), srcMax);
    float4 tk = load_safe(base + int2( 1,  1), srcMax);
    float3 tl = load_safe_rgb(base + int2( 2,  1), srcMax);
    float3 tn = load_safe_rgb(base + int2( 0,  2), srcMax);
    float3 to = load_safe_rgb(base + int2( 1,  2), srcMax);

    // Perceptual HDR luma for edge direction & anisotropy estimation.
    float lB = hdr_luma(tb);
    float lC = hdr_luma(tc);
    float lE = hdr_luma(te);
    float lF = hdr_luma(tf.rgb);
    float lG = hdr_luma(tg.rgb);
    float lH = hdr_luma(th);
    float lI = hdr_luma(ti);
    float lJ = hdr_luma(tj.rgb);
    float lK = hdr_luma(tk.rgb);
    float lL = hdr_luma(tl);
    float lN = hdr_luma(tn);
    float lO = hdr_luma(to);

    // Stage 1: Edge-adaptive directional Lanczos2 (EASU).
    float2 dir = float2(0.0, 0.0);
    float len = 0.0;
    easu_set(dir, len, (1.0 - f.x) * (1.0 - f.y), lB, lE, lF, lG, lJ);
    easu_set(dir, len,        f.x  * (1.0 - f.y), lC, lF, lG, lH, lK);
    easu_set(dir, len, (1.0 - f.x) *        f.y,  lF, lI, lJ, lK, lN);
    easu_set(dir, len,        f.x  *        f.y,  lG, lJ, lK, lL, lO);

    float dirR = dot(dir, dir);
    bool zro = dirR < (1.0 / 32768.0);
    dir = zro ? float2(1.0, 0.0) : (dir * rsqrt(dirR));

    len = len * 0.5;
    len *= len;
    float stretch = dot(dir, dir) * rcp(max(abs(dir.x), abs(dir.y)));
    float2 len2 = float2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    float2 dirX = dir * len2.x;
    float2 dirY = float2(-dir.y, dir.x) * len2.y;
    float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp = rcp(lob);

    float3 aC = float3(0.0, 0.0, 0.0);
    float aW = 0.0;
    easu_tap(aC, aW, float2( 0.0, -1.0) - f, dirX, dirY, lob, clp, tb);
    easu_tap(aC, aW, float2( 1.0, -1.0) - f, dirX, dirY, lob, clp, tc);
    easu_tap(aC, aW, float2(-1.0,  0.0) - f, dirX, dirY, lob, clp, te);
    easu_tap(aC, aW, float2( 0.0,  0.0) - f, dirX, dirY, lob, clp, tf.rgb);
    easu_tap(aC, aW, float2( 1.0,  0.0) - f, dirX, dirY, lob, clp, tg.rgb);
    easu_tap(aC, aW, float2( 2.0,  0.0) - f, dirX, dirY, lob, clp, th);
    easu_tap(aC, aW, float2(-1.0,  1.0) - f, dirX, dirY, lob, clp, ti);
    easu_tap(aC, aW, float2( 0.0,  1.0) - f, dirX, dirY, lob, clp, tj.rgb);
    easu_tap(aC, aW, float2( 1.0,  1.0) - f, dirX, dirY, lob, clp, tk.rgb);
    easu_tap(aC, aW, float2( 2.0,  1.0) - f, dirX, dirY, lob, clp, tl);
    easu_tap(aC, aW, float2( 0.0,  2.0) - f, dirX, dirY, lob, clp, tn);
    easu_tap(aC, aW, float2( 1.0,  2.0) - f, dirX, dirY, lob, clp, to);

    // Local 2x2 anti-ringing bounds (prevents negative-lobe Lanczos2 haloing).
    float3 min4 = min(min(tf.rgb, tg.rgb), min(tj.rgb, tk.rgb));
    float3 max4 = max(max(tf.rgb, tg.rgb), max(tj.rgb, tk.rgb));
    float3 easuRgb = clamp(aC * rcp(max(aW, 1e-5)), min4, max4);

    // Stage 2: HDR tonemap-aware adaptive RCAS sharpening from the same 12 taps.
    // Reconstruct the 4 cardinal destination-pixel neighbors at +-min(invScale, 1.0).
    float2 step = min(invScale, float2(1.0, 1.0));

    float3 rowTop   = lerp(tb, tc, f.x);
    float3 row0     = lerp(tf.rgb, tg.rgb, f.x);
    float3 row1     = lerp(tj.rgb, tk.rgb, f.x);
    float3 rowBot   = lerp(tn, to, f.x);
    float3 colLeft  = lerp(te, ti, f.y);
    float3 col0     = lerp(tf.rgb, tj.rgb, f.y);
    float3 col1     = lerp(tg.rgb, tk.rgb, f.y);
    float3 colRight = lerp(th, tl, f.y);

    float yN = f.y - step.y;
    float3 nbN = (yN < 0.0) ? lerp(rowTop, row0, yN + 1.0) : lerp(row0, row1, yN);
    float yS = f.y + step.y;
    float3 nbS = (yS < 1.0) ? lerp(row0, row1, yS) : lerp(row1, rowBot, yS - 1.0);
    float xW = f.x - step.x;
    float3 nbW = (xW < 0.0) ? lerp(colLeft, col0, xW + 1.0) : lerp(col0, col1, xW);
    float xE = f.x + step.x;
    float3 nbE = (xE < 1.0) ? lerp(col0, col1, xE) : lerp(col1, colRight, xE - 1.0);

    float pC = hdr_luma(easuRgb);
    float pN = hdr_luma(nbN);
    float pS = hdr_luma(nbS);
    float pW = hdr_luma(nbW);
    float pE = hdr_luma(nbE);

    float mnL = min(pC, min(min(pN, pS), min(pW, pE)));
    float mxL = max(pC, max(max(pN, pS), max(pW, pE)));

    float hitMin = mnL / max(4.0 * mxL, 1e-5);
    float hitMax = (1.0 - mxL) / max(4.0 * (1.0 - mnL), 1e-5);
    float lobe = max(-0.1875, -min(hitMin, hitMax));
    // Roll off RCAS across <= 1-LSB smooth gradients so 8-bit / R11G11B10 quantization
    // steps never get sharpened into sky/wall Mach bands, while keeping full sharpness
    // on real edges and texture detail.
    float gradGate = smoothstep(0.0015, 0.012, mxL - mnL);
    float wRcas = lobe * (RCAS_SHARPNESS * gradGate);

    float pDelta = (wRcas * rcp(1.0 + 4.0 * wRcas)) * ((pN + pS + pW + pE) - 4.0 * pC);
    float hdrFade = saturate((0.99 - pC) * 20.0);
    float dp = pDelta * hdrFade;
    float pSharp = clamp(pC + dp, 0.0, 0.995);

    float gain = 1.0;
    if (pC > 1e-4 && abs(dp) > 1e-6) {
        float r = pSharp / pC;
        float numer = max(1.0 - pC * pC, 1e-4);
        float denom = max(1.0 - pSharp * pSharp, 1e-4);
        gain = clamp((r * r) * (numer / denom), 0.5, 2.0);
    }

    // Hue-preserving luma sharpening clamped strictly to local [min4, max4] (zero haloing).
    float3 outRgb = clamp(easuRgb * gain, min4, max4);
    float outA = lerp(lerp(tf.a, tg.a, f.x), lerp(tj.a, tk.a, f.x), f.y);

    dstTex[tid.xy] = float4(outRgb, outA);
}
"#;

const UPSCALE_PAIRS: u32 = 128;

struct Upscaler {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
}

fn compile_fxc_stage(
    source: &str,
    name: PCSTR,
    entry: PCSTR,
    target: PCSTR,
) -> Result<Vec<u8>, (windows::core::Error, String)> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: `source` and static C strings live through the call; out pointers are ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            entry,
            target,
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        let mut msg = String::new();
        if let Some(errors) = errors {
            // SAFETY: `errors` is a valid ID3DBlob owned by us.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            msg = String::from_utf8_lossy(text).trim().to_string();
        }
        return Err((e, msg));
    }
    let code = code.ok_or_else(|| (windows::core::Error::empty(), String::new()))?;
    // SAFETY: `code` is a valid ID3DBlob owned by us.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
            .to_vec()
    };
    Ok(bytes)
}

fn compile_fxc(source: &str) -> Result<Vec<u8>, (windows::core::Error, String)> {
    compile_fxc_stage(source, s!("upscale_cs"), s!("main"), s!("cs_5_1"))
}

impl Upscaler {
    fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        let bytecode = compile_fxc(UPSCALE_HLSL).map_err(|(e, msg)| {
            if !msg.is_empty() {
                tracing::warn!("built-in EASU+RCAS upscaler: D3DCompile: {msg}");
            }
            e
        })?;
        let ranges = [
            D3D12_DESCRIPTOR_RANGE1 {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 1,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                OffsetInDescriptorsFromTableStart: 0,
            },
            D3D12_DESCRIPTOR_RANGE1 {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                NumDescriptors: 1,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                OffsetInDescriptorsFromTableStart: 1,
            },
        ];
        let param = D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                    NumDescriptorRanges: ranges.len() as u32,
                    pDescriptorRanges: ranges.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            BorderColor: D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            MinLOD: 0.0,
            MaxLOD: f32::MAX,
            ShaderRegister: 0,
            RegisterSpace: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            ..Default::default()
        };
        let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 1,
                    pParameters: &raw const param,
                    NumStaticSamplers: 1,
                    pStaticSamplers: &raw const sampler,
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
                },
            },
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut err: Option<ID3DBlob> = None;
        // SAFETY: valid root signature description and compiled compute shader bytecode.
        unsafe {
            D3D12SerializeVersionedRootSignature(&rs_desc, &mut blob, Some(&mut err))?;
            let blob = blob.ok_or_else(windows::core::Error::empty)?;
            let rs_bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, rs_bytes)?;
            let pso: ID3D12PipelineState =
                device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytecode.as_ptr().cast(),
                        BytecodeLength: bytecode.len(),
                    },
                    ..Default::default()
                })?;
            let heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: UPSCALE_PAIRS * 2,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let step =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            Ok(Upscaler {
                root,
                pso,
                heap,
                step,
                next: AtomicU32::new(0),
            })
        }
    }
}

fn upscaler() -> Option<&'static Upscaler> {
    static UP: OnceLock<Option<Upscaler>> = OnceLock::new();
    UP.get_or_init(|| match Upscaler::new(&host()?.device) {
        Ok(u) => Some(u),
        Err(e) => {
            tracing::warn!(error = %e, "built-in EASU+RCAS upscaler not created");
            None
        }
    })
    .as_ref()
}

pub(crate) fn concrete_hdr_format(fmt: DXGI_FORMAT) -> DXGI_FORMAT {
    match fmt.0 {
        9 => DXGI_FORMAT(10),       // R16G16B16A16_TYPELESS -> FLOAT
        23 => DXGI_FORMAT(24),      // R10G10B10A2_TYPELESS -> UNORM
        27 | 29 => DXGI_FORMAT(28), // R8G8B8A8_TYPELESS / SRGB -> UNORM
        90 | 91 => DXGI_FORMAT(87), // B8G8R8A8_TYPELESS / SRGB -> UNORM
        _ => fmt,
    }
}

/// Upscales 2D texture `src` into 2D UAV texture `dst` on `list` using
/// single-pass EASU (directional Lanczos2) + HDR-safe adaptive RCAS.
///
/// Callers must ensure `src` is in shader-resource state and `dst` is in
/// unordered-access state, and must invalidate cached host heaps/PSO/root.
pub(crate) fn upscale_texture(
    list: &HostList,
    src: &ID3D12Resource,
    dst: &ID3D12Resource,
) -> bool {
    let (Some(h), Some(u)) = (host(), upscaler()) else {
        return false;
    };
    // SAFETY: live D3D12 resources and open command list.
    unsafe {
        let sdesc = src.GetDesc();
        let ddesc = dst.GetDesc();
        if sdesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || ddesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || sdesc.Width == 0
            || sdesc.Height == 0
            || ddesc.Width == 0
            || ddesc.Height == 0
            || ddesc.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0
        {
            return false;
        }
        let pair = u.next.fetch_add(1, Ordering::Relaxed) % UPSCALE_PAIRS;
        let base_slot = pair * 2;
        let cpu_start = u.heap.GetCPUDescriptorHandleForHeapStart().ptr;
        let gpu_start = u.heap.GetGPUDescriptorHandleForHeapStart().ptr;
        let srv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu_start + (base_slot * u.step) as usize,
        };
        let uav_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu_start + ((base_slot + 1) * u.step) as usize,
        };
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: concrete_hdr_format(sdesc.Format),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: concrete_hdr_format(ddesc.Format),
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        h.device
            .CreateShaderResourceView(src, Some(&srv_desc), srv_cpu);
        h.device
            .CreateUnorderedAccessView(dst, None, Some(&uav_desc), uav_cpu);
        list.keep(src);
        list.keep(dst);
        list.keep(&u.heap);
        let l = &list.list;
        l.SetDescriptorHeaps(&[Some(u.heap.clone())]);
        l.SetComputeRootSignature(&u.root);
        l.SetPipelineState(&u.pso);
        l.SetComputeRootDescriptorTable(
            0,
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_start + u64::from(base_slot) * u64::from(u.step),
            },
        );
        let gx = (ddesc.Width as u32).div_ceil(8);
        let gy = ddesc.Height.div_ceil(8);
        l.Dispatch(gx, gy, 1);
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    // Borrowed pointer; `ManuallyDrop` never releases it (no refcount leak).
                    pResource: std::mem::transmute_copy(dst),
                }),
            },
        };
        l.ResourceBarrier(&[barrier]);
    }
    true
}

/// Maximum number of `PresentX` planes composited per frame.
pub(crate) const MAX_COMPOSE_PLANES: usize = 8;
const COMPOSE_RTV_SLOTS: u32 = 32;
const COMPOSE_SRV_SETS: u32 = 64;

/// Fullscreen-triangle multi-plane `PresentX` compositor HLSL (`vs`, `ps_opaque`,
/// `ps_overlay`). Plane 0 uses 1-tap `Load` when 1:1, 12-tap EASU + smooth-gradient-safe
/// HDR RCAS when upscaling (e.g. Series S `1920x1080` scene -> `3840x2160` display),
/// and bilinear filtering when downscaling. Planes `1..N` (e.g. Series S `3840x2160`
/// UI/menu overlay) composite via hardware premultiplied-alpha ROP blending directly
/// into the swap-chain back buffer (`0 MB` extra intermediate VRAM).
pub(crate) const PRESENT_COMPOSITE_HLSL: &str = r#"
Texture2D<float4> srcTex : register(t0);
SamplerState linSamp : register(s0);

cbuffer ComposeParams : register(b0) {
    uint dstW;
    uint dstH;
    // bit 0: straight alpha (0 = Xbox default premultiplied alpha)
    // bit 1: SDR sRGB overlay onto linear scRGB (R16G16B16A16_FLOAT) target
    uint flags;
    float paperWhite;
};

static const float HDR_MAX = 65024.0;
static const float RCAS_SHARPNESS = 0.75;

float sanitize_f(float v, float fallback) {
    return isfinite(v) ? clamp(v, 0.0, HDR_MAX) : fallback;
}

float4 load_safe(int2 coord, int2 srcMax) {
    float4 c = srcTex.Load(int3(clamp(coord, int2(0, 0), srcMax), 0));
    return float4(
        sanitize_f(c.r, 0.0),
        sanitize_f(c.g, 0.0),
        sanitize_f(c.b, 0.0),
        sanitize_f(c.a, 1.0)
    );
}

float3 load_safe_rgb(int2 coord, int2 srcMax) {
    float3 c = srcTex.Load(int3(clamp(coord, int2(0, 0), srcMax), 0)).rgb;
    return float3(
        sanitize_f(c.r, 0.0),
        sanitize_f(c.g, 0.0),
        sanitize_f(c.b, 0.0)
    );
}

float hdr_luma(float3 rgb) {
    float l = dot(rgb, float3(0.2126, 0.7152, 0.0722));
    return sqrt(l * rcp(1.0 + l));
}

void easu_set(
    inout float2 dir,
    inout float len,
    float w,
    float lTop,
    float lLeft,
    float lCenter,
    float lRight,
    float lBot
) {
    float dc = lRight - lCenter;
    float cb = lCenter - lLeft;
    float lenX = max(abs(dc), abs(cb));
    float dirX = lRight - lLeft;
    dir.x += dirX * w;
    float normX = (lenX > 1e-5) ? saturate(abs(dirX) / lenX) : 0.0;
    len += (normX * normX) * w;

    float ec = lBot - lCenter;
    float ca = lCenter - lTop;
    float lenY = max(abs(ec), abs(ca));
    float dirY = lBot - lTop;
    dir.y += dirY * w;
    float normY = (lenY > 1e-5) ? saturate(abs(dirY) / lenY) : 0.0;
    len += (normY * normY) * w;
}

void easu_tap(
    inout float3 aC,
    inout float aW,
    float2 off,
    float2 dirX,
    float2 dirY,
    float lob,
    float clp,
    float3 c
) {
    float aX = dot(off, dirX);
    float aY = dot(off, dirY);
    float d2 = min(aX * aX + aY * aY, clp);
    float wB = (2.0 / 5.0) * d2 - 1.0;
    float wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = (25.0 / 16.0) * wB - (25.0 / 16.0 - 1.0);
    float w = wB * wA;
    aC += c * w;
    aW += w;
}

float3 srgb_to_linear(float3 c) {
    float3 lo = c * (1.0 / 12.92);
    float3 hi = pow(abs((c + 0.055) * (1.0 / 1.055)), 2.4);
    return float3(
        c.x <= 0.04045 ? lo.x : hi.x,
        c.y <= 0.04045 ? lo.y : hi.y,
        c.z <= 0.04045 ? lo.z : hi.z
    );
}

void vs(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD0) {
    uv = float2((id << 1u) & 2u, id & 2u);
    pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float4 ps_opaque(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    uint srcW, srcH;
    srcTex.GetDimensions(srcW, srcH);
    if (srcW == 0u || srcH == 0u || dstW == 0u || dstH == 0u) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    int2 srcMax = int2(int(srcW) - 1, int(srcH) - 1);
    if (srcW == dstW && srcH == dstH) {
        float4 c = load_safe(int2(pos.xy), srcMax);
        return float4(c.rgb, 1.0);
    }
    if (srcW >= dstW && srcH >= dstH) {
        float4 c = srcTex.SampleLevel(linSamp, uv, 0.0);
        return float4(
            sanitize_f(c.r, 0.0),
            sanitize_f(c.g, 0.0),
            sanitize_f(c.b, 0.0),
            1.0
        );
    }

    float2 srcSize = float2(srcW, srcH);
    float2 dstSize = float2(dstW, dstH);
    float2 invScale = srcSize / dstSize;
    float2 pp = pos.xy * invScale - 0.5;
    float2 fp = floor(pp);
    float2 f = pp - fp;
    int2 base = int2(fp);

    float3 tb = load_safe_rgb(base + int2( 0, -1), srcMax);
    float3 tc = load_safe_rgb(base + int2( 1, -1), srcMax);
    float3 te = load_safe_rgb(base + int2(-1,  0), srcMax);
    float4 tf = load_safe(base + int2( 0,  0), srcMax);
    float4 tg = load_safe(base + int2( 1,  0), srcMax);
    float3 th = load_safe_rgb(base + int2( 2,  0), srcMax);
    float3 ti = load_safe_rgb(base + int2(-1,  1), srcMax);
    float4 tj = load_safe(base + int2( 0,  1), srcMax);
    float4 tk = load_safe(base + int2( 1,  1), srcMax);
    float3 tl = load_safe_rgb(base + int2( 2,  1), srcMax);
    float3 tn = load_safe_rgb(base + int2( 0,  2), srcMax);
    float3 to = load_safe_rgb(base + int2( 1,  2), srcMax);

    float lB = hdr_luma(tb);
    float lC = hdr_luma(tc);
    float lE = hdr_luma(te);
    float lF = hdr_luma(tf.rgb);
    float lG = hdr_luma(tg.rgb);
    float lH = hdr_luma(th);
    float lI = hdr_luma(ti);
    float lJ = hdr_luma(tj.rgb);
    float lK = hdr_luma(tk.rgb);
    float lL = hdr_luma(tl);
    float lN = hdr_luma(tn);
    float lO = hdr_luma(to);

    float2 dir = float2(0.0, 0.0);
    float len = 0.0;
    easu_set(dir, len, (1.0 - f.x) * (1.0 - f.y), lB, lE, lF, lG, lJ);
    easu_set(dir, len,        f.x  * (1.0 - f.y), lC, lF, lG, lH, lK);
    easu_set(dir, len, (1.0 - f.x) *        f.y,  lF, lI, lJ, lK, lN);
    easu_set(dir, len,        f.x  *        f.y,  lG, lJ, lK, lL, lO);

    float dirR = dot(dir, dir);
    bool zro = dirR < (1.0 / 32768.0);
    dir = zro ? float2(1.0, 0.0) : (dir * rsqrt(dirR));

    len = len * 0.5;
    len *= len;
    float stretch = dot(dir, dir) * rcp(max(abs(dir.x), abs(dir.y)));
    float2 len2 = float2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    float2 dirX = dir * len2.x;
    float2 dirY = float2(-dir.y, dir.x) * len2.y;
    float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp = rcp(lob);

    float3 aC = float3(0.0, 0.0, 0.0);
    float aW = 0.0;
    easu_tap(aC, aW, float2( 0.0, -1.0) - f, dirX, dirY, lob, clp, tb);
    easu_tap(aC, aW, float2( 1.0, -1.0) - f, dirX, dirY, lob, clp, tc);
    easu_tap(aC, aW, float2(-1.0,  0.0) - f, dirX, dirY, lob, clp, te);
    easu_tap(aC, aW, float2( 0.0,  0.0) - f, dirX, dirY, lob, clp, tf.rgb);
    easu_tap(aC, aW, float2( 1.0,  0.0) - f, dirX, dirY, lob, clp, tg.rgb);
    easu_tap(aC, aW, float2( 2.0,  0.0) - f, dirX, dirY, lob, clp, th);
    easu_tap(aC, aW, float2(-1.0,  1.0) - f, dirX, dirY, lob, clp, ti);
    easu_tap(aC, aW, float2( 0.0,  1.0) - f, dirX, dirY, lob, clp, tj.rgb);
    easu_tap(aC, aW, float2( 1.0,  1.0) - f, dirX, dirY, lob, clp, tk.rgb);
    easu_tap(aC, aW, float2( 2.0,  1.0) - f, dirX, dirY, lob, clp, tl);
    easu_tap(aC, aW, float2( 0.0,  2.0) - f, dirX, dirY, lob, clp, tn);
    easu_tap(aC, aW, float2( 1.0,  2.0) - f, dirX, dirY, lob, clp, to);

    float3 min4 = min(min(tf.rgb, tg.rgb), min(tj.rgb, tk.rgb));
    float3 max4 = max(max(tf.rgb, tg.rgb), max(tj.rgb, tk.rgb));
    float3 easuRgb = clamp(aC * rcp(max(aW, 1e-5)), min4, max4);

    float2 step = min(invScale, float2(1.0, 1.0));
    float3 rowTop   = lerp(tb, tc, f.x);
    float3 row0     = lerp(tf.rgb, tg.rgb, f.x);
    float3 row1     = lerp(tj.rgb, tk.rgb, f.x);
    float3 rowBot   = lerp(tn, to, f.x);
    float3 colLeft  = lerp(te, ti, f.y);
    float3 col0     = lerp(tf.rgb, tj.rgb, f.y);
    float3 col1     = lerp(tg.rgb, tk.rgb, f.y);
    float3 colRight = lerp(th, tl, f.y);

    float yN = f.y - step.y;
    float3 nbN = (yN < 0.0) ? lerp(rowTop, row0, yN + 1.0) : lerp(row0, row1, yN);
    float yS = f.y + step.y;
    float3 nbS = (yS < 1.0) ? lerp(row0, row1, yS) : lerp(row1, rowBot, yS - 1.0);
    float xW = f.x - step.x;
    float3 nbW = (xW < 0.0) ? lerp(colLeft, col0, xW + 1.0) : lerp(col0, col1, xW);
    float xE = f.x + step.x;
    float3 nbE = (xE < 1.0) ? lerp(col0, col1, xE) : lerp(col1, colRight, xE - 1.0);

    float pC = hdr_luma(easuRgb);
    float pN = hdr_luma(nbN);
    float pS = hdr_luma(nbS);
    float pW = hdr_luma(nbW);
    float pE = hdr_luma(nbE);

    float mnL = min(pC, min(min(pN, pS), min(pW, pE)));
    float mxL = max(pC, max(max(pN, pS), max(pW, pE)));
    float hitMin = mnL / max(4.0 * mxL, 1e-5);
    float hitMax = (1.0 - mxL) / max(4.0 * (1.0 - mnL), 1e-5);
    float lobe = max(-0.1875, -min(hitMin, hitMax));
    float gradGate = smoothstep(0.0015, 0.012, mxL - mnL);
    float wRcas = lobe * (RCAS_SHARPNESS * gradGate);

    float pDelta = (wRcas * rcp(1.0 + 4.0 * wRcas)) * ((pN + pS + pW + pE) - 4.0 * pC);
    float hdrFade = saturate((0.99 - pC) * 20.0);
    float dp = pDelta * hdrFade;
    float pSharp = clamp(pC + dp, 0.0, 0.995);

    float gain = 1.0;
    if (pC > 1e-4 && abs(dp) > 1e-6) {
        float r = pSharp / pC;
        float numer = max(1.0 - pC * pC, 1e-4);
        float denom = max(1.0 - pSharp * pSharp, 1e-4);
        gain = clamp((r * r) * (numer / denom), 0.5, 2.0);
    }

    float3 outRgb = clamp(easuRgb * gain, min4, max4);
    return float4(outRgb, 1.0);
}

float4 ps_overlay(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    uint srcW, srcH;
    srcTex.GetDimensions(srcW, srcH);
    if (srcW == 0u || srcH == 0u) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    float4 raw = (srcW == dstW && srcH == dstH)
        ? srcTex.Load(int3(clamp(int2(pos.xy), int2(0, 0), int2(int(srcW) - 1, int(srcH) - 1)), 0))
        : srcTex.SampleLevel(linSamp, uv, 0.0);
    float a = isfinite(raw.a) ? saturate(raw.a) : 0.0;
    float3 rgb = float3(
        sanitize_f(raw.r, 0.0),
        sanitize_f(raw.g, 0.0),
        sanitize_f(raw.b, 0.0)
    );
    if ((flags & 2u) != 0u) {
        if ((flags & 1u) != 0u) {
            rgb = srgb_to_linear(saturate(rgb)) * paperWhite;
        } else if (a > 1e-4) {
            float3 unpremul = saturate(rgb * rcp(a));
            rgb = srgb_to_linear(unpremul) * (paperWhite * a);
        } else {
            rgb = srgb_to_linear(saturate(rgb)) * paperWhite;
        }
    }
    return float4(rgb, a);
}
"#;

struct PlaneEnv {
    enabled: bool,
    straight_alpha: bool,
    output_base: bool,
    paper_white: f32,
}

fn plane_env() -> &'static PlaneEnv {
    static ENV: OnceLock<PlaneEnv> = OnceLock::new();
    ENV.get_or_init(|| {
        let var = |k: &str| std::env::var(k).ok();
        PlaneEnv {
            enabled: var("VICTORY_PLANES").is_none_or(|v| v != "0"),
            straight_alpha: var("VICTORY_PLANE_ALPHA")
                .is_some_and(|v| v.eq_ignore_ascii_case("straight")),
            output_base: var("VICTORY_PLANE_OUT").is_some_and(|v| v.eq_ignore_ascii_case("base"))
                || var("VICTORY_PRESENT_1080P").is_some_and(|v| v != "0"),
            paper_white: var("VICTORY_PLANE_PAPER_WHITE")
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(2.5),
        }
    })
}

/// Whether multi-plane `PresentX` compositing is enabled (`VICTORY_PLANES != "0"`).
pub(crate) fn planes_enabled() -> bool {
    plane_env().enabled
}

/// Whether multi-plane `PresentX` should keep the swap chain at plane 0's base
/// resolution (`VICTORY_PLANE_OUT=base` / `VICTORY_PRESENT_1080P=1`) instead of
/// growing to the largest overlay plane.
pub(crate) fn plane_output_base() -> bool {
    plane_env().output_base
}

fn is_float_format(fmt: DXGI_FORMAT) -> bool {
    matches!(fmt.0, 9..=14 | 26)
}

/// Built-in fullscreen-triangle graphics compositor for multi-plane `PresentX`
/// and `R11G11B10_FLOAT` / scaled swap-chain presentation, with flight-safe
/// ringed RTV and SRV descriptor heaps for `late_free_present`.
pub(crate) struct PresentCompositor {
    format: DXGI_FORMAT,
    root: ID3D12RootSignature,
    opaque: ID3D12PipelineState,
    overlay_premul: ID3D12PipelineState,
    overlay_straight: ID3D12PipelineState,
    rtv_heap: ID3D12DescriptorHeap,
    rtv_step: u32,
    next_rtv: AtomicU32,
    srv_heap: ID3D12DescriptorHeap,
    srv_step: u32,
    next_srv: AtomicU32,
}

impl PresentCompositor {
    pub(crate) fn format(&self) -> DXGI_FORMAT {
        self.format
    }

    pub(crate) fn new(device: &ID3D12Device, format: DXGI_FORMAT) -> windows::core::Result<Self> {
        let compile = |entry: PCSTR, target: PCSTR| -> windows::core::Result<Vec<u8>> {
            compile_fxc_stage(
                PRESENT_COMPOSITE_HLSL,
                s!("present_composite"),
                entry,
                target,
            )
            .map_err(|(e, msg)| {
                if !msg.is_empty() {
                    tracing::warn!("present compositor D3DCompile: {msg}");
                }
                e
            })
        };
        let vs_bytes = compile(s!("vs"), s!("vs_5_1"))?;
        let ps_opaque_bytes = compile(s!("ps_opaque"), s!("ps_5_1"))?;
        let ps_overlay_bytes = compile(s!("ps_overlay"), s!("ps_5_1"))?;

        let srv_range = D3D12_DESCRIPTOR_RANGE1 {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &raw const srv_range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 4,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
        ];
        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            BorderColor: D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            MinLOD: 0.0,
            MaxLOD: f32::MAX,
            ShaderRegister: 0,
            RegisterSpace: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            ..Default::default()
        };
        let rs_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 1,
                    pStaticSamplers: &raw const sampler,
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
                },
            },
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut err: Option<ID3DBlob> = None;
        // SAFETY: valid root signature description and compiled VS/PS shader blobs.
        unsafe {
            D3D12SerializeVersionedRootSignature(&rs_desc, &mut blob, Some(&mut err))?;
            let blob = blob.ok_or_else(windows::core::Error::empty)?;
            let rs_bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, rs_bytes)?;

            let make_pso = |ps: &[u8], blend: Option<D3D12_BLEND>| -> windows::core::Result<ID3D12PipelineState> {
                let mut rt_blend = D3D12_RENDER_TARGET_BLEND_DESC {
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
                if let Some(src_blend) = blend {
                    rt_blend.BlendEnable = true.into();
                    rt_blend.SrcBlend = src_blend;
                    rt_blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                    rt_blend.BlendOp = D3D12_BLEND_OP_ADD;
                    rt_blend.SrcBlendAlpha = D3D12_BLEND_ONE;
                    rt_blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                    rt_blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                }
                let mut blend_desc = D3D12_BLEND_DESC::default();
                blend_desc.RenderTarget[0] = rt_blend;
                let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
                rtv_formats[0] = format;
                device.CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    // Borrowed pointer; `ManuallyDrop` inside `pRootSignature` never releases it.
                    pRootSignature: std::mem::transmute_copy(&root),
                    VS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: vs_bytes.as_ptr().cast(),
                        BytecodeLength: vs_bytes.len(),
                    },
                    PS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: ps.as_ptr().cast(),
                        BytecodeLength: ps.len(),
                    },
                    BlendState: blend_desc,
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        DepthClipEnable: false.into(),
                        ..Default::default()
                    },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    RTVFormats: rtv_formats,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                })
            };

            let opaque = make_pso(&ps_opaque_bytes, None)?;
            let overlay_premul = make_pso(&ps_overlay_bytes, Some(D3D12_BLEND_ONE))?;
            let overlay_straight = make_pso(&ps_overlay_bytes, Some(D3D12_BLEND_SRC_ALPHA))?;

            let rtv_heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: COMPOSE_RTV_SLOTS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })?;
            let rtv_step =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

            let srv_heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: COMPOSE_SRV_SETS * (MAX_COMPOSE_PLANES as u32),
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let srv_step =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            Ok(Self {
                format,
                root,
                opaque,
                overlay_premul,
                overlay_straight,
                rtv_heap,
                rtv_step,
                next_rtv: AtomicU32::new(0),
                srv_heap,
                srv_step,
                next_srv: AtomicU32::new(0),
            })
        }
    }

    /// Composites `planes` (`(texture, host_state)` pairs, plane 0 opaque with
    /// EASU+RCAS upscaling when smaller than `width x height`, planes `1..`
    /// alpha-blended on top) directly into `back` (transitioned from
    /// `back_before` to `RENDER_TARGET`, and afterwards to `back_after`),
    /// restoring every plane texture to its original `host_state`.
    ///
    /// Never calls `list.keep(back)` so swap-chain `ResizeBuffers` never sees
    /// lingering back-buffer refcounts across settled `ahead` lists.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn composite(
        &self,
        device: &ID3D12Device,
        list: &HostList,
        back: &ID3D12Resource,
        back_before: D3D12_RESOURCE_STATES,
        back_after: D3D12_RESOURCE_STATES,
        width: u32,
        height: u32,
        planes: &[(&ID3D12Resource, u32)],
    ) -> bool {
        let count = planes.len().min(MAX_COMPOSE_PLANES);
        if count == 0 || width == 0 || height == 0 {
            return false;
        }
        let env = plane_env();
        let out_is_hdr_float = is_float_format(self.format);
        // SAFETY: live device, open command list, live swap-chain target and plane resources.
        unsafe {
            for &(tex, _) in &planes[..count] {
                let d = tex.GetDesc();
                if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
                    || d.Width == 0
                    || d.Height == 0
                    || d.SampleDesc.Count != 1
                    || d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 != 0
                {
                    return false;
                }
            }

            let rtv_slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % COMPOSE_RTV_SLOTS;
            let rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.rtv_heap.GetCPUDescriptorHandleForHeapStart().ptr
                    + (rtv_slot * self.rtv_step) as usize,
            };
            device.CreateRenderTargetView(
                back,
                Some(&D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: self.format,
                    ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_RTV {
                            MipSlice: 0,
                            PlaneSlice: 0,
                        },
                    },
                }),
                rtv,
            );

            let srv_set = self.next_srv.fetch_add(1, Ordering::Relaxed) % COMPOSE_SRV_SETS;
            let srv_base = srv_set * (MAX_COMPOSE_PLANES as u32);
            let srv_cpu_start = self.srv_heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let srv_gpu_start = self.srv_heap.GetGPUDescriptorHandleForHeapStart().ptr;

            let mut seen_raw: Vec<*mut std::ffi::c_void> = Vec::with_capacity(count);
            let mut before = Vec::with_capacity(count + 1);
            let mut plane_formats = [DXGI_FORMAT_UNKNOWN; MAX_COMPOSE_PLANES];

            for (i, &(tex, state)) in planes[..count].iter().enumerate() {
                let d = tex.GetDesc();
                plane_formats[i] = d.Format;
                let srv_fmt = concrete_hdr_format(d.Format);
                let srv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: srv_cpu_start + ((srv_base + i as u32) * self.srv_step) as usize,
                };
                device.CreateShaderResourceView(
                    tex,
                    Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: srv_fmt,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV {
                                MostDetailedMip: 0,
                                MipLevels: 1,
                                PlaneSlice: 0,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    }),
                    srv_cpu,
                );
                list.keep(tex);
                let raw = tex.as_raw();
                if !seen_raw.contains(&raw) {
                    seen_raw.push(raw);
                    let state = D3D12_RESOURCE_STATES(state as i32);
                    if state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                        before.push(crate::host::transition(
                            tex,
                            state,
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        ));
                    }
                }
            }

            if back_before != D3D12_RESOURCE_STATE_RENDER_TARGET {
                before.push(crate::host::transition(
                    back,
                    back_before,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                ));
            }
            let l = &list.list;
            if !before.is_empty() {
                l.ResourceBarrier(&before);
            }

            let overlay_pso = if env.straight_alpha {
                &self.overlay_straight
            } else {
                &self.overlay_premul
            };
            list.keep(&self.srv_heap);
            list.keep(&self.root);
            list.keep(&self.opaque);
            list.keep(overlay_pso);

            l.SetDescriptorHeaps(&[Some(self.srv_heap.clone())]);
            l.SetGraphicsRootSignature(&self.root);
            l.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            l.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);
            l.OMSetRenderTargets(1, Some(&rtv), false, None);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            for (i, &fmt) in plane_formats[..count].iter().enumerate() {
                l.SetPipelineState(if i == 0 { &self.opaque } else { overlay_pso });
                let sdr_on_hdr = i > 0 && out_is_hdr_float && !is_float_format(fmt);
                let flags = u32::from(env.straight_alpha) | (u32::from(sdr_on_hdr) << 1);
                let constants = [width, height, flags, env.paper_white.to_bits()];
                l.SetGraphicsRootDescriptorTable(
                    0,
                    D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: srv_gpu_start
                            + u64::from(srv_base + i as u32) * u64::from(self.srv_step),
                    },
                );
                l.SetGraphicsRoot32BitConstants(1, 4, constants.as_ptr().cast(), 0);
                l.DrawInstanced(3, 1, 0, 0);
            }

            let mut after = Vec::with_capacity(count + 1);
            if back_after != D3D12_RESOURCE_STATE_RENDER_TARGET {
                after.push(crate::host::transition(
                    back,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    back_after,
                ));
            }
            seen_raw.clear();
            for &(tex, state) in &planes[..count] {
                let raw = tex.as_raw();
                if !seen_raw.contains(&raw) {
                    seen_raw.push(raw);
                    let state = D3D12_RESOURCE_STATES(state as i32);
                    if state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                        after.push(crate::host::transition(
                            tex,
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                            state,
                        ));
                    }
                }
            }
            if !after.is_empty() {
                l.ResourceBarrier(&after);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dxc_exe() -> Option<PathBuf> {
        let p = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => PathBuf::from(p),
            None => victory_shader::emit::validator::Validator::default_path()?
                .with_file_name("dxc.exe"),
        };
        p.exists().then_some(p)
    }

    #[test]
    fn concrete_hdr_format_maps_typeless_formats() {
        assert_eq!(concrete_hdr_format(DXGI_FORMAT(9)), DXGI_FORMAT(10));
        assert_eq!(concrete_hdr_format(DXGI_FORMAT(10)), DXGI_FORMAT(10));
        assert_eq!(concrete_hdr_format(DXGI_FORMAT(23)), DXGI_FORMAT(24));
        assert_eq!(concrete_hdr_format(DXGI_FORMAT(24)), DXGI_FORMAT(24));
        assert_eq!(concrete_hdr_format(DXGI_FORMAT(26)), DXGI_FORMAT(26));
        assert_eq!(concrete_hdr_format(DXGI_FORMAT(27)), DXGI_FORMAT(28));
        assert_eq!(concrete_hdr_format(DXGI_FORMAT(28)), DXGI_FORMAT(28));
        assert_eq!(concrete_hdr_format(DXGI_FORMAT(29)), DXGI_FORMAT(28));
        assert_eq!(concrete_hdr_format(DXGI_FORMAT(87)), DXGI_FORMAT(87));
        assert_eq!(concrete_hdr_format(DXGI_FORMAT(90)), DXGI_FORMAT(87));
        assert_eq!(concrete_hdr_format(DXGI_FORMAT(91)), DXGI_FORMAT(87));
    }

    /// Both `UPSCALE_HLSL` and `PRESENT_COMPOSITE_HLSL` compile cleanly with `D3DCompile`.
    #[test]
    fn hlsl_compiles_fxc() {
        match compile_fxc(UPSCALE_HLSL) {
            Ok(bytes) => assert!(bytes.starts_with(b"DXBC"), "expected DXBC container"),
            Err((e, msg)) => panic!("upscale HLSL D3DCompile failed ({e}): {msg}"),
        }
        for (entry, target) in [
            (s!("vs"), s!("vs_5_1")),
            (s!("ps_opaque"), s!("ps_5_1")),
            (s!("ps_overlay"), s!("ps_5_1")),
        ] {
            match compile_fxc_stage(PRESENT_COMPOSITE_HLSL, s!("present_composite"), entry, target)
            {
                Ok(bytes) => assert!(bytes.starts_with(b"DXBC"), "expected DXBC container"),
                Err((e, msg)) => panic!("composite HLSL D3DCompile failed ({e}): {msg}"),
            }
        }
    }

    /// Both `UPSCALE_HLSL` and `PRESENT_COMPOSITE_HLSL` compile cleanly with `dxc.exe` (`-O3 -WX`).
    #[test]
    fn hlsl_compiles_dxc() {
        let Some(dxc) = dxc_exe() else {
            eprintln!("skipped upscale::tests::hlsl_compiles_dxc: no PC dxc.exe");
            return;
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let check = |file: &str, hlsl: &str, profile: &str, entry: &str| {
            let src_path = dir.path().join(format!("{file}_{entry}.hlsl"));
            let out_path = dir.path().join(format!("{file}_{entry}.dxil"));
            std::fs::write(&src_path, hlsl).expect("write hlsl");
            let output = std::process::Command::new(&dxc)
                .args(["-T", profile, "-E", entry, "-O3", "-WX", "-Fo"])
                .arg(&out_path)
                .arg(&src_path)
                .output()
                .expect("spawn dxc.exe");
            assert!(
                output.status.success(),
                "dxc.exe ({file}:{entry}) failed:\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            let dxil = std::fs::read(&out_path).expect("read dxil");
            assert!(dxil.len() > 32 && dxil.starts_with(b"DXBC"), "invalid DXIL container");
        };
        check("upscale", UPSCALE_HLSL, "cs_6_0", "main");
        check("composite", PRESENT_COMPOSITE_HLSL, "vs_6_0", "vs");
        check("composite", PRESENT_COMPOSITE_HLSL, "ps_6_0", "ps_opaque");
        check("composite", PRESENT_COMPOSITE_HLSL, "ps_6_0", "ps_overlay");
    }

    /// The upscaler compiles and the host D3D12 device accepts its root signature and PSO.
    #[test]
    fn upscaler_builds() {
        if host().is_none() {
            return;
        }
        assert!(upscaler().is_some(), "built-in EASU+RCAS upscaler failed to build");
    }

    fn f32_to_f16(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let frac = bits & 0x7fffff;
        if exp == 255 {
            if frac != 0 {
                return sign | 0x7e00;
            }
            return sign | 0x7c00;
        }
        let unbiased = exp - 127;
        if unbiased > 15 {
            return sign | 0x7c00;
        }
        if unbiased < -14 {
            if unbiased < -24 {
                return sign;
            }
            let shift = (-14 - unbiased) as u32;
            let mant = (frac | 0x800000) >> (shift + 13);
            return sign | mant as u16;
        }
        sign | (((unbiased + 15) as u16) << 10) | ((frac >> 13) as u16)
    }

    fn f16_to_f32(h: u16) -> f32 {
        let sign = u32::from(h & 0x8000) << 16;
        let exp = u32::from((h >> 10) & 0x1f);
        let frac = u32::from(h & 0x3ff);
        let bits = if exp == 0 {
            if frac == 0 {
                sign
            } else {
                let mut e = 127 - 14;
                let mut f = frac;
                while (f & 0x400) == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3ff;
                sign | (e << 23) | (f << 13)
            }
        } else if exp == 31 {
            sign | (0xff << 23) | (frac << 13)
        } else {
            sign | ((exp + 127 - 15) << 23) | (frac << 13)
        };
        f32::from_bits(bits)
    }

    fn f32_to_ufloat(v: f32, mant_bits: u32) -> u32 {
        if v.is_nan() {
            return (0x1f << mant_bits) | 1;
        }
        if v <= 0.0 {
            return 0;
        }
        if v.is_infinite() {
            return 0x1f << mant_bits;
        }
        let bits = v.to_bits();
        let exp = ((bits >> 23) & 0xff) as i32 - 127;
        let frac = bits & 0x7fffff;
        if exp > 15 {
            return (0x1e << mant_bits) | ((1 << mant_bits) - 1);
        }
        if exp < -14 {
            return 0;
        }
        (((exp + 15) as u32) << mant_bits) | (frac >> (23 - mant_bits))
    }

    fn ufloat_to_f32(u: u32, mant_bits: u32) -> f32 {
        let exp = (u >> mant_bits) & 0x1f;
        let frac = u & ((1 << mant_bits) - 1);
        let bits = if exp == 0 {
            if frac == 0 {
                0
            } else {
                let mut e = 127 - 14;
                let mut f = frac;
                let top = 1 << mant_bits;
                while (f & top) == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= top - 1;
                (e << 23) | (f << (23 - mant_bits))
            }
        } else if exp == 31 {
            (0xff << 23) | (frac << (23 - mant_bits))
        } else {
            ((exp + 127 - 15) << 23) | (frac << (23 - mant_bits))
        };
        f32::from_bits(bits)
    }

    fn pack_r11g11b10(r: f32, g: f32, b: f32) -> u32 {
        f32_to_ufloat(r, 6) | (f32_to_ufloat(g, 6) << 11) | (f32_to_ufloat(b, 5) << 22)
    }

    fn unpack_r11g11b10(p: u32) -> [f32; 3] {
        [
            ufloat_to_f32(p & 0x7ff, 6),
            ufloat_to_f32((p >> 11) & 0x7ff, 6),
            ufloat_to_f32((p >> 22) & 0x3ff, 5),
        ]
    }

    fn create_tex2d(
        device: &ID3D12Device,
        w: u32,
        h: u32,
        fmt: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        initial: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(w),
            Height: h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: fmt,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: flags,
            ..Default::default()
        };
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: valid D3D12 device and texture description.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, initial, None, &mut res)
                .expect("CreateCommittedResource texture");
        }
        res.expect("texture")
    }

    fn run_gpu_upscale(
        fmt: DXGI_FORMAT,
        bpp: usize,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        src_pixels: &[u8],
    ) -> (Vec<u8>, usize) {
        let h = host().expect("host GPU");
        let list = h.new_list().expect("host command list");
        let src_tex = create_tex2d(
            &h.device,
            src_w,
            src_h,
            fmt,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let dst_tex = create_tex2d(
            &h.device,
            dst_w,
            dst_h,
            fmt,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let mut src_fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut src_total = 0u64;
        let mut dst_fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut dst_total = 0u64;
        // SAFETY: live device and texture descriptors.
        unsafe {
            let sdesc = src_tex.GetDesc();
            let ddesc = dst_tex.GetDesc();
            h.device.GetCopyableFootprints(
                &sdesc,
                0,
                1,
                0,
                Some(&mut src_fp),
                None,
                None,
                Some(&mut src_total),
            );
            h.device.GetCopyableFootprints(
                &ddesc,
                0,
                1,
                0,
                Some(&mut dst_fp),
                None,
                None,
                Some(&mut dst_total),
            );
        }
        let (upload_buf, upload_ptr) = h.shared_buffer(src_total.max(256)).expect("upload");
        let (read_buf, read_ptr) = h.shared_buffer(dst_total.max(256)).expect("readback");
        let src_row_pitch = src_fp.Footprint.RowPitch as usize;
        let dst_row_pitch = dst_fp.Footprint.RowPitch as usize;

        // SAFETY: `upload_ptr` is mapped for `src_total` bytes and `list` is open.
        unsafe {
            for y in 0..src_h as usize {
                let row_bytes = src_w as usize * bpp;
                std::ptr::copy_nonoverlapping(
                    src_pixels[y * row_bytes..].as_ptr(),
                    upload_ptr.add(y * src_row_pitch),
                    row_bytes,
                );
            }
            let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&src_tex),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&upload_buf),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: src_fp },
            };
            list.list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
            list.list.ResourceBarrier(&[crate::host::transition(
                &src_tex,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
        }

        assert!(upscale_texture(&list, &src_tex, &dst_tex));

        // SAFETY: `dst_tex` and `read_buf` are live until `host.execute` finishes.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::transition(
                &dst_tex,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let read_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&read_buf),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: dst_fp },
            };
            let tex_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&dst_tex),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            list.list.CopyTextureRegion(&read_loc, 0, 0, 0, &tex_loc, None);
        }
        list.close();
        h.execute(&[&list]);

        // SAFETY: GPU execution finished; `read_ptr` holds `dst_total` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(read_ptr, dst_total as usize).to_vec() };
        (bytes, dst_row_pitch)
    }

    /// End-to-end GPU verification on `R16G16B16A16_FLOAT` (`f10`) and
    /// `R11G11B10_FLOAT` (`f26`) at 1.5x scale (exact 720p -> 1080p ratio):
    /// flat HDR accuracy, crisp anti-ringing step edges, extreme specular
    /// preservation, and zero NaN/Inf/negative propagation.
    #[test]
    fn gpu_upscale_hdr_quality_and_nan_safety() {
        if host().is_none() {
            return;
        }
        const SW: u32 = 16;
        const SH: u32 = 16;
        const DW: u32 = 24;
        const DH: u32 = 24;

        let mut src_f10 = vec![0u8; (SW * SH) as usize * 8];
        let set_f10 = |buf: &mut [u8], x: u32, y: u32, rgba: [f32; 4]| {
            let off = (y * SW + x) as usize * 8;
            for (c, &v) in rgba.iter().enumerate() {
                buf[off + c * 2..off + c * 2 + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes());
            }
        };
        for y in 0..SH {
            for x in 0..SW {
                let rgba = if y < 4 {
                    [2.5, 1.25, 0.5, 0.75]
                } else if y < 8 {
                    let v = if x < 8 { 0.05 } else { 12.0 };
                    [v, v, v, 1.0]
                } else if y < 12 {
                    [20000.0, 20000.0, 20000.0, 1.0]
                } else {
                    match (x, y) {
                        (4, 14) => [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
                        (8, 14) => [-5.0, -0.25, f32::NAN, -1.0],
                        _ => [0.5, 0.5, 0.5, 1.0],
                    }
                };
                set_f10(&mut src_f10, x, y, rgba);
            }
        }
        let (out_f10, pitch_f10) =
            run_gpu_upscale(DXGI_FORMAT(10), 8, SW, SH, DW, DH, &src_f10);
        let get_f10 = |x: u32, y: u32| -> [f32; 4] {
            let off = y as usize * pitch_f10 + x as usize * 8;
            [
                f16_to_f32(u16::from_le_bytes([out_f10[off], out_f10[off + 1]])),
                f16_to_f32(u16::from_le_bytes([out_f10[off + 2], out_f10[off + 3]])),
                f16_to_f32(u16::from_le_bytes([out_f10[off + 4], out_f10[off + 5]])),
                f16_to_f32(u16::from_le_bytes([out_f10[off + 6], out_f10[off + 7]])),
            ]
        };

        for y in 0..DH {
            for x in 0..DW {
                let p = get_f10(x, y);
                for (c, &v) in p.iter().enumerate() {
                    assert!(v.is_finite(), "non-finite f10 output at ({x},{y}) ch {c}: {v}");
                    assert!(v >= 0.0, "negative f10 output at ({x},{y}) ch {c}: {v}");
                }
            }
        }

        for x in 4..20 {
            let p = get_f10(x, 2);
            assert!((p[0] - 2.5).abs() < 0.02, "flat R drift at x={x}: {}", p[0]);
            assert!((p[1] - 1.25).abs() < 0.02, "flat G drift at x={x}: {}", p[1]);
            assert!((p[2] - 0.5).abs() < 0.02, "flat B drift at x={x}: {}", p[2]);
            assert!((p[3] - 0.75).abs() < 0.02, "flat A drift at x={x}: {}", p[3]);
        }

        for x in 2..22 {
            let p = get_f10(x, 9);
            assert!(
                p[0] >= 0.049 && p[0] <= 12.01,
                "halo/ringing on HDR step edge at x={x}: {}",
                p[0]
            );
        }
        let edge_dark = get_f10(11, 9)[0];
        let edge_bright = get_f10(12, 9)[0];
        let bilinear_dark = 0.05 * (5.0 / 6.0) + 12.0 * (1.0 / 6.0);
        let bilinear_bright = 0.05 * (1.0 / 6.0) + 12.0 * (5.0 / 6.0);
        assert!(
            edge_bright - edge_dark > bilinear_bright - bilinear_dark,
            "expected sharper edge than bilinear: got [{edge_dark}, {edge_bright}] vs bilinear [{bilinear_dark}, {bilinear_bright}]"
        );

        let spec = get_f10(12, 15)[0];
        assert!(
            (spec - 20000.0).abs() < 64.0,
            "extreme HDR specular drifted: got {spec}, expected ~20000.0"
        );

        let mut src_f26 = vec![0u8; (SW * SH) as usize * 4];
        for y in 0..SH {
            for x in 0..SW {
                let p = if x == 2 && y == 14 {
                    pack_r11g11b10(f32::NAN, f32::INFINITY, 1.0)
                } else if x + y < SW {
                    pack_r11g11b10(0.1, 0.2, 0.4)
                } else {
                    pack_r11g11b10(8.0, 4.0, 2.0)
                };
                let off = (y * SW + x) as usize * 4;
                src_f26[off..off + 4].copy_from_slice(&p.to_le_bytes());
            }
        }
        let (out_f26, pitch_f26) =
            run_gpu_upscale(DXGI_FORMAT(26), 4, SW, SH, DW, DH, &src_f26);
        for y in 0..DH {
            for x in 0..DW {
                let off = y as usize * pitch_f26 + x as usize * 4;
                let word = u32::from_le_bytes([
                    out_f26[off],
                    out_f26[off + 1],
                    out_f26[off + 2],
                    out_f26[off + 3],
                ]);
                let rgb = unpack_r11g11b10(word);
                for (c, &v) in rgb.iter().enumerate() {
                    assert!(v.is_finite() && v >= 0.0, "bad f26 at ({x},{y}) ch {c}: {v}");
                }
                if y < 14 {
                    assert!(
                        rgb[0] >= 0.09 && rgb[0] <= 8.05,
                        "f26 diagonal halo at ({x},{y}): {}",
                        rgb[0]
                    );
                }
            }
        }
    }

    /// Smoke test at the title's FSR4 resolution (`1280x720 f10 -> 1920x1080 f10`).
    #[test]
    fn gpu_upscale_720p_to_1080p_smoke() {
        let Some(h) = host() else {
            return;
        };
        let src_tex = create_tex2d(
            &h.device,
            1280,
            720,
            DXGI_FORMAT(10),
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        let dst_tex = create_tex2d(
            &h.device,
            1920,
            1080,
            DXGI_FORMAT(10),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let list = h.new_list().expect("host command list");
        assert!(upscale_texture(&list, &src_tex, &dst_tex));
        list.close();
        h.execute(&[&list]);
    }

    /// `PresentCompositor` builds cleanly for every DXGI flip-model swap-chain format.
    #[test]
    fn present_compositor_builds_all_swap_formats() {
        let Some(h) = host() else {
            return;
        };
        for fmt in [
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            DXGI_FORMAT_R10G10B10A2_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
        ] {
            let comp = PresentCompositor::new(&h.device, fmt)
                .unwrap_or_else(|e| panic!("PresentCompositor::new({}): {e}", fmt.0));
            assert_eq!(comp.format(), fmt);
        }
    }

    /// Verifies 1080p->4K (`2.0x`) and 900p->1080p (`1.2x`) EASU+RCAS quality:
    /// monotonic smooth sky gradients (zero Mach-band stair-stepping), zero ringing
    /// on HDR step edges, and live full-resolution `1920x1080 -> 3840x2160` (`f10`)
    /// + `1600x900 -> 1920x1080` (`f26`) dispatches.
    #[test]
    fn gpu_upscale_1080p_to_4k_and_900p_to_1080p_zero_banding_zero_ringing() {
        let Some(h) = host() else {
            return;
        };

        // 1. 2.0x exact scale (1080p -> 4K ratio: 20x12 -> 40x24) on f10:
        //    Rows 0..6: subtle sky gradient (0.10 -> 0.138, ~1-LSB steps)
        //    Rows 6..12: sharp HDR step edge (0.04 -> 16.0 at x = 10)
        const SW2: u32 = 20;
        const SH2: u32 = 12;
        const DW2: u32 = 40;
        const DH2: u32 = 24;
        let mut src_2x = vec![0u8; (SW2 * SH2) as usize * 8];
        for y in 0..SH2 {
            for x in 0..SW2 {
                let v = if y < 6 {
                    // Quantized 1/255 sky stair-steps to test Mach-band suppression.
                    (25 + (x / 2)) as f32 / 255.0
                } else if x < 10 {
                    0.04
                } else {
                    16.0
                };
                let off = (y * SW2 + x) as usize * 8;
                for (c, ch) in [v, v, v, 1.0].into_iter().enumerate() {
                    src_2x[off + c * 2..off + c * 2 + 2]
                        .copy_from_slice(&f32_to_f16(ch).to_le_bytes());
                }
            }
        }
        let (out_2x, pitch_2x) =
            run_gpu_upscale(DXGI_FORMAT(10), 8, SW2, SH2, DW2, DH2, &src_2x);
        let sample_2x = |x: u32, y: u32| -> f32 {
            let off = y as usize * pitch_2x + x as usize * 8;
            f16_to_f32(u16::from_le_bytes([out_2x[off], out_2x[off + 1]]))
        };
        let min_grad = 25.0f32 / 255.0;
        let max_grad = (25 + ((SW2 - 1) / 2)) as f32 / 255.0;
        let mut prev = sample_2x(0, 4);
        for x in 0..DW2 {
            let g = sample_2x(x, 4);
            assert!(
                g >= min_grad - 1e-3 && g <= max_grad + 1e-3,
                "2x sky gradient overshoot at x={x}: {g}"
            );
            assert!(
                g + 6e-4 >= prev,
                "2x sky gradient non-monotonic Mach band at x={x}: prev={prev}, cur={g}"
            );
            prev = g;

            let e = sample_2x(x, 18);
            assert!(
                (0.039..=16.02).contains(&e),
                "2x HDR step ringing at x={x}: {e}"
            );
        }
        // At 2.0x, x=19 (f=0.75 on left of edge x=9..10) vs x=20 (f=0.25 on right):
        let dark_2x = sample_2x(19, 18);
        let bright_2x = sample_2x(20, 18);
        let bilin_gap = (0.04 * 0.25 + 16.0 * 0.75) - (0.04 * 0.75 + 16.0 * 0.25);
        assert!(
            bright_2x - dark_2x > bilin_gap,
            "expected 2x EASU+RCAS sharper than bilinear: [{dark_2x}, {bright_2x}] vs gap {bilin_gap}"
        );

        // 2. 1.2x exact scale (900p -> 1080p ratio: 20x10 -> 24x12) on R8G8B8A8_UNORM (f28):
        const SW12: u32 = 20;
        const SH12: u32 = 10;
        const DW12: u32 = 24;
        const DH12: u32 = 12;
        let mut src_12x = vec![0u8; (SW12 * SH12) as usize * 4];
        for y in 0..SH12 {
            for x in 0..SW12 {
                let u = (40 + x * 2) as u8;
                let off = (y * SW12 + x) as usize * 4;
                src_12x[off..off + 4].copy_from_slice(&[u, u, u, 255]);
            }
        }
        let (out_12x, pitch_12x) =
            run_gpu_upscale(DXGI_FORMAT(28), 4, SW12, SH12, DW12, DH12, &src_12x);
        let mut prev_u = out_12x[6 * pitch_12x];
        for x in 0..DW12 as usize {
            let cur_u = out_12x[6 * pitch_12x + x * 4];
            assert!(
                (40..=78).contains(&cur_u),
                "1.2x 900p->1080p UNORM out of bounds at x={x}: {cur_u}"
            );
            assert!(
                cur_u + 1 >= prev_u,
                "1.2x 900p->1080p UNORM gradient banding dip at x={x}: {prev_u} -> {cur_u}"
            );
            prev_u = cur_u;
        }

        // 3. Full-resolution smoke: 1920x1080 -> 3840x2160 (f10) & 1600x900 -> 1920x1080 (f26).
        let list = h.new_list().expect("host command list");
        let s_1080 = create_tex2d(
            &h.device,
            1920,
            1080,
            DXGI_FORMAT(10),
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        let d_4k = create_tex2d(
            &h.device,
            3840,
            2160,
            DXGI_FORMAT(10),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let s_900 = create_tex2d(
            &h.device,
            1600,
            900,
            DXGI_FORMAT(26),
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        let d_1080 = create_tex2d(
            &h.device,
            1920,
            1080,
            DXGI_FORMAT(26),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        assert!(upscale_texture(&list, &s_1080, &d_4k));
        assert!(upscale_texture(&list, &s_900, &d_1080));
        list.close();
        h.execute(&[&list]);
    }

    /// End-to-end GPU verification of `PresentCompositor`:
    /// 1. Series S 2-plane present (`1080p` game scene on plane 0 + premultiplied
    ///    `4K` UI/menu overlay on plane 1 -> `4K` `R8G8B8A8_UNORM` back buffer):
    ///    - `alpha == 0` transparent gameplay area preserves EASU+RCAS scene;
    ///    - `alpha == 1` opaque menu area shows exact 4K UI;
    ///    - `alpha == 0.5` translucent HUD blends `scene * (1 - a) + ui_rgb`;
    ///    - duplicate-plane barrier deduplication and full `1920x1080 + 3840x2160` smoke.
    /// 2. HDR `R11G11B10_FLOAT` (`26`) scene + `R8G8B8A8_UNORM` (`28`) UI overlay
    ///    onto `R16G16B16A16_FLOAT` (`10`) RTV with NaN sanitization.
    #[test]
    fn gpu_present_compositor_series_s_two_plane_and_hdr_f26() {
        let Some(h) = host() else {
            return;
        };

        // --- Part 1: 2-plane SDR (16x16 scene + 32x32 UI -> 32x32 back buffer) ---
        let comp_sdr =
            PresentCompositor::new(&h.device, DXGI_FORMAT_R8G8B8A8_UNORM).expect("comp sdr");
        let list = h.new_list().expect("list");
        let plane0 = create_tex2d(
            &h.device,
            16,
            16,
            DXGI_FORMAT(28),
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let plane1 = create_tex2d(
            &h.device,
            32,
            32,
            DXGI_FORMAT(28),
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let back_sdr = create_tex2d(
            &h.device,
            32,
            32,
            DXGI_FORMAT(28),
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT,
        );

        let mut fp0 = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut tot0 = 0u64;
        let mut fp1 = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut tot1 = 0u64;
        let mut fpb = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut totb = 0u64;
        // SAFETY: live device and texture descriptions.
        unsafe {
            h.device.GetCopyableFootprints(
                &plane0.GetDesc(),
                0,
                1,
                0,
                Some(&mut fp0),
                None,
                None,
                Some(&mut tot0),
            );
            h.device.GetCopyableFootprints(
                &plane1.GetDesc(),
                0,
                1,
                0,
                Some(&mut fp1),
                None,
                None,
                Some(&mut tot1),
            );
            h.device.GetCopyableFootprints(
                &back_sdr.GetDesc(),
                0,
                1,
                0,
                Some(&mut fpb),
                None,
                None,
                Some(&mut totb),
            );
        }
        let (up0, p0) = h.shared_buffer(tot0.max(256)).expect("up0");
        let (up1, p1) = h.shared_buffer(tot1.max(256)).expect("up1");
        let (rdb, pb) = h.shared_buffer(totb.max(256)).expect("rdb");

        // Plane 0 (16x16 scene): solid [50, 100, 200, 255].
        // Plane 1 (32x32 UI):
        //   y < 10: transparent gameplay [0, 0, 0, 0]
        //   10 <= y < 20: opaque menu [220, 40, 20, 255]
        //   y >= 20: premultiplied 50%-alpha HUD [100, 50, 0, 128]
        // SAFETY: mapped upload buffers sized from GetCopyableFootprints.
        unsafe {
            for y in 0..16usize {
                for x in 0..16usize {
                    std::ptr::copy_nonoverlapping(
                        [50u8, 100, 200, 255].as_ptr(),
                        p0.add(y * fp0.Footprint.RowPitch as usize + x * 4),
                        4,
                    );
                }
            }
            for y in 0..32usize {
                let px = if y < 10 {
                    [0u8, 0, 0, 0]
                } else if y < 20 {
                    [220u8, 40, 20, 255]
                } else {
                    [100u8, 50, 0, 128]
                };
                for x in 0..32usize {
                    std::ptr::copy_nonoverlapping(
                        px.as_ptr(),
                        p1.add(y * fp1.Footprint.RowPitch as usize + x * 4),
                        4,
                    );
                }
            }
            let copy_up = |dst: &ID3D12Resource, src: &ID3D12Resource, fp| {
                let dloc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(dst),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                let sloc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(src),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
                };
                list.list.CopyTextureRegion(&dloc, 0, 0, 0, &sloc, None);
            };
            copy_up(&plane0, &up0, fp0);
            copy_up(&plane1, &up1, fp1);
            list.list.ResourceBarrier(&[
                crate::host::transition(
                    &plane0,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
                crate::host::transition(
                    &plane1,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                ),
            ]);
        }

        assert!(comp_sdr.composite(
            &h.device,
            &list,
            &back_sdr,
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            32,
            32,
            &[
                (&plane0, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32),
                (&plane1, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32),
            ],
        ));

        // SAFETY: `back_sdr` is now in COPY_SOURCE; read it back and restore PRESENT.
        unsafe {
            let rloc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&rdb),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fpb },
            };
            let tloc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&back_sdr),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            list.list.CopyTextureRegion(&rloc, 0, 0, 0, &tloc, None);
            // Also verify plane0 was restored to COPY_SOURCE and plane1 to RENDER_TARGET
            // by transitioning them cleanly without validation error.
            list.list.ResourceBarrier(&[
                crate::host::transition(
                    &back_sdr,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
                crate::host::transition(
                    &plane0,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
                crate::host::transition(
                    &plane1,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
            ]);
        }
        list.close();
        h.execute(&[&list]);

        // SAFETY: GPU finished; `pb` holds `totb` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(pb, totb as usize) };
        let px = |x: usize, y: usize| -> [u8; 4] {
            let o = y * fpb.Footprint.RowPitch as usize + x * 4;
            [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]
        };
        let game = px(16, 4);
        assert!(
            (i32::from(game[0]) - 50).abs() <= 1
                && (i32::from(game[1]) - 100).abs() <= 1
                && (i32::from(game[2]) - 200).abs() <= 1,
            "transparent UI hid scene plane 0: got {game:?}"
        );
        let menu = px(16, 15);
        assert!(
            (i32::from(menu[0]) - 220).abs() <= 1
                && (i32::from(menu[1]) - 40).abs() <= 1
                && (i32::from(menu[2]) - 20).abs() <= 1,
            "opaque menu overlay mismatch: got {menu:?}"
        );
        // Premultiplied blend: round(50 * (1 - 128/255) + 100) ~= 125,
        // round(100 * (1 - 128/255) + 50) ~= 100, round(200 * (1 - 128/255) + 0) ~= 100
        let hud = px(16, 26);
        assert!(
            (i32::from(hud[0]) - 125).abs() <= 2
                && (i32::from(hud[1]) - 100).abs() <= 2
                && (i32::from(hud[2]) - 100).abs() <= 2,
            "50% premultiplied HUD blend mismatch: got {hud:?}"
        );

        // --- Part 2: HDR f26 scene + f28 UI -> f10 RTV + Series S full-res smoke ---
        let comp_hdr =
            PresentCompositor::new(&h.device, DXGI_FORMAT_R16G16B16A16_FLOAT).expect("comp hdr");
        let list2 = h.new_list().expect("list2");
        let p0_f26 = create_tex2d(
            &h.device,
            16,
            16,
            DXGI_FORMAT(26),
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        let back_hdr = create_tex2d(
            &h.device,
            32,
            32,
            DXGI_FORMAT(10),
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT,
        );
        assert!(comp_hdr.composite(
            &h.device,
            &list2,
            &back_hdr,
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_PRESENT,
            32,
            32,
            &[
                (&p0_f26, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 as u32),
                (&plane1, D3D12_RESOURCE_STATE_COMMON.0 as u32),
            ],
        ));
        // Duplicate-plane guard + Series S full-resolution (1920x1080 + 3840x2160 -> 3840x2160) smoke:
        let ss_p0 = create_tex2d(
            &h.device,
            1920,
            1080,
            DXGI_FORMAT(28),
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        let ss_p1 = create_tex2d(
            &h.device,
            3840,
            2160,
            DXGI_FORMAT(28),
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        let ss_back = create_tex2d(
            &h.device,
            3840,
            2160,
            DXGI_FORMAT(28),
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT,
        );
        assert!(comp_sdr.composite(
            &h.device,
            &list2,
            &ss_back,
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_PRESENT,
            3840,
            2160,
            &[
                (&ss_p0, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32),
                (&ss_p1, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32),
            ],
        ));
        list2.close();
        h.execute(&[&list2]);
    }
}
