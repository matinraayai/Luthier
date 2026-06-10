# Phase 0: `luthier-tool-compile` Driver Inventory

> **Superseded (2026-06-05).** The `luthier-tool-compile` C++ driver was
> removed; the entire device-compile → unbundle → rebundle → host-compile →
> link pipeline now lives in CMake (`luthier_create_offload_bundle` in
> `cmake/modules/LuthierCreateOffloadBundle.cmake`). The ISA matrix is no
> longer auto-enumerated: the user supplies complete offload target IDs via
> the `LUTHIER_HIP_TARGETS` cache variable (or the per-call `TARGETS`
> argument), and CMake parses each into the `--offload-arch=` target ID
> (xnack/sramecc) plus standalone `-m` flags (wavefrontsize64/cumode). When
> unset, one bare target per `CMAKE_HIP_ARCHITECTURES` arch is synthesized.
> The flag contract below (device/host compile flags, bundle label format,
> alignment, probes) still describes what the CMake commands emit.


This document catalogues every behavior the existing `luthier_add_tool`
CMake helper relies on, so that the upcoming custom driver
(`luthier-tool-compile`) honors the same contract. It also captures the
small probes that pin down the upstream-clang/LLVM behavior the driver
will depend on, so we don't re-discover the gotchas at Phase 1.

This is reference material for Phase 1+, not a design spec — design
calls are recorded inline with the corresponding inventory entry.

## Scope

`luthier_add_tool` produces a Luthier *tool*: a host shared object that
HSA / rocprofiler-sdk loads at process init time, plus an embedded HIP
fat binary containing one or more AMDGCN slices that the tool's
dispatch-time instrumentation pipeline executes against application
kernels.

The driver replaces the per-(arch, wave, cumode) compile orchestration,
the extract-then-rebundle dance, the bundle-target-label construction,
and the include/define propagation logic currently scattered across
`cmake/modules/LuthierAddTool.cmake`. The host-side `--cuda-host-only`
compile stays in CMake's HIP language for IDE integration; the driver
produces the `.hipfb` only.

## Helper behaviors the driver MUST preserve

### Device-side compile (per slice)

For each ISA in the matrix, the helper invokes
`${CMAKE_HIP_COMPILER}` (the clang++ resolved by `enable_language(HIP)`
via `hipconfig --hipclangpath`) with:

| Flag | Purpose |
|---|---|
| `-x hip` | Treat input as HIP source. |
| `--cuda-device-only` | Skip host codegen; emit a HIP offload bundle containing the AMDGCN slice + a host placeholder. |
| `--offload-arch=<arch>[:feat±[:feat±]]` | Target arch + optional xnack/sramecc feature suffix (clang's standard AMDGPU target-ID grammar). |
| `-mwavefrontsize64` / `-mno-wavefrontsize64` | Pin wave size. Required for gfx10+ where wave is variable; no-op (wave64) or silently ignored (wave32) on gfx9. |
| `-mcumode` / `-mno-cumode` | Pin CU/WGP mode. Variable on gfx10+ only. |
| `-std=c++<N>` | Mirrors `CMAKE_HIP_STANDARD` (default 20). |
| `-O3` | **Non-optional.** Without `-O3`, HIP-Clang leaves out-of-line device functions (`__assert_fail`, `__ockl_fprintf_*`, lane/popcount intrinsics, `__cxa_*`) as separate post-ISel pre-RA bodies in the embedded `.llvmbc`. `TargetModulePatcherPass::cloneIModuleIntoTarget` then drags those bodies into the target MIR module, where `AMDGPUResourceUsageAnalysisImpl::analyzeResourceUsage` asserts on the first virtual register (no `isPhysical()` guard). `-O3` inlines the helpers into the hooks and DCEs unreachable ones. |
| `-fpass-plugin=${IRPlugin}` | Loads `LuthierToolIRCompilationPlugin`. Runs the IR-level pipeline (`MarkAnnotationsPass`, `FinalizeIntrinsicsPass`, `StripKernelsPass`, `ExternalizeGlobalsPass`, `SubstituteAMDGCNIntrinsicsPass`, `LoadHIPFATBinaryInfoPass`, `LuthierFunctionIndirectionPass` *(currently disabled)*, `CreateAndEmbedIModulePass`, `SubtargetMarkerPass`, `StripDeviceFunctionBodiesPass`). |
| Include flags | The driver must thread Luthier-internal headers in. See "Include dir / define propagation" below. |

The IR plugin's behavior depends on `target-features` and `target-cpu`
attributes clang sets on every function in the TU — the
`SubtargetMarkerPass` reads them via `firstAttrDonor`. The driver does
not need to do anything special here; it just has to invoke clang with
the right flags and the plugin will see consistent attributes.

### Host-side compile (CMake-driven, unchanged)

The host TU compiles through CMake's HIP language, also via
`CMAKE_HIP_COMPILER`. Helper-set options:

| Flag | Purpose |
|---|---|
| `--cuda-host-only` | Skip device codegen. Emit only the `__hip_register*` glue, extern refs to `__hip_fatbin` / `__hip_gpubin_handle`. |
| `-fno-gpu-rdc` | Disable separable device compilation. |
| `-fuse-cuid=none` | Drop the per-TU `__hip_fatbin_<CUID>` / `__hip_gpubin_handle_<CUID>` suffixing so the symbols are the plain unsuffixed names the loader (and `LoadHIPFATBinaryInfoPass`) expect. Restricts the tool to one HIP TU's worth of fat binary. |
| `-Xclang -fcuda-include-gpubinary -Xclang ${_fatbin}` | Splice the bundle bytes into the host TU as `__hip_fatbin` (`[N x i8]`, section `.hip_fatbin`) plus the matching `__hip_fatbin_wrapper`. Critical: each `-Xclang` needs its own driver wrapper, and CMake's `SHELL:` prefix is mandatory or `-Xclang;-Xclang` collapses under de-dup. |
| `-fpass-plugin=${IRPlugin}` | Runs `LoadHIPFATBinaryInfoPass` on the host TU. This populates `HipFatBinaries`/`HipFunctions`/`HipDeviceVars` slots from the `__hipRegister*` call sites. Without it on the host side, the loader sees an empty `HipFatBinaries`. |
| `-fplugin=${CXXPlugin}` | Runs `LuthierToolCXXCompilationPlugin`. Host-only AST rewrites for `LUTHIER_HOOK_*` annotations: kernel-handle synthesis, `DeclRefExpr` rewrite. No-op for tools without hook annotations. |

The driver does **not** orchestrate the host compile. It only produces
the fat binary that CMake threads in via `-fcuda-include-gpubinary=`.

### Include dir / define propagation

The current helper extracts these at configure time from the
`LuthierTooling` target's properties:

- `INTERFACE_INCLUDE_DIRECTORIES` (plus `LuthierAMDGPU`'s, which carries
  the tablegen `.inc` directory). After unwrapping `$<BUILD_INTERFACE:…>`
  and dropping `$<INSTALL_INTERFACE:…>`:
  - `${LLVM_INCLUDE_DIRS}` — LLVM headers.
  - `${CMAKE_SOURCE_DIR}/include` — Luthier public headers.
  - `${CMAKE_BINARY_DIR}/include` — generated Luthier headers.
  - `${CMAKE_BINARY_DIR}/include/luthier/AMDGPU/` — generated AMDGPU
    header copies.
  - `${CMAKE_BINARY_DIR}/src/lib/AMDGPU/` — tablegen `.inc` outputs.
- `INTERFACE_COMPILE_DEFINITIONS`, filtered:
  - `AMD_INTERNAL_BUILD` — switches the bundled HSA headers to the
    direct-include form (`hsa_ext_*.h` rather than `inc/hsa_ext_*.h`).
  - The `LLVM_DEFINITIONS` blob is filtered out (pre-existing
    LuthierTooling CMakeLists bug: it stuffs the entire space-separated
    `LLVM_DEFINITIONS` string into one `INTERFACE_COMPILE_DEFINITIONS`
    list element).

The HIP runtime headers live at `${hip_INCLUDE_DIR}` (from
`find_package(hip)`) and are spliced in via `-idirafter` so user code's
own `-isystem` paths take precedence.

The driver receives these paths via argv (`--include-dir=`, `--define=`,
`--hip-include-dir=`) — CMake assembles the list once and passes it
through. The driver doesn't need to know they came from a CMake target.

### Bundle output shape

The final `.hipfb` is a Clang offload bundle:

- Magic `__CLANG_OFFLOAD_BUNDLE__`.
- One host slot at index 0 with target `host-${CMAKE_SYSTEM_PROCESSOR}-unknown-linux-gnu` and `/dev/null` body.
- One slot per AMDGCN slice with target
  `hipv4-amdgcn-amd-amdhsa--gfxN[:feat±[:feat±]]` (LLVM AMDGPU
  target-ID feature syntax — see "Bundle label format" below).

The loader's `DeviceToolCodeParser::parseOffloadBundle` reads this with
`llvm::object::OffloadBundleFatBin::create`, walks the entries, and
extracts each AMDGCN slice's bytes. Each slice carries the
`SubtargetMarkerPass`-emitted `__luthier_subtarget` ELF symbol that the
loader uses for dispatch-time wave/cumode matching.

### Subtarget marker symbol (in every emitted slice)

`SubtargetMarkerPass` (runs at `OptimizerLastEPCallback`, after
`CreateAndEmbedIModulePass`, before `StripDeviceFunctionBodiesPass`)
emits:

```
@__luthier_subtarget = protected addrspace(1) constant i32 <encoded>
section ".luthier.subtarget"
@llvm.used = appending addrspace(1) global [… @__luthier_subtarget …]
```

Encoding (`<encoded>`):
- bit 0 — set if the slice runs in wave64; clear for wave32.
- bit 1 — set if the slice runs in CU mode; clear for WGP mode.

CU mode bit semantics:
- gfx10+: from `+cumode` / `-cumode` in `target-features`. Absent means
  WGP (the per-arch default).
- gfx9-: always set (no WGP concept).

Wave bit comes from `+wavefrontsize64` / `+wavefrontsize32` in
`target-features` — these are always positive (one of the two is
present, never `-wavefrontsize64`).

The driver doesn't need to know any of this — it just has to compile
with the right flags. The marker pass and the loader handle the rest.

### Targets to link / depend on

The helper unconditionally adds these to the final `add_library(${target} MODULE …)`:

- `hip::host` (from `find_package(hip)`).
- `LuthierTooling` (the main lib).
- `LuthierToolIRCompilationPlugin` and `LuthierToolCXXCompilationPlugin`
  as build-order dependencies (so the plugin `.so`s exist before the
  examples reference them via `$<TARGET_FILE:…>`).
- `HIP_ARCHITECTURES` target property mirrors the ISA matrix the helper
  produced device code for (so the host TU's `--cuda-host-only` compile
  doesn't see a divergent arch set).

The driver doesn't touch these — they stay in `luthier_add_tool` as
CMake-side wiring after the `add_custom_command` invocation.

## Helper behaviors the driver gets to redesign

### ISA-matrix planning

Today: CMake reads `CMAKE_HIP_ARCHITECTURES`, loops, and asks the user
to supply `WAVE_MODES` if non-default. No xnack/sramecc handling at
all.

Driver: enumerates the full matrix automatically from
`llvm::AMDGPU::getArchAttrAMDGCN` per arch, applies the policy decisions
recorded in `ToolCompileDriver-Phase0-Inventory.md` ("Default policy per
feature" table — repeated below for convenience):

- `wavefrontsize64`, `cumode`: full matrix where variable; user can
  `--pin=<feat>±` to narrow.
- `xnack`, `sramecc`: default `any` (single slice, no flag passed to
  `--offload-arch=`, ELF `e_flags` get `_ANY_V4` state); user can
  `--vary=<feat>` to opt into the matrix.

### Bundle label format

LLVM AMDGPU target-ID feature syntax:
`hipv4-amdgcn-amd-amdhsa--gfxN[:feat±[:feat±]]`

Driver-side rule: the suffix names exactly the features the driver
explicitly pinned in this slice's codegen. Features that varied are in
the suffix; features that were left implicit (`any` for
xnack/sramecc, fixed-by-arch for wave on gfx9, etc.) are not.

This matches LLVM's own convention of only naming features that were
explicitly set on the codegen command line. The label is *minimal but
truthful*. External tooling (rocgdb, rocm-debug-agent,
`clang-offload-bundler --list`) recognizes the form because it's the
same one `--offload-arch=` accepts (modulo wave/cumode not being in
clang's target-ID feature whitelist — see Probe 1 below).

### Plumbing (subprocesses vs in-process)

Today: each per-slice compile is one `add_custom_command` with clang.
Each per-slice extract is another with `clang-offload-bundler --unbundle`.
The rebundle is a third. Build-system glue counts each as a job.

Driver phase 1: subprocess each call (`llvm::sys::ExecuteAndWait`).
Same compile commands, just orchestrated from C++ where dedup, parallel
fan-out, and validation are natural to add.

Driver phase 5 (optional): in-process bundle writer. LLVM exposes
`llvm::object::OffloadBundleFatBin::create` (read side) but the writer
is clang-internal. Either subprocess forever, vendor a minimal writer
(format is small and stable), or upstream a public API patch.

## Capability lookup: use LLVM, don't probe or bake

`llvm::AMDGPU::TargetParser` (`llvm/TargetParser/TargetParser.h`)
exposes the per-arch capability bitmask directly:

```cpp
namespace llvm::AMDGPU {
  enum ArchFeatureKind : uint32_t {
    FEATURE_WAVE32       = 1 << 6,  // arch can do wave32 (implies ± toggle)
    FEATURE_XNACK        = 1 << 7,  // xnack target-ID feature available
    FEATURE_SRAMECC      = 1 << 8,  // sramecc target-ID feature available
    FEATURE_WGP          = 1 << 9,  // WGP mode available (implies cumode ±)
    FEATURE_XNACK_ALWAYS = 1 << 10, // xnack is the default
    // (other bits omitted)
  };

  GPUKind parseArchAMDGCN(StringRef CPU);
  unsigned getArchAttrAMDGCN(GPUKind);
  void fillValidArchListAMDGCN(SmallVectorImpl<StringRef> &);
}
```

The driver links `LLVMTargetParser` and calls these directly. No probe
loop, no baked table. Verified per-arch values from
`llvm/lib/TargetParser/TargetParser.cpp`:

| Arch | Variable wave64 (`WAVE32`) | Variable cumode (`WGP`) | Variable xnack (`XNACK`) | Variable sramecc (`SRAMECC`) |
|---|:---:|:---:|:---:|:---:|
| gfx906, gfx908, gfx90a, gfx942, gfx950 | — | — | ✓ | ✓ |
| gfx1010 | ✓ | ✓ | ✓ | — |
| gfx1030 | ✓ | ✓ | — | — |
| gfx1100, gfx1150, gfx1200, gfx1201 | ✓ | ✓ | — | — |

So `gfx1201` defaults: 4 slices (wave ± × cumode ±). With
`--vary=xnack`: still 4 (gfx1201 doesn't expose xnack). With
`--pin=wavefrontsize64+`: 2 slices.

`gfx942` defaults: 1 slice (`hipv4-amdgcn-amd-amdhsa--gfx942`, with
`_ANY` xnack and sramecc in `e_flags`). With `--vary=xnack`: 2 slices.
With `--vary=xnack --vary=sramecc`: 4.

## Probes (evidence)

### Probe 1: feature-suffix in `--offload-arch=` composes with `-m{wave,cumode}` flags

Result: clang accepts the combination *when the feature is in the
arch's bitmask*. Composition succeeds on gfx906 (`xnack+`) and gfx942
(`xnack+:sramecc+`) with `-mwavefrontsize64`. Rejected on gfx1030
(`xnack+`) because `FEATURE_XNACK` is not set for gfx1030 — the error
message references the standard AMDGPU target-ID grammar.

Implication: the driver must validate every `--offload-arch=` it
constructs against `getArchAttrAMDGCN` before forking clang. We don't
trial-compile and recover from errors.

### Probe 2: bare `--offload-arch=gfxN` → `_ANY` ELF flags

Result on gfx906:

| `--offload-arch=` | `e_flags` | Decoded |
|---|---|---|
| `gfx906` (bare) | `0x52F` | mach=0x2F xnack=any sramecc=any |
| `gfx906:xnack+` | `0x72F` | xnack=on sramecc=any |
| `gfx906:xnack-` | `0x62F` | xnack=off sramecc=any |
| `gfx906:sramecc+:xnack+` | `0xF2F` | xnack=on sramecc=on |
| `gfx906:sramecc-:xnack-` | `0xA2F` | xnack=off sramecc=off |

Implication: the default-`any` opt-out for xnack/sramecc costs nothing —
just *don't* add the feature to `--offload-arch=`. The HSA loader treats
`_ANY` slices as compatible with either hardware state.

### Probe 3: `-mwavefrontsize64` / `-mcumode` → IR `target-features`

Result on gfx1030 (RDNA2, both features variable):

| Flags | `target-features` (relevant subset) |
|---|---|
| `--offload-arch=gfx1030` | `+wavefrontsize32` |
| `--offload-arch=gfx1030 -mwavefrontsize64` | `+wavefrontsize64` |
| `--offload-arch=gfx1030 -mwavefrontsize64 -mcumode` | `+wavefrontsize64,+cumode` |
| `--offload-arch=gfx1030 -mno-cumode` | `+wavefrontsize32,-cumode` |

Result on gfx906 (CDNA, wave fixed, no WGP):

| Flags | `target-features` (relevant subset) |
|---|---|
| `--offload-arch=gfx906` | `+wavefrontsize64` |
| `--offload-arch=gfx906 -mwavefrontsize64` | `+wavefrontsize64` (no-op) |
| `--offload-arch=gfx906 -mcumode` | `+wavefrontsize64,+cumode` (cosmetic — gfx9 always CU) |
| `--offload-arch=gfx906 -mno-cumode` | (compiles, but clang warns `-Woption-ignored`) |

Implication: the `SubtargetMarkerPass`'s reliance on the
`target-features` attribute is correct for both archs. On gfx9 the
attribute is forced by hardware (always `+wavefrontsize64`); the
pass's default-CU rule handles cumode-absence correctly.

The driver should *not* pass `-mwavefrontsize64` / `-mno-wavefrontsize64`
on archs where wave is fixed (gfx9-), and *not* pass `-mcumode` /
`-mno-cumode` on archs without WGP (gfx9-). It's harmless on
`-mwavefrontsize64` (no-op) but the `-mno-*` forms generate noise that
shows up in build logs. Save the noise; the capability table already
tells us when each flag is meaningful.

### Probe 4: bundle target-ID feature labels compose

Result: `clang-offload-bundler --targets=hipv4-amdgcn-amd-amdhsa--gfx1030:wavefrontsize64+:cumode+`
is accepted, roundtrips via `--unbundle`, shows correctly in `--list`.
Multiple slices with distinct feature tails coexist in one bundle (the
"Duplicate targets are not allowed" rule applies to *identical* target
strings).

Implication: the driver can emit labels that fully describe each
slice's subtarget features without compatibility tradeoffs against the
bundler. The wave/cumode features aren't in clang's target-ID
*whitelist* (so `--offload-arch=gfx1030:wavefrontsize64+` is rejected by
clang's arg parser), but the *bundler* treats target IDs opaquely and
accepts the same syntax.

## Driver contract (summary)

Inputs (argv):
- `--hip-clang=<path>` — required.
- `--plugin-ir=<path>` — required.
- `--plugin-cxx=<path>` — optional (only needed if a future host-side
  path lands; today the CXX plugin is host-only, driver doesn't use it).
- `--hip-include-dir=<path>` — required.
- `--include-dir=<path>` — repeatable.
- `--define=<name[=value]>` — repeatable.
- `--hip-std=<N>` — default 20.
- `--arch=<gfxN>` — repeatable; arches the user wants coverage for.
- `--vary=<xnack|sramecc>` — opt these into the matrix. Off by default.
- `--pin=<wavefrontsize64+/-|cumode+/->` — pin a normally-variable
  feature to a single value.
- `--output=<path>` — fatbin output.
- Positional: HIP source file(s).

Behavior:
1. Parse `--arch=`. Reject any arch not in `fillValidArchListAMDGCN`.
2. For each arch, look up capabilities via `getArchAttrAMDGCN`.
3. Apply the user's `--vary` / `--pin` policy on top of defaults.
4. Enumerate the resulting `ISASpec` set.
5. Per spec, fork clang with the appropriate flags (parallel fan-out).
6. Read each `.co`'s inner bundle, extract the AMDGCN slice.
7. Validate: decode each slice's `__luthier_subtarget` marker; cross-
   check against the `ISASpec`. Hash slices; dedupe byte-identical ones.
8. Construct each slice's outer bundle label per the format rules.
9. Write the final `.hipfb` via `clang-offload-bundler`.

Exit behavior: fail-loud on any per-slice compile failure, on any
marker mismatch, on any duplicate label.

## Open questions deferred to Phase 1+

1. **Caching.** Two tools targeting the same arch matrix produce
   identical per-slice ELFs from the same Luthier IR plugin. CMake
   doesn't cache across `add_custom_command` outputs. The driver could
   add a content-addressed cache (`~/.cache/luthier/slices/`) keyed on
   `(source_hash, isa_spec, plugin_revision, clang_revision)`. Defer to
   Phase 4+; subprocess-everything is fine for now.

2. **Cross-tool dedup.** Today each tool produces its own per-arch
   slices even when the source is identical. Cross-tool dedup would
   require a build-system-aware step (cmake doesn't naturally express
   "this artifact is shared across N targets"). Defer.

3. **In-process bundle writer.** LLVM exposes the reader; the writer is
   clang-internal. Phase 5+ decision.

4. **Multi-source tools.** The current helper compiles each `.hip`
   source both halves; the driver's V1 could either match that or pin
   to single-source (the simpler contract). Practice today is one source
   per tool; locking V1 to single-source is reasonable.

5. **Per-slice `target-cpu` / `target-features` cross-check.** The
   driver could also verify that the IR `target-features` it sees in
   each compiled slice's `.llvmbc` (extracted via `llvm-objcopy`) match
   the ISA spec. Belt-and-braces with the marker; defer to Phase 3.
