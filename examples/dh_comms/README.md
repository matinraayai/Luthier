# dh_comms — stream GPU memory accesses to the CPU

A Luthier tool that instruments every global-memory instruction of a kernel
and sends each access to the CPU *while the kernel runs*, using the protocol
of AMD's [dh_comms](../../../../ThreadTracing/dh_comms/explanation.md).

For every access the GPU sends **where it landed, not the address**: the
kernel-argument buffer and the element index inside it.

## How it works

```
  GPU (instrumented kernel)                       CPU (tool)
  ─────────────────────────                       ──────────
  hook before each global load/store              before dispatch:
    address → (arg slot, element index)             resolve kernel-arg buffers
    lock sub-buffer                                  (hsa_amd_pointer_info)
    full? → set flag, wait for CPU ───────┐          start drain thread
    write wave header + one record/lane   │
    publish size, unlock                  │       drain thread:
                                          └────►    flag==1 → parse, empty,
        fine-grained pinned host memory              clear flag
        (sub-buffers, sizes, flags)                after kernel: drain the rest,
                                                     print the report
```

The shared buffers are allocated the HSA way — CPU agent's fine-grained pool,
`memoryPoolAllocate`, `agentsAllowAccess` for the GPUs — which is what
`hipHostMalloc(…, hipHostMallocCoherent)` does underneath. A Luthier tool
cannot call HIP for this.

## Layout

| file | side | what |
|---|---|---|
| `include/dh_comms/Protocol.h` | both | wire format: `WaveHeader`, `AccessRecord`, `DeviceDescriptor` (no bitfields: GCC builds the host side, clang the device side) |
| `include/dh_comms/DeviceSubmit.h` | GPU | `submitMemoryIndex`: port of dh_comms' `generic_submit_message` |
| `include/dh_comms/SharedBuffers.h` | CPU | pinned host buffers via HSA |
| `include/dh_comms/MessageProcessor.h` | CPU | drain thread: port of dh_comms' `processing_loop` |
| `include/dh_comms/AccessIndexHandler.h` | CPU | per-instruction index summary — **write your analyses as another `MessageHandler`** |
| `src/DhCommsTool.hip` | both | the tool: hooks, what gets instrumented, per-dispatch setup |

## Run

```bash
LD_PRELOAD=<build>/lib/libLuthierTooling.so:<build>/examples/dh_comms/libLuthierDhComms.so ./app
```

Options go in `LUTHIER_ARGS`:

| option | default | |
|---|---|---|
| `--dh-comms-sub-buffers` | 256 | power of two |
| `--dh-comms-sub-buffer-bytes` | 65536 | |
| `--dh-comms-element-size` | 4 | index = byte offset / element size |
| `--dh-comms-row-length`, `--dh-comms-num-rows` | 1024 | for the (row, col) range check |
| `--kernel-begin-interval`, `--kernel-end-interval` | all | which dispatches |
| `--instr-begin-interval`, `--instr-end-interval` | all | which instrumentation points (ids in the report) |
| `--dh-comms-noop-hooks` | off | diagnostic: same arguments, no protocol |
| `--dh-comms-dump-object=<path>` | — | diagnostic: write each instrumented code object |

## Things that differ from dh_comms, and why

* **Spin loops are wave-uniform.** In a Luthier payload a spin loop run by a
  divergent subset of lanes deadlocks; dh_comms' `if (lane == 0) while (…)` is
  exactly that shape. Here all active lanes stay in the loop and it exits on a
  `readfirstlane`-broadcast condition; only the atomic is guarded.
* **Sub-buffer choice hashes the access address, not the workgroup id.**
  Reading the workgroup id in a hook needs `luthier::readSVA`, which this
  Luthier build left unlowered in hook code (the placeholder reached the
  assembler as `luthier::readSVA`). The lock makes any wave-uniform choice
  correct. `WaveHeader::BlockIdx*` are therefore `0xffff`.
* **No shader-engine / CU ids.** dh_comms reads them with `s_getreg` inline
  asm; Luthier rejects inline asm in a hook.
* **Locks and flags are 32-bit** (native atomics) instead of bytes.

## Required Luthier fix

Payloads that clobber `VCC` corrupted the application: Luthier's
`InjectedPayloadPreserveLiveRegsPass` restored `VCC` on payload exit but the
matching save was deleted, so `VCC` came back as garbage. On the `mmm` SGEMM
test the later C stores — guarded by `s_and_b64 …, vcc` — ran with 17–22 of 64
lanes and the result was wrong. The fix (in `src/lib/ToolCodeGen/
InjectedPayloadPreserveLiveRegsPass.cpp`) gives `VCC` its dummy def per half
(`$vcc_lo`, `$vcc_hi`) so the 64-bit save survives `ProcessImplicitDefs`.

## Building

The device slices compile a copy of `DhCommsTool.hip` without a dependency
file, so `CMakeLists.txt` lists the headers explicitly; otherwise a header edit
leaves stale hook code inside the tool.
