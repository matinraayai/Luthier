# Instruction Semantic Fuzzer

## What the fuzzer does

For each representative instruction it builds a *reference kernel* — a
hand-assembled `MachineFunction` that seeds the operands from kernarg, runs the
instruction under test, and writes the results to an output buffer — and then
runs two tiers:

- **Tier 1** — build, run the machine verifier, emit a loadable code object.
- **Tier 2** — dispatch the reference kernel on the GPU, then raise the same
  `MachineFunction` back to LLVM IR through `luthier::TraceFunctionTranslator`,
  recompile it, dispatch that, and compare the outputs.

A tier-2 mismatch means the reference (real hardware) and the translated kernel
disagree, i.e. **a translator or instruction-semantics bug**. Failing cases are
written to `luthier-fuzzer-dumps/<OPCODE>.txt` (override with
`LUTHIER_FUZZER_DUMP_DIR`) containing the reference MIR, the lifted IR, and the
per-output values.

## Running it

```sh
ctest -R RefPath                 # the whole reference/translation suite
LUTHIER_WAVE=32 ctest -R RefPath # force wave32 (RDNA supports both)
```

### Groups at 100% (Tested on RDNA and CDNA GPUs)

100 tests passed from VOP1, VOP2, VOP3, VOP3P, VOPC, SOP (incl. wide/64-bit), GLOBAL (offset, vaddr,
sub-dword, cmpswap), FLAT, SCRATCH (incl. sub-dword), wide-mem, DS, DS-permute.

## CDNA Failed Instructions

1. OFFEN addressing (BUFFER_{LOAD,STORE}_DWORD_OFFEN, BUFFER_ATOMIC_ADD_OFFEN_RTN): the per-lane vaddr goes into the struct.buffer vindex arg instead of voffset — struct.buffer.
store.i32(data, rsrc, %tid<<2, i32 0, …). With the raw V# (stride 0), vindex*stride = 0, so all 64 lanes collapse to element 0 and race. (Plain OFFSET-dword passes because it has no vaddr.)

2. MTBUF unmodeled (TBUFFER_LOAD_FORMAT_X_*): the load is dropped — its dest becomes %x = freeze i32 poison, so output is garbage.


3. S_BUFFER_LOAD_DWORD_SGPR host segfault — gdb/stack-trace root cause
Used LLVM's in-process PrintStackTraceOnErrorSignal, which prints a symbolized backtrace under ctest:

```cpp
#4 MachineOperand::getType()                          ← SIGSEGV (null ref)
#5 getOperandAsValue(MachineOperand const&, Type*)
#6 getOperandAsValue(MachineInstr const&, OpName, Type*)
#7 raiseMachineInstr<S_BUFFER_LOAD_DWORD_SGPR>
#8 translate()
```

## RDNA Failed Instructions

1. `S_BUFFER_LOAD` tier-2 that cannot be scored: the
translator segfaults partway through that group (see below), which kills the
process. Excluding that one test is what makes the numbers above measurable in a
single run:

2. Instructions that are wrong

- BUFFER Loads
```
BUFFER_LOAD_UBYTE_OFFSET
BUFFER_LOAD_USHORT_OFFSET
BUFFER_STORE_BYTE_OFFSET
BUFFER_LOAD_DWORD_OFFEN
BUFFER_STORE_DWORD_OFFEN
BUFFER_ATOMIC_ADD_OFFEN_RTN
TBUFFER_LOAD_FORMAT_X_OFFEN
```

- Scalar memory (SMEM) — wide loads

```
S_LOAD_DWORDX8_IMM          consistent
S_LOAD_DWORDX2_IMM          intermittent
S_LOAD_DWORDX4_IMM          intermittent
S_BUFFER_LOAD_DWORDX4_IMM   consistent
```

The pattern is multi-dword scalar loads. The lifted IR is structurally correct
(it extracts elements 0..N-1 from the loaded vector and stores each), so this is
a value-level bug rather than a missing semantic.

- Translator crash

```
S_BUFFER_LOAD_DWORD_SGPR
```

`TraceFunctionTranslator` segfaults while raising this instruction — the dump is
left with `STATUS : TRANSLATING` and an empty lifted-IR section, which is the
report's way of saying the process never got past the translator. This aborts
the whole test binary, so it must be filtered out to score the rest of the suite.