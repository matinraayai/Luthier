# Machine Instruction Annotation

## Background

When running Luthier instrumentation passes we may want to append additional
information to a set of `llvm::MachineInstr`s in the target
`llvm::MachineModuleInfo` (MMI). Some example reasons include:

- Keeping track of the injected payload functions that will eventually be
  patched before or after a target MMI instruction.
- Keeping track of manually injected instructions in the target MMI so that
  we don't modify them after patching is done.

A naive solution is to keep a map between pointers of machine instruction to
whatever information we want to maintain in an analysis pass. However, there
are two issues with this approach:

1. Internally, the `llvm::MachineBasicBlock` stores on to its machine
   instructions in an intrusive linked list and has a list element recycler
   for that re-uses removed elements' memory. In between target module passes,
   there is a chance the element recycler is invoked to re-arrange the
   instructions (especially after target MMI is patched) rendering our pointer
   maps stale.
2. It is hard to serialize the state of the Luthier instrumentation pipeline as
   machine instruction pointer locations can change between runs. Serialization
   is very useful for testing and debugging.

To keep track of machine instructions, we instead have to rely on appending
extra information to the machine instruction itself to guarantee the information
doesn't get lost while the machine instructions are transformed in between
passes. Luckily, there is an `Info` field in machine instructions designed for
[this specific purpose](https://reviews.llvm.org/D50701). There are a limited
set of extra information that can be appended to this field. Luthier opts to
define a custom formatted version of
the [PC Sections Metadata](https://llvm.org/docs/PCSectionsMetadata.html) for
the following reasons:

1. PC Sections has been designed for use for instrumentation, and has a flexible
   formatting that can be expanded.
2. PC Sections gets emitted by the assembly printer. This means that combined
   with a custom `llvm::AsmPrinterHandler`, Luthier can emit verbose comments
   regarding Luthier-specific information in the `.s` version of the
   final instrumented code object.
3. PC Sections seem to be preserved in LLVM MIR passes (despite this not being
   guaranteed in the IR passes).
4. The extra info field does not seem to have an adverse side effect on other
   LLVM MIR passes in the AMDGPU backend.

There is one limitation for using the `PCSections` in the machine instructions:
**It is not yet serialized by the MIR parser**, meaning that we either have
to wait until it's serialized or implement a workaround ourselves.

The following avenues for appending extra information to a machine
instructions were investigated but abandoned:

1. Using the debug instruction number attached to machine instructions for
   tracking the register values they have defined after register allocation.
   This seems like a good candidate, as it also has serialization support by the
   LLVM MIR parser. However, it has the potential to interfere with debug info
   and can be dropped by different passes if the instruction has been modified
   such that it doesn't define its original output registers anymore. It is
   better to leave this field for representing lifted DWARF information instead.
2. Attaching additional `llvm::MDNode` operands at the end of the explicit
   operand list of the instruction. Unfortunately, this causes the metadata
   operands to also included in the implicit operand list, contradicting
   the assumption that all implicit operands are registers.
   More specifically, all operands in a machine instruction (regardless of them
   being explicit or implicit) are stored in the same dynamically allocated
   array. The beginning index of the implicit operands in the operand
   list is calculated using the
   [
   `getNumExplicitOperands`](https://github.com/llvm/llvm-project/blob/8518d2c4057d9aa4249b8466a4d77771e4f1bf4f/llvm/lib/CodeGen/MachineInstr.cpp#L838-L854)
   method. If the machine instruction's explicit operands are well-formed,
   then this logic causes the `implicit_operands()` iterator of the machine
   instruction to also return metadata operands.

## Luthier PC Sections Metadata Format

According to the
[PC Sections Metadata Documentation](https://llvm.org/docs/PCSectionsMetadata.html)
and the assembly printer method `emitPCSections`, the Metadata must have the
following format:

```llvm
!0 = !{
  !"<section#1>"
  [ , !1 ... ]
  !"<section#2">
  [ , !2 ... ]
  ...
}
!1 = !{ iXX <aux-consts#1>, ... }
!2 = !{ iXX <aux-consts#2>, ... }
...
```

Essentially, an `MDString` header metadata is followed by an array of constant
value metadata nodes. PCSections Metadata can be created using the
`llvm::MDBuilder::createPCSections` method. We use every even-numbered
entry in the `MDNode` array to introduce a header, with every odd-numbered
entry introduce the contents for the header.

### Instruction Identifier

The header is `"luthier.instr.id"`, and it is followed by a single element array
of constant `uint64_t` number. Provides a unique identifier for each instruction
that is preserved between passes. The module will have a metadata named
`"luthier.next.instr.id"` that indicates the next ID available for any
instruction requiring an ID.

### Trace Instruction Identifier

The header is `"luthier.instr.trace"`, and it is followed by a single element
array of constant `uint64_t` number. Indicates to passes that the instruction
was obtained by inspecting and lifting loaded memory, and the constant number
indicates the memory address it was obtained from.

### Mutable Instruction

The header is `"luthier.instr.mutable"`, followed by an empty list of constants.
It indicates (especially to post-patching passes) that an instruction is free
to be modified or replaced while ensuring the original behavior of the
instruction. For example, a short mutable branch in the post-patching passes
indicates it can be relaxed (i.e., converted to a long branch) in case it cannot
reach its original target. By default, all instructions are treated as immutable
unless stated otherwise.

## Helpful Links

- [PC Sections Metadata Documentation](https://llvm.org/docs/PCSectionsMetadata.html)

