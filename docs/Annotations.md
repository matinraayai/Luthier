# Annotations and Metadata in Luthier's Code Generation Process

Luthier uses LLVM IR annotations and metadata to append additional information
to both target and instrumentation modules. This information is then used to
guide Luthier's instrumentation and target module passes. In some cases, the
annotations will even propagate into the final printed assembly,
and is used by Luthier's loader for the target runtime.

In this section we go over the format of the annotations and metadata used in
Luthier and their purpose.

# Function Annotations and Metadata

## `"luthier.function.hook"` Annotation

Used in instrumentation module. Indicates that the annotated function is a hook
i.e. instrumentation function.

## `"luthier.function.injected_payload"`

Used in instrumentation module. Indicates that the function is an "injected
payload" i.e. its contents will be patched before instruction(s) in the
target module.

## `"luthier.function.entrypoint_addr` Metadata

Used in target modules. Indicates that the annotated function was obtained
via lifting the device code located at the given entry point. If the calling
convention of the annotated function is of AMDGPU kernel type it indicates the
device memory address of its kernel descriptor; Otherwise, it indicates its
starting device address entry point.

# Machine Instruction Annotation

## Background

When running Luthier instrumentation passes we may want to append additional
information to a set of `llvm::MachineInstr`s in the target application. Some
example reasons include:

- Keeping track of the injected payload functions that will eventually be
  patched before or after a target machine instruction.
- Keeping track of branches that can be relaxed after patching is done.

A naive solution is to maintain a map between pointers of machine instruction to
whatever information we want to maintain in an analysis pass. However, there
are two issues with this approach:

1. Internally, the `llvm::MachineBasicBlock` stores its machine
   instructions in an intrusive linked list and has a list element recycler
   that re-uses removed elements' memory. In between target module passes,
   there is a chance the element recycler is invoked to re-arrange the
   instructions (especially after machine functions are patched) rendering our
   pointer maps stale.
2. It is hard to serialize the state of the Luthier instrumentation pipeline as
   machine instruction pointer locations can change between runs. Serialization
   is very useful for testing and debugging.

To get around these issues we instead have to rely on appending extra
information to the machine instruction itself to guarantee the information
doesn't get lost while the machine instructions are transformed in between
passes. Luckily, there is an `Info` field in machine instructions designed for
[this specific purpose](https://reviews.llvm.org/D50701). There are a limited
set of extra information that
can be appended to this field. Luthier opts to define a custom formatted version
of the [PC Sections Metadata](https://llvm.org/docs/PCSectionsMetadata.html) for
the following reasons:

1. PC Sections has been designed for use for instrumentation, and has a flexible
   formatting that can be expanded in the future.
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
instructions were investigated in the past but abandoned in favor of the
`PCSections` field:

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

## Luthier Target Module PC Sections Metadata Format

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

The following subsections define the entries present in the PC sections of
a target module in Luthier.

### Trace Instruction Identifier

This section's header is `"luthier.machine_instr.trace_addr"`, followed by an
optional constant `uint64_t` element. Presence of the constant `uint64_t` number
indicates that the machine instruction was obtained by lifting a trace
instruction. The constant number indicates the trace instruction's device memory
address. The absence of the constant number indicates that the machine
instruction is not a trace instruction i.e., it was manually inserted and does
not belong to the original application.

### "Can Relax Branch"

This section's header is `luthier.machine_instr.can_relax_direct_branch`,
followed by a constant boolean element. It only applies to direct branch
instructions (both conditional and unconditional). If set to `true`,
post patching passes are free to relax the branch machine instruction in
case it cannot make its target. If set to `false`, the post patching passes
instead will insert a second branch to be the target of the machine instruction
in question.

###              

`"luthier.instr.indirect_jmp_call_targets"`, followed by the list of functions.
A value of `undef` at the end of the list means there are unresolved call
targets.

## Helpful Links

- [PC Sections Metadata Documentation](https://llvm.org/docs/PCSectionsMetadata.html)

