# Luthier's Code Lifting Process

Luthier's code lifting process involves inspecting the contents of a code object to retrieve an
**equivalent** and **relocatable** [LLVM Machine IR (MIR)](https://llvm.org/docs/MIRLangRef.html) of the code object.
In other words, the lifting process is essentially a
"Reverse [LLVM Assembly Printer](https://llvm.org/doxygen/classllvm_1_1AsmPrinter.html) Pass". We refer to the recovered
MIR representation in the lifting process as a "Lifted Representation". The lifted representation is considered to be 
equivalent to the inspected code object and relocatable if running the recovered MIR through the assembly printer pass 
pipeline will produce a new binary that:

- Contains all the original instructions in the same order and layout as the original; and
- Can be loaded anywhere else on the same device in such a way that executing it instead of the
  original binary is guaranteed to produce the same output. To satisfy this condition, minor
  modifications of operands of the original instructions is allowed.


----




If a direct PC manipulation instruction (i.e., `S_SETPC_B64` or `S_SWAPPC_B64`) is encountered, we perform some 
analysis 

As most of the analysis in LLVM is done in the SSA form of the code, and some work best with SSA, 

It is possible that pc manipulation instructions 


The lifting process is implemented as a series of machine function/module passes in the new
LLVM pass manager. In the following sections we go through each pass in the lifting process and explain what they
do in more detail.

## 1. Populate Symbols Pass
The very first step in the lifting process is to lift the symbols present inside the code object. In LLVM, the 
symbol bindings of the final binary reside in the `llvm::Module` itself.



Here we see if the code object contains relocation information

During the symbol lifting stage, we inspect both the `symtab` and `dynsym` sections of the code object. 
Ideally, we want access to the `symtab` section because we can then easily identify static variable or device functions
inside the code. Identifying the static variables in addition to the relocation information ensures that we can replace 
any address calculation to access the static variable can be replaced with its `GOT` variant. That way, we can then
make the HSA loader point the lifted code to the correct location of the variable without us having to do much work.

If a `symtab` section is not present inside the code, `dynsym` will only tell us about the kernel functions inside


## 2. Populate Functions Pass
Because ultimately our goal is to instrument kernels, we only populate device functions that are reachable from the
kernel entry point 

1. We first have to do the following:
    - The binary has relocation information: This is the most ideal scenario. If the binary leaves
      the relocation information intact, and if the binary contains device function calls or
      accesses to static variables, we are able to detect them and add them as operands. When
      `setpc` or a `swappc` instruction is encountered, we can trace the value used to calculate
      the register containing the destination address to some extent. This doesn't mean we will
      always succeed; But it will avoid some last-resort scenarios we have to make to ensure
      the final binary is relocatable. Also references to a static variable can be instead
      replaced with its GOT variant.
    - The binary has no relocation information, but contains symbols: In this scenario,
      static variable references might be detectable and optimized via
    - The binary is stripped: In this case, if the

One of the objectives of the lifting process is to provide the user with a "higher-level" view

To recover a lifted representation from the

We generally avoid instrumentation at the code object level, and only perform instrumentation at
the entry kernel level for performance reasons. A single code object might easily include 100+
kernels, and some of them might not even. Though as it can be seen fit, the user themselves can
implement lifting passes that recovers the
Inspection: we want to be able to inspect the code's control-flow graph (CFG), its instructions.
Generally, we are able to. Even if we're not able to infer the basic blocks and

Instrumentation:

The reason behind lifting the inspected binary to LLVM MIR is as follows:

1. LLVM MIR is essentially a **high-level and flexible ELF editor**. When prototyping
   Luthier, after it became apparent that duplicating the code objects was the way forward
   for AMD GPU instrumentation, we looked for a way to manipulate ELFs in order to
   inject instrumentation logic. We first investigated frameworks like
   [ELFIO](https://github.com/serge1/ELFIO) to modify an AMD GPU ELF; But we soon realized
   that ELFIO was not meeting our needs; Not only we need to append instrumentation logic in
   the text section, but we also might need to remove unused function/variable symbols,
   or make them external so that we can link them with the originally loaded binary.
   Even though ELFIO provided these functionalities to some extent, it was very conservative
   and wasteful in its removal; For example, it zeroed out removed functions to ensure the
   original binary would stay intact, and it would make unnecessary copies both during parsing
   and patching of the binary.

   In contrast to ELFIO, if we manage to recover the LLVM MIR of the binary, we can freely use it
   to insert/remove instructions and symbols from the ELF/lifted representation using LLVM APIs,
   making it ideal for our instrumentation process.

2. LLVM MIR makes analyzing the binary **convenient**, as the entirety of the LLVM AMDGPU backend
   as well as LLVM's target independent code generation (CodeGen) infrastructure is at our disposal
   once the lifted representation is obtained.

Luthier's lifting process is not entirely novel, and it has its similarities and differences when
compared to the means of analysis other binary-based frameworks employ:

- [LLVM Bolt](https://github.com/llvm/llvm-project/blob/main/bolt/README.md) chooses not to
  lift the analyzed binary all the way up to LLVM MIR, and roles out its own
  [Bolt IR](https://llvm.org/devmtg/2023-05/slides/Tutorial-May10/02-Ayupov-BOLTPass.pdf) on top
  of LLVM MC instead.
  According to one of LLVM Bolt's developers
  on [LLVM's Discord](https://discord.com/channels/636084430946959380/930647188944613406/1218367771532988436),
  this was because raising to LLVM MC was enough for them to do code layout optimization.
  We decided to instead use Machine IR, which saved us a significant amount of development time,
  and proved to be an easier choice for patching instrumentation code. In the future
  as more use cases are identified for Luthier, changing the IR that Luthier performs its inspections
  might become justified.
- Frameworks focused on binary translation/emulation
  (e.g., [RPCX3](https://rpcs3.net/), [MCTOLL](https://github.com/microsoft/llvm-mctoll))
  find it useful to lift the translated binary into LLVM IR, and then compile it to the
  target CPU's architecture. Lifting to LLVM IR is generally undesirable for binary instrumentation,
  as the process can change or even remove the original instructions of the instrumented binary. Luthier's
  instrumentation avoids this adverse sideeffect by only lifting the inspected code to LLVM MIR and staying as close
  to the binary-level as possible.

During the lifting process, we have an entry point kernel

The lifting approach and directly modifying the lifted representation to obtain a stand-alone

Here are some

### Worst Case Scenario: No relocation Information

In this scenario, we https://qbdi.quarkslab.com/

## Lifting Passes


Originally, a singleton class called the `CodeLifter` was in charge of creating lifted representations. 

In the future, if any caching mechanism was to be implemented for the code lifting stage, it should be done at the 
object level, as code objects are easier to hash compared to symbols.