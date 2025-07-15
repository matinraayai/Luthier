# Luthier's Code Lifting Process

Luthier's code lifting process involves inspecting the contents of a code object and
retrieving an equivalent [LLVM Machine IR (MIR)](https://llvm.org/docs/MIRLangRef.html)
representation. We consider the recovered MIR to be equivalent to the inspected binary
if running the recovered MIR through the assembly printer pass pipeline of the
target machine will produce the "same" binary. Therefore, we can consider the
lifting process to essentially be a "Reverse Assembly Printer".
The recovered MIR representation is also referred to as a "Lifted Representation".

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
- Frameworks focused on translation/emulation
  (e.g., [RPCX3](https://rpcs3.net/), [MCTOLL](https://github.com/microsoft/llvm-mctoll))
  find it useful to lift the translated binary into LLVM IR, which then can be compiled to the
  target CPU's architecture. Lifting to LLVM IR is generally undesirable for binary instrumentation,
  as the process can change the original instructions of the instrumented binary. Luthier's instrumentation
  avoids this adverse sideeffect by only lifting the inspected code to LLVM MIR and staying as close
  to the binary-level as possible.
