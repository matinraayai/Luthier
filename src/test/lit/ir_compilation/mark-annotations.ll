; RUN: opt -load-pass-plugin=%luthier_tool_ir_compilation_plugin_path -passes=luthier-mark-annotations -S %s | %tee_out FileCheck %s
; Verifies that:
;   - functions annotated with luthier.intrinsic get the matching fn-attr
;   - llvm.global.annotations / llvm.used / llvm.compiler.used are removed

target triple = "amdgcn-amd-amdhsa"

@.str.intr = private unnamed_addr constant [18 x i8] c"luthier.intrinsic\00", section "llvm.metadata"
@.str.file = private unnamed_addr constant [4 x i8] c"f.c\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @my_intrinsic, ptr @.str.intr, ptr @.str.file, i32 2, ptr null }
], section "llvm.metadata"

@llvm.used = appending global [1 x ptr] [ptr @my_intrinsic], section "llvm.metadata"

declare i32 @my_intrinsic()

; CHECK-NOT: @llvm.global.annotations

; CHECK: declare i32 @my_intrinsic() #[[INTR:[0-9]+]]

; CHECK-DAG: attributes #[[INTR]] = {{.*}}"luthier.intrinsic"
