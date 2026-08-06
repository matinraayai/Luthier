; RUN: opt -load-pass-plugin=%luthier_tool_ir_compilation_plugin_path -passes="luthier-tool-device-code-offload-parser-pass" %s -S | %tee_out FileCheck %s

; CHECK: define i32 @add_numbers(i32 %0, i32 %1) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = add i32 %0, %1
; CHECK-NEXT:   ret i32 %2
; CHECK-NEXT: }

define i32 @add_numbers(i32 %0, i32 %1) {
entry:
  %2 = add i32 %0, %1
  ret i32 %2
}