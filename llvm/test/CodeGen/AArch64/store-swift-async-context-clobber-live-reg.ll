; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 4
; RUN: llc -o - -mtriple=arm64e-apple-macosx %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

define swifttailcc void @test_async_with_jumptable(ptr %src, ptr swiftasync %as) #0 {
; CHECK-LABEL: test_async_with_jumptable:
; CHECK:       ; %bb.0: ; %entry
; CHECK-NEXT:    orr x29, x29, #0x1000000000000000
; CHECK-NEXT:    str x19, [sp, #-32]! ; 8-byte Folded Spill
; CHECK-NEXT:    stp x29, x30, [sp, #16] ; 16-byte Folded Spill
; CHECK-NEXT:    add x16, sp, #8
; CHECK-NEXT:    movk x16, #49946, lsl #48
; CHECK-NEXT:    mov x17, x22
; CHECK-NEXT:    pacdb x17, x16
; CHECK-NEXT:    str x17, [sp, #8]
; CHECK-NEXT:    add x29, sp, #16
; CHECK-NEXT:    .cfi_def_cfa w29, 16
; CHECK-NEXT:    .cfi_offset w30, -8
; CHECK-NEXT:    .cfi_offset w29, -16
; CHECK-NEXT:    .cfi_offset w19, -32
; CHECK-NEXT:    mov x20, x22
; CHECK-NEXT:    ldr x8, [x0]
; CHECK-NEXT:    cmp x8, #2
; CHECK-NEXT:    csel x9, xzr, x0, eq
; CHECK-NEXT:    cmp x8, #0
; CHECK-NEXT:    csel x10, x22, xzr, eq
; CHECK-NEXT:    cmp x8, #1
; CHECK-NEXT:    csel x19, x9, x10, gt
; CHECK-NEXT:    mov x22, x0
; CHECK-NEXT:    bl _foo
; CHECK-NEXT:    mov x2, x0
; CHECK-NEXT:    mov x0, x19
; CHECK-NEXT:    mov x1, x20
; CHECK-NEXT:    ldp x29, x30, [sp, #16] ; 16-byte Folded Reload
; CHECK-NEXT:    ldr x19, [sp], #32 ; 8-byte Folded Reload
; CHECK-NEXT:    and x29, x29, #0xefffffffffffffff
; CHECK-NEXT:    br x2
entry:
  %l = load i64, ptr %src, align 8
  switch i64 %l, label %dead [
    i64 0, label %exit
    i64 1, label %then.1
    i64 2, label %then.2
    i64 3, label %then.3
  ]

then.1:
  br label %exit

then.2:
  br label %exit

then.3:
  br label %exit

dead:                                                ; preds = %entryresume.5
  unreachable

exit:
  %p = phi ptr [ %src, %then.3 ], [ null, %then.2 ], [ %as, %entry ], [ null, %then.1 ]
  %r = call i64 @foo()
  %fn = inttoptr i64 %r to ptr
  musttail call swifttailcc void %fn(ptr swiftasync %src, ptr %p, ptr %as)
  ret void
}

declare i64 @foo()

attributes #0 = { "frame-pointer"="non-leaf" }