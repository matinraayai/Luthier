// The test kernel: store one value to one address.
//
// Deliberately tiny, because it is a measuring instrument rather than a
// workload. Every dispatch in the suite writes a distinct value to its own
// address, so a dropped, duplicated or reordered packet is visible in memory
// instead of being averaged away.
//
// ARGUMENTS
//   AQL passes a *pointer* to the argument block in s[0:1] (that is what
//   ENABLE_SGPR_KERNARG_SEGMENT_PTR in the kernel descriptor asks for), unlike
//   the PM4 path which preloads the argument values themselves into registers.
//   So the kernel loads its arguments from memory first.
//
//     struct { uint64_t Destination; uint32_t Value; uint32_t Pad; }
//
// TWO DELIBERATE CHOICES
//   * global_* rather than flat_* addressing. Flat addressing needs the
//     group and private aperture fields of the queue descriptor set up
//     correctly; global addressing needs nothing.
//   * No cache modifiers on the store. They are spelled differently on
//     different GPU generations (glc/slc, then sc0/sc1/nt), which would make
//     this file architecture-specific. Leaving them off also makes the test
//     meaningful: whether the host can see this write then depends on the
//     packet's release fence, which is exactly what we want to be able to check.

.text
.globl luthier_test_store_dword
.p2align 8
.type luthier_test_store_dword,@function

luthier_test_store_dword:
    // s[0:1] holds the address of the argument block.
    s_load_dwordx4 s[4:7], s[0:1], 0x0
    s_waitcnt lgkmcnt(0)

    // s[4:5] = destination address, s6 = value to store.
    v_mov_b32 v0, s4
    v_mov_b32 v1, s5
    v_mov_b32 v2, s6

    global_store_dword v[0:1], v2, off
    s_waitcnt vmcnt(0)
    s_endpgm
