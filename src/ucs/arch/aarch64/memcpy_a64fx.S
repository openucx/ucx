# code taken from https://public-inbox.org/libc-alpha/20210512092809.901182-1-naohirot@fujitsu.com/
# using temporarily for benchmarking
# TODO request Fujitsu to contribute it

# define MEMCPY __memcpy_a64fx

/* Assumptions:
 *
 * ARMv8.2-a, AArch64, unaligned accesses, sve
 *
 */

#define L(name)         .L##name
#define L2_SIZE         (8*1024*1024)/2 // L2 8MB/2
#define CACHE_LINE_SIZE 256
#define ZF_DIST         (CACHE_LINE_SIZE * 21)  // Zerofill distance
#define dest            x0
#define src             x1
#define n               x2      // size
#define tmp1            x3
#define tmp2            x4
#define tmp3            x5
#define rest            x6
#define dest_ptr        x7
#define src_ptr         x8
#define vector_length   x9
#define cl_remainder    x10     // CACHE_LINE_SIZE remainder

    .arch armv8.2-a+sve

    .macro dc_zva times
    dc          zva, tmp1
    add         tmp1, tmp1, CACHE_LINE_SIZE
    .if \times-1
    dc_zva "(\times-1)"
    .endif
    .endm

    .macro ld1b_unroll8
    ld1b        z0.b, p0/z, [src_ptr, #0, mul vl]
    ld1b        z1.b, p0/z, [src_ptr, #1, mul vl]
    ld1b        z2.b, p0/z, [src_ptr, #2, mul vl]
    ld1b        z3.b, p0/z, [src_ptr, #3, mul vl]
    ld1b        z4.b, p0/z, [src_ptr, #4, mul vl]
    ld1b        z5.b, p0/z, [src_ptr, #5, mul vl]
    ld1b        z6.b, p0/z, [src_ptr, #6, mul vl]
    ld1b        z7.b, p0/z, [src_ptr, #7, mul vl]
    .endm

    .macro stld1b_unroll4a
    st1b        z0.b, p0,   [dest_ptr, #0, mul vl]
    st1b        z1.b, p0,   [dest_ptr, #1, mul vl]
    ld1b        z0.b, p0/z, [src_ptr,  #0, mul vl]
    ld1b        z1.b, p0/z, [src_ptr,  #1, mul vl]
    st1b        z2.b, p0,   [dest_ptr, #2, mul vl]
    st1b        z3.b, p0,   [dest_ptr, #3, mul vl]
    ld1b        z2.b, p0/z, [src_ptr,  #2, mul vl]
    ld1b        z3.b, p0/z, [src_ptr,  #3, mul vl]
    .endm

    .macro stld1b_unroll4b
    st1b        z4.b, p0,   [dest_ptr, #4, mul vl]
    st1b        z5.b, p0,   [dest_ptr, #5, mul vl]
    ld1b        z4.b, p0/z, [src_ptr,  #4, mul vl]
    ld1b        z5.b, p0/z, [src_ptr,  #5, mul vl]
    st1b        z6.b, p0,   [dest_ptr, #6, mul vl]
    st1b        z7.b, p0,   [dest_ptr, #7, mul vl]
    ld1b        z6.b, p0/z, [src_ptr,  #6, mul vl]
    ld1b        z7.b, p0/z, [src_ptr,  #7, mul vl]
    .endm

    .macro stld1b_unroll8
    stld1b_unroll4a
    stld1b_unroll4b
    .endm

    .macro st1b_unroll8
    st1b        z0.b, p0,   [dest_ptr, #0, mul vl]
    st1b        z1.b, p0,   [dest_ptr, #1, mul vl]
    st1b        z2.b, p0,   [dest_ptr, #2, mul vl]
    st1b        z3.b, p0,   [dest_ptr, #3, mul vl]
    st1b        z4.b, p0,   [dest_ptr, #4, mul vl]
    st1b        z5.b, p0,   [dest_ptr, #5, mul vl]
    st1b        z6.b, p0,   [dest_ptr, #6, mul vl]
    st1b        z7.b, p0,   [dest_ptr, #7, mul vl]
    .endm

    .macro shortcut_for_small_size exit
    // if rest <= vector_length * 2
    whilelo     p0.b, xzr, n
    whilelo     p1.b, vector_length, n
    b.last      1f
    ld1b        z0.b, p0/z, [src, #0, mul vl]
    ld1b        z1.b, p1/z, [src, #1, mul vl]
    st1b        z0.b, p0, [dest, #0, mul vl]
    st1b        z1.b, p1, [dest, #1, mul vl]
    ret
1:  // if rest > vector_length * 8
    cmp         n, vector_length, lsl 3 // vector_length * 8
    b.hi        \exit
    // if rest <= vector_length * 4
    lsl         tmp1, vector_length, 1  // vector_length * 2
    whilelo     p2.b, tmp1, n
    incb        tmp1
    whilelo     p3.b, tmp1, n
    b.last      1f
    ld1b        z0.b, p0/z, [src, #0, mul vl]
    ld1b        z1.b, p1/z, [src, #1, mul vl]
    ld1b        z2.b, p2/z, [src, #2, mul vl]
    ld1b        z3.b, p3/z, [src, #3, mul vl]
    st1b        z0.b, p0, [dest, #0, mul vl]
    st1b        z1.b, p1, [dest, #1, mul vl]
    st1b        z2.b, p2, [dest, #2, mul vl]
    st1b        z3.b, p3, [dest, #3, mul vl]
    ret
1:  // if rest <= vector_length * 8
    lsl         tmp1, vector_length, 2  // vector_length * 4
    whilelo     p4.b, tmp1, n
    incb        tmp1
    whilelo     p5.b, tmp1, n
    b.last      1f
    ld1b        z0.b, p0/z, [src, #0, mul vl]
    ld1b        z1.b, p1/z, [src, #1, mul vl]
    ld1b        z2.b, p2/z, [src, #2, mul vl]
    ld1b        z3.b, p3/z, [src, #3, mul vl]
    ld1b        z4.b, p4/z, [src, #4, mul vl]
    ld1b        z5.b, p5/z, [src, #5, mul vl]
    st1b        z0.b, p0, [dest, #0, mul vl]
    st1b        z1.b, p1, [dest, #1, mul vl]
    st1b        z2.b, p2, [dest, #2, mul vl]
    st1b        z3.b, p3, [dest, #3, mul vl]
    st1b        z4.b, p4, [dest, #4, mul vl]
    st1b        z5.b, p5, [dest, #5, mul vl]
    ret
1:  lsl         tmp1, vector_length, 2  // vector_length * 4
    incb        tmp1                    // vector_length * 5
    incb        tmp1                    // vector_length * 6
    whilelo     p6.b, tmp1, n
    incb        tmp1
    whilelo     p7.b, tmp1, n
    ld1b        z0.b, p0/z, [src, #0, mul vl]
    ld1b        z1.b, p1/z, [src, #1, mul vl]
    ld1b        z2.b, p2/z, [src, #2, mul vl]
    ld1b        z3.b, p3/z, [src, #3, mul vl]
    ld1b        z4.b, p4/z, [src, #4, mul vl]
    ld1b        z5.b, p5/z, [src, #5, mul vl]
    ld1b        z6.b, p6/z, [src, #6, mul vl]
    ld1b        z7.b, p7/z, [src, #7, mul vl]
    st1b        z0.b, p0, [dest, #0, mul vl]
    st1b        z1.b, p1, [dest, #1, mul vl]
    st1b        z2.b, p2, [dest, #2, mul vl]
    st1b        z3.b, p3, [dest, #3, mul vl]
    st1b        z4.b, p4, [dest, #4, mul vl]
    st1b        z5.b, p5, [dest, #5, mul vl]
    st1b        z6.b, p6, [dest, #6, mul vl]
    st1b        z7.b, p7, [dest, #7, mul vl]
    ret
    .endm

    .globl MEMCPY
    .type MEMCPY,%function
    .p2align 6

MEMCPY:
    .cfi_startproc


L(memcpy):
    cntb        vector_length
    // shortcut for less than vector_length * 8
    // gives a free ptrue to p0.b for n >= vector_length
    shortcut_for_small_size L(vl_agnostic)
    // end of shortcut

L(vl_agnostic): // VL Agnostic
    mov         rest, n
    mov         dest_ptr, dest
    mov         src_ptr, src
    // if rest >= L2_SIZE && vector_length == 64 then L(L2)
    mov         tmp1, 64
    cmp         rest, L2_SIZE
    ccmp        vector_length, tmp1, 0, cs
    b.eq        L(L2)

L(unroll8): // unrolling and software pipeline
    lsl         tmp1, vector_length, 3  // vector_length * 8
    .p2align 3
    cmp         rest, tmp1
    b.cc        L(last)
    ld1b_unroll8
    add         src_ptr, src_ptr, tmp1
    sub         rest, rest, tmp1
    cmp         rest, tmp1
    b.cc        2f
    .p2align 3
1:  stld1b_unroll8
    add         dest_ptr, dest_ptr, tmp1
    add         src_ptr, src_ptr, tmp1
    sub         rest, rest, tmp1
    cmp         rest, tmp1
    b.ge        1b
2:  st1b_unroll8
    add         dest_ptr, dest_ptr, tmp1

    .p2align 3
L(last):
    whilelo     p0.b, xzr, rest
    whilelo     p1.b, vector_length, rest
    b.last      1f
    ld1b        z0.b, p0/z, [src_ptr, #0, mul vl]
    ld1b        z1.b, p1/z, [src_ptr, #1, mul vl]
    st1b        z0.b, p0, [dest_ptr, #0, mul vl]
    st1b        z1.b, p1, [dest_ptr, #1, mul vl]
    ret
1:  lsl         tmp1, vector_length, 1  // vector_length * 2
    whilelo     p2.b, tmp1, rest
    incb        tmp1
    whilelo     p3.b, tmp1, rest
    b.last      1f
    ld1b        z0.b, p0/z, [src_ptr, #0, mul vl]
    ld1b        z1.b, p1/z, [src_ptr, #1, mul vl]
    ld1b        z2.b, p2/z, [src_ptr, #2, mul vl]
    ld1b        z3.b, p3/z, [src_ptr, #3, mul vl]
    st1b        z0.b, p0, [dest_ptr, #0, mul vl]
    st1b        z1.b, p1, [dest_ptr, #1, mul vl]
    st1b        z2.b, p2, [dest_ptr, #2, mul vl]
    st1b        z3.b, p3, [dest_ptr, #3, mul vl]
    ret
1:  lsl         tmp1, vector_length, 2  // vector_length * 4
    whilelo     p4.b, tmp1, rest
    incb        tmp1
    whilelo     p5.b, tmp1, rest
    incb        tmp1
    whilelo     p6.b, tmp1, rest
    incb        tmp1
    whilelo     p7.b, tmp1, rest
    ld1b        z0.b, p0/z, [src_ptr, #0, mul vl]
    ld1b        z1.b, p1/z, [src_ptr, #1, mul vl]
    ld1b        z2.b, p2/z, [src_ptr, #2, mul vl]
    ld1b        z3.b, p3/z, [src_ptr, #3, mul vl]
    ld1b        z4.b, p4/z, [src_ptr, #4, mul vl]
    ld1b        z5.b, p5/z, [src_ptr, #5, mul vl]
    ld1b        z6.b, p6/z, [src_ptr, #6, mul vl]
    ld1b        z7.b, p7/z, [src_ptr, #7, mul vl]
    st1b        z0.b, p0, [dest_ptr, #0, mul vl]
    st1b        z1.b, p1, [dest_ptr, #1, mul vl]
    st1b        z2.b, p2, [dest_ptr, #2, mul vl]
    st1b        z3.b, p3, [dest_ptr, #3, mul vl]
    st1b        z4.b, p4, [dest_ptr, #4, mul vl]
    st1b        z5.b, p5, [dest_ptr, #5, mul vl]
    st1b        z6.b, p6, [dest_ptr, #6, mul vl]
    st1b        z7.b, p7, [dest_ptr, #7, mul vl]
    ret

L(L2):
    // align dest address at CACHE_LINE_SIZE byte boundary
    mov         tmp1, CACHE_LINE_SIZE
    ands        tmp2, dest_ptr, CACHE_LINE_SIZE - 1
    // if cl_remainder == 0
    b.eq        L(L2_dc_zva)
    sub         cl_remainder, tmp1, tmp2
    // process remainder until the first CACHE_LINE_SIZE boundary
    whilelo     p1.b, xzr, cl_remainder        // keep p0.b all true
    whilelo     p2.b, vector_length, cl_remainder
    b.last      1f
    ld1b        z1.b, p1/z, [src_ptr, #0, mul vl]
    ld1b        z2.b, p2/z, [src_ptr, #1, mul vl]
    st1b        z1.b, p1, [dest_ptr, #0, mul vl]
    st1b        z2.b, p2, [dest_ptr, #1, mul vl]
    b           2f
1:  lsl         tmp1, vector_length, 1  // vector_length * 2
    whilelo     p3.b, tmp1, cl_remainder
    incb        tmp1
    whilelo     p4.b, tmp1, cl_remainder
    ld1b        z1.b, p1/z, [src_ptr, #0, mul vl]
    ld1b        z2.b, p2/z, [src_ptr, #1, mul vl]
    ld1b        z3.b, p3/z, [src_ptr, #2, mul vl]
    ld1b        z4.b, p4/z, [src_ptr, #3, mul vl]
    st1b        z1.b, p1, [dest_ptr, #0, mul vl]
    st1b        z2.b, p2, [dest_ptr, #1, mul vl]
    st1b        z3.b, p3, [dest_ptr, #2, mul vl]
    st1b        z4.b, p4, [dest_ptr, #3, mul vl]
2:  add         dest_ptr, dest_ptr, cl_remainder
    add         src_ptr, src_ptr, cl_remainder
    sub         rest, rest, cl_remainder

L(L2_dc_zva):
    // zero fill
    and         tmp1, dest, 0xffffffffffffff
    and         tmp2, src, 0xffffffffffffff
    subs        tmp1, tmp1, tmp2     // diff
    b.ge        1f
    neg         tmp1, tmp1
1:  mov         tmp3, ZF_DIST + CACHE_LINE_SIZE * 2
    cmp         tmp1, tmp3
    b.lo        L(unroll8)
    mov         tmp1, dest_ptr
    dc_zva      (ZF_DIST / CACHE_LINE_SIZE) - 1
    // unroll
    ld1b_unroll8        // this line has to be after "b.lo L(unroll8)"
    add         src_ptr, src_ptr, CACHE_LINE_SIZE * 2
    sub         rest, rest, CACHE_LINE_SIZE * 2
    mov         tmp1, ZF_DIST
    .p2align 3
1:  stld1b_unroll4a
    add         tmp2, dest_ptr, tmp1    // dest_ptr + ZF_DIST
    dc          zva, tmp2
    stld1b_unroll4b
    add         tmp2, tmp2, CACHE_LINE_SIZE
    dc          zva, tmp2
    add         dest_ptr, dest_ptr, CACHE_LINE_SIZE * 2
    add         src_ptr, src_ptr, CACHE_LINE_SIZE * 2
    sub         rest, rest, CACHE_LINE_SIZE * 2
    cmp         rest, tmp3      // ZF_DIST + CACHE_LINE_SIZE * 2
    b.ge        1b
    st1b_unroll8
    add         dest_ptr, dest_ptr, CACHE_LINE_SIZE * 2
    b           L(unroll8)

    .cfi_endproc
    .size MEMCPY,.-MEMCPY

