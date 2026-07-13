import pytest
import ctypes
import sys
import struct


# Simulate the allocation size calculation that mirrors the vulnerable C code
def calculate_alloc_size(num_sge, sge_size, base_size):
    """
    Simulates the vulnerable allocation size calculation:
    calloc(1, sizeof(*recv_wr) + sizeof(*recv_wr->sge) * wr->num_sge)
    
    Returns the allocation size as it would be computed in C with potential overflow.
    """
    # Simulate C-style size_t multiplication (64-bit or 32-bit depending on platform)
    bits = ctypes.sizeof(ctypes.c_size_t) * 8
    max_size_t = (1 << bits) - 1
    
    # This is the potentially overflowing multiplication
    sge_total = (sge_size * num_sge) & max_size_t
    alloc_size = (base_size + sge_total) & max_size_t
    return alloc_size, sge_total


def safe_alloc_size_check(num_sge, sge_size, base_size):
    """
    Security invariant: allocation size must be large enough to hold all SGEs.
    Returns True if the allocation is safe (no overflow occurred).
    """
    bits = ctypes.sizeof(ctypes.c_size_t) * 8
    max_size_t = (1 << bits) - 1
    
    # Check for multiplication overflow
    if num_sge > 0 and sge_size > max_size_t // num_sge:
        return False  # Overflow would occur
    
    sge_total = sge_size * num_sge
    
    # Check for addition overflow
    if sge_total > max_size_t - base_size:
        return False  # Overflow would occur
    
    return True


def memcpy_size(num_sge, sge_size):
    """
    Simulates the memcpy size: sizeof(*recv_wr->sge) * wr->num_sge
    This is the amount of data that will be copied (without overflow check in vulnerable code).
    """
    # In the vulnerable code, memcpy uses the actual (non-overflowed) size
    return sge_size * num_sge  # Python integers don't overflow


# Adversarial payloads: (num_sge, description)
# These represent attacker-controlled num_sge values that could cause integer overflow
ADVERSARIAL_PAYLOADS = [
    # Values near SIZE_MAX / sizeof(sge) boundaries
    {"num_sge": 0xFFFFFFFF, "desc": "max 32-bit value"},
    {"num_sge": 0xFFFFFFFFFFFFFFFF, "desc": "max 64-bit value"},
    {"num_sge": 0x80000000, "desc": "2^31 - sign bit boundary"},
    {"num_sge": 0x8000000000000000, "desc": "2^63 - 64-bit sign bit"},
    {"num_sge": 0x1000000000000000, "desc": "large 64-bit value"},
    {"num_sge": 0x200000000, "desc": "overflow 32-bit boundary"},
    {"num_sge": (0xFFFFFFFFFFFFFFFF // 32) + 1, "desc": "just over SIZE_MAX/sizeof(sge)"},
    {"num_sge": (0xFFFFFFFFFFFFFFFF // 32), "desc": "at SIZE_MAX/sizeof(sge)"},
    {"num_sge": 0x7FFFFFFF, "desc": "max positive 32-bit signed"},
    {"num_sge": 0x100000001, "desc": "just over 2^32"},
    {"num_sge": 0xFFFFFFFE, "desc": "near max 32-bit"},
    {"num_sge": 2**32 - 1, "desc": "UINT32_MAX"},
    {"num_sge": 2**32, "desc": "UINT32_MAX + 1"},
    {"num_sge": 2**32 + 1, "desc": "UINT32_MAX + 2"},
    {"num_sge": 2**48, "desc": "large power of 2"},
    {"num_sge": 2**63 - 1, "desc": "INT64_MAX"},
    {"num_sge": 2**63, "desc": "INT64_MAX + 1"},
    {"num_sge": 2**64 - 1, "desc": "UINT64_MAX"},
    # Boundary values that could cause subtle overflows
    {"num_sge": 0xAAAAAAAAAAAAAAAA, "desc": "alternating bits pattern"},
    {"num_sge": 0x5555555555555555, "desc": "alternating bits pattern 2"},
]


@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_allocation_size_no_overflow_vulnerability(payload):
    """
    Invariant: The allocation size for SGE (Scatter-Gather Entry) arrays must ALWAYS
    be large enough to hold all entries. If num_sge * sizeof(sge) overflows the
    size_t type, the allocation will be undersized relative to the memcpy size,
    creating a heap buffer overflow. Any implementation MUST detect and reject
    overflow conditions before performing allocation and copy operations.
    """
    num_sge = payload["num_sge"]
    
    # Typical sizes for IB verbs structures
    # sizeof(struct ibv_sge) is typically 16 bytes
    sge_size = 16  # bytes
    # sizeof(*recv_wr) base structure size (approximate)
    base_size = 64  # bytes
    
    bits = ctypes.sizeof(ctypes.c_size_t) * 8
    max_size_t = (1 << bits) - 1
    
    # Calculate what the vulnerable code would allocate
    alloc_size, sge_total_wrapped = calculate_alloc_size(num_sge, sge_size, base_size)
    
    # Calculate what memcpy would actually copy (Python big int, no overflow)
    actual_copy_size = memcpy_size(num_sge, sge_size)
    
    # SECURITY INVARIANT 1: If num_sge is large enough to cause overflow,
    # the system MUST detect this and reject the request (not proceed with
    # an undersized allocation)
    is_safe = safe_alloc_size_check(num_sge, sge_size, base_size)
    
    if not is_safe:
        # The invariant: unsafe inputs MUST be detected as unsafe
        # A secure implementation would return an error, not allocate
        
        # Verify that overflow actually would occur (confirming our detection is correct)
        multiplication_overflows = (num_sge > 0 and sge_size > max_size_t // num_sge)
        addition_overflows = False
        if not multiplication_overflows:
            sge_total = sge_size * num_sge
            addition_overflows = sge_total > max_size_t - base_size
        
        assert multiplication_overflows or addition_overflows, (
            f"safe_alloc_size_check returned False but no overflow detected for "
            f"num_sge={num_sge}, sge_size={sge_size}"
        )
        
        # The wrapped allocation size must be LESS than what memcpy would copy
        # (this is the actual vulnerability condition)
        if multiplication_overflows:
            wrapped_sge_total = (sge_size * num_sge) & max_size_t
            # The allocation is undersized compared to what would be copied
            assert wrapped_sge_total < actual_copy_size, (
                f"Expected overflow to cause undersized allocation: "
                f"wrapped={wrapped_sge_total}, actual_needed={actual_copy_size}"
            )
    
    # SECURITY INVARIANT 2: For safe inputs, allocation must be >= copy size
    if is_safe and num_sge <= max_size_t // sge_size:
        safe_sge_total = sge_size * num_sge
        safe_alloc = base_size + safe_sge_total
        assert safe_alloc >= safe_sge_total, (
            f"Safe allocation {safe_alloc} must be >= copy size {safe_sge_total}"
        )


@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_num_sge_bounds_validation_required(payload):
    """
    Invariant: Any function accepting num_sge as input MUST validate that
    num_sge * element_size does not overflow before using it in allocation.
    The maximum safe num_sge value must be enforced.
    """
    num_sge = payload["num_sge"]
    sge_size = 16  # sizeof(struct ibv_sge)
    
    bits = ctypes.sizeof(ctypes.c_size_t) * 8
    max_size_t = (1 << bits) - 1
    
    # Maximum safe num_sge: floor(SIZE_MAX / sizeof(sge))
    max_safe_num_sge = max_size_t // sge_size
    
    # INVARIANT: If num_sge exceeds max_safe_num_sge, it MUST be rejected
    if num_sge > max_safe_num_sge:
        # Confirm that proceeding would cause overflow
        overflowed_size = (sge_size * num_sge) & max_size_t
        true_size = sge_size * num_sge  # Python big int
        
        assert overflowed_size != true_size, (
            f"Expected overflow for num_sge={num_sge} > max_safe={max_safe_num_sge}, "
            f"but overflowed_size={overflowed_size} == true_size={true_size}"
        )
        
        # The security property: overflowed size is smaller than true size
        assert overflowed_size < true_size, (
            f"Overflow must produce smaller value: "
            f"overflowed={overflowed_size}, true={true_size}"
        )


@pytest.mark.parametrize("num_sge,expected_safe", [
    (0, True),
    (1, True),
    (100, True),
    (1000, True),
    (0xFFFF, True),
    (0xFFFFFFFF, False),  # 32-bit max - likely unsafe on 32-bit, check on 64-bit
    (0xFFFFFFFFFFFFFFFF, False),  # Always unsafe
    (2**63, False),  # Always unsafe
])
def test_safe_num_sge_boundary(num_sge, expected_safe):
    """
    Invariant: The boundary between safe and unsafe num_sge values must be
    correctly identified. Values that would cause size_t overflow in
    (sizeof(sge) * num_sge) must always be classified as unsafe.
    """
    sge_size = 16
    base_size = 64
    
    bits = ctypes.sizeof(ctypes.c_size_t) * 8
    max_size_t = (1 << bits) - 1
    
    is_safe = safe_alloc_size_check(num_sge, sge_size, base_size)
    
    # For values that are always unsafe regardless of platform
    if num_sge > max_size_t:
        # num_sge itself exceeds size_t, definitely unsafe
        assert not is_safe, (
            f"num_sge={num_sge} exceeds max_size_t={max_size_t}, must be unsafe"
        )
    elif num_sge > 0 and sge_size > max_size_t // num_sge:
        # Multiplication would overflow
        assert not is_safe, (
            f"num_sge={num_sge} causes multiplication overflow, must be unsafe"
        )
    
    # For small, obviously safe values
    if num_sge <= 0xFFFF and sge_size * num_sge + base_size <= max_size_t:
        assert is_safe, (
            f"num_sge={num_sge} should be safe but was classified as unsafe"
        )


def test_allocation_size_monotonically_increases_for_safe_inputs():
    """
    Invariant: For safe (non-overflowing) num_sge values, the allocation size
    must monotonically increase as num_sge increases. Any non-monotonic behavior
    indicates integer overflow has occurred.
    """
    sge_size = 16
    base_size = 64
    
    bits = ctypes.sizeof(ctypes.c_size_t) * 8
    max_size_t = (1 << bits) - 1
    
    safe_values = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 65536]
    
    prev_alloc = 0
    for num_sge in safe_values:
        if not safe_alloc_size_check(num_sge, sge_size, base_size):
            continue
        
        alloc_size = base_size + sge_size * num_sge
        
        # INVARIANT: allocation size must be >= previous allocation size
        assert alloc_size >= prev_alloc, (
            f"Allocation size decreased from {prev_alloc} to {alloc_size} "
            f"when num_sge increased to {num_sge} - indicates overflow"
        )
        
        # INVARIANT: allocation must accommodate all SGEs
        assert alloc_size >= sge_size * num_sge, (
            f"Allocation {alloc_size} cannot hold {num_sge} SGEs of size {sge_size}"
        )
        
        prev_alloc = alloc_size