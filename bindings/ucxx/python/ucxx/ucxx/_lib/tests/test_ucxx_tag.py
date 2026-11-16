# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from ucxx._lib import libucxx as ucx_api


def test_ucxx_tag_equality():
    """Test equality comparison for UCXXTag objects."""

    # Test basic equality
    tag1 = ucx_api.UCXXTag(42)
    tag2 = ucx_api.UCXXTag(42)
    tag3 = ucx_api.UCXXTag(100)

    assert tag1 == tag2
    assert tag1 != tag3
    assert tag2 != tag3

    # Test equality with integers
    assert tag1 == 42
    assert tag1 != 100
    assert 42 == tag1
    assert 100 != tag1

    # Test inequality with different types
    assert tag1 != "string"
    assert tag1 != 3.14
    assert tag1 is not None

    # Test hash functionality
    assert hash(tag1) == hash(tag2)
    assert hash(tag1) != hash(tag3)

    # Test string representation
    assert repr(tag1) == "UCXXTag(42)"
    assert repr(tag3) == "UCXXTag(100)"


def test_ucxx_tag_mask_equality():
    """Test equality comparison for UCXXTagMask objects."""

    # Test basic equality
    mask1 = ucx_api.UCXXTagMask(0xFF)
    mask2 = ucx_api.UCXXTagMask(0xFF)
    mask3 = ucx_api.UCXXTagMask(0xAA)

    assert mask1 == mask2
    assert mask1 != mask3
    assert mask2 != mask3

    # Test equality with integers
    assert mask1 == 0xFF
    assert mask1 != 0xAA
    assert 0xFF == mask1
    assert 0xAA != mask1

    # Test inequality with different types
    assert mask1 != "string"
    assert mask1 != 3.14
    assert mask1 is not None

    # Test hash functionality
    assert hash(mask1) == hash(mask2)
    assert hash(mask1) != hash(mask3)

    # Test string representation
    assert repr(mask1) == "UCXXTagMask(255)"
    assert repr(mask3) == "UCXXTagMask(170)"


def test_ucxx_tag_mask_full():
    """Test the UCXXTagMaskFull constant."""

    # Test that UCXXTagMaskFull is correctly defined
    assert ucx_api.UCXXTagMaskFull.value == 2**64 - 1

    # Test equality
    full_mask = ucx_api.UCXXTagMask(2**64 - 1)
    assert ucx_api.UCXXTagMaskFull == full_mask
    assert ucx_api.UCXXTagMaskFull == (2**64 - 1)

    # Test string representation
    expected_repr = f"UCXXTagMask({2**64 - 1})"
    assert repr(ucx_api.UCXXTagMaskFull) == expected_repr


def test_tag_comparison_edge_cases():
    """Test edge cases for tag comparison."""

    # Test zero values
    zero_tag = ucx_api.UCXXTag(0)
    zero_mask = ucx_api.UCXXTagMask(0)

    assert zero_tag == 0
    assert zero_mask == 0

    # Test large values
    large_tag = ucx_api.UCXXTag(2**63)
    large_mask = ucx_api.UCXXTagMask(2**63)

    assert large_tag == 2**63
    assert large_mask == 2**63

    # Test negative integers (should work with unsigned comparison)
    tag = ucx_api.UCXXTag(42)
    assert tag != -1  # Should not be equal to negative value

    # Test with None
    tag = ucx_api.UCXXTag(42)
    assert tag is not None
    assert None is not tag


def test_tag_hash_consistency():
    """Test that hash values are consistent and work in sets/dicts."""

    # Test that equal objects have equal hashes
    tag1 = ucx_api.UCXXTag(42)
    tag2 = ucx_api.UCXXTag(42)
    assert hash(tag1) == hash(tag2)

    # Test that different objects have different hashes
    tag3 = ucx_api.UCXXTag(100)
    assert hash(tag1) != hash(tag3)

    # Test that tags can be used in sets
    tag_set = {ucx_api.UCXXTag(1), ucx_api.UCXXTag(2), ucx_api.UCXXTag(1)}
    assert len(tag_set) == 2  # Duplicate should be removed

    # Test that tags can be used as dictionary keys
    tag_dict = {ucx_api.UCXXTag(1): "one", ucx_api.UCXXTag(2): "two"}
    assert tag_dict[ucx_api.UCXXTag(1)] == "one"
    assert tag_dict[ucx_api.UCXXTag(2)] == "two"


def test_tag_mask_hash_consistency():
    """Test that hash values are consistent for tag masks."""

    # Test that equal objects have equal hashes
    mask1 = ucx_api.UCXXTagMask(0xFF)
    mask2 = ucx_api.UCXXTagMask(0xFF)
    assert hash(mask1) == hash(mask2)

    # Test that different objects have different hashes
    mask3 = ucx_api.UCXXTagMask(0xAA)
    assert hash(mask1) != hash(mask3)

    # Test that masks can be used in sets
    mask_set = {ucx_api.UCXXTagMask(1), ucx_api.UCXXTagMask(2), ucx_api.UCXXTagMask(1)}
    assert len(mask_set) == 2  # Duplicate should be removed

    # Test that masks can be used as dictionary keys
    mask_dict = {ucx_api.UCXXTagMask(1): "one", ucx_api.UCXXTagMask(2): "two"}
    assert mask_dict[ucx_api.UCXXTagMask(1)] == "one"
    assert mask_dict[ucx_api.UCXXTagMask(2)] == "two"
