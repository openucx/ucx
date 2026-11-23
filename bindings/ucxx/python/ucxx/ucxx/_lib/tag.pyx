# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


from libc.stdint cimport uint64_t


cdef class UCXXTag:
    cdef readonly uint64_t value

    def __cinit__(self, uint64_t tag) -> None:
        self.value = tag

    def __eq__(self, other) -> bool:
        """Compare this tag with another tag or integer value.

        Parameters
        ----------
        other : UCXXTag or int
            The tag or integer value to compare with.

        Returns
        -------
        bool
            True if the tags are equal, False otherwise.
        """
        if isinstance(other, UCXXTag):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        """Compare this tag with another tag or integer value for inequality.

        Parameters
        ----------
        other : UCXXTag or int
            The tag or integer value to compare with.

        Returns
        -------
        bool
            True if the tags are not equal, False otherwise.
        """
        if isinstance(other, UCXXTag):
            return self.value != other.value
        elif isinstance(other, int):
            return self.value != other
        else:
            return NotImplemented

    def __hash__(self) -> int:
        """Return hash value for this tag.

        Returns
        -------
        int
            Hash value based on the tag value.
        """
        return hash(self.value)

    def __repr__(self) -> str:
        """Return string representation of this tag.

        Returns
        -------
        str
            String representation showing the tag value.
        """
        return f"UCXXTag({self.value})"


cdef class UCXXTagMask:
    cdef readonly uint64_t value

    def __cinit__(self, uint64_t tag_mask) -> None:
        self.value = tag_mask

    def __eq__(self, other) -> bool:
        """Compare this tag mask with another tag mask or integer value.

        Parameters
        ----------
        other : UCXXTagMask or int
            The tag mask or integer value to compare with.

        Returns
        -------
        bool
            True if the tag masks are equal, False otherwise.
        """
        if isinstance(other, UCXXTagMask):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        """Compare this tag mask with another tag mask or integer value for inequality.

        Parameters
        ----------
        other : UCXXTagMask or int
            The tag mask or integer value to compare with.

        Returns
        -------
        bool
            True if the tag masks are not equal, False otherwise.
        """
        if isinstance(other, UCXXTagMask):
            return self.value != other.value
        elif isinstance(other, int):
            return self.value != other
        else:
            return NotImplemented

    def __hash__(self) -> int:
        """Return hash value for this tag mask.

        Returns
        -------
        int
            Hash value based on the tag mask value.
        """
        return hash(self.value)

    def __repr__(self) -> str:
        """Return string representation of this tag mask.

        Returns
        -------
        str
            String representation showing the tag mask value.
        """
        return f"UCXXTagMask({self.value})"


UCXXTagMaskFull = UCXXTagMask(2 ** 64 - 1)
