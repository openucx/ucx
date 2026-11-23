/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <array>
#include <string>
#include <vector>

namespace ucxx {

const size_t HeaderFramesSize =
  100;  ///< The number of buffers contained in a single `ucxx::Header` object.

/**
 * @brief A serializable object containing metadata of multiple buffers.
 *
 * A serializable object containing metadata of a pre-defined number of buffers used to
 * inform the remote endpoint of multiple incoming messages from buffers of given
 * properties.
 */
class Header {
 private:
  /**
   * @brief Deserialize header.
   *
   * Deserialize a fixed-size header from serialized data.
   *
   * @param[in] serializedHeader  the header in serialized format.
   */
  void deserialize(const std::string& serializedHeader);

 public:
  bool next;                                  ///< Whether there is a next header
  size_t nframes;                             ///< Number of frames
  std::array<int, HeaderFramesSize> isCUDA;   ///< Flag for whether each frame is CUDA or host
  std::array<size_t, HeaderFramesSize> size;  ///< Size in bytes of each frame

  Header() = delete;

  /**
   * @brief Constructor of a fixed-size header.
   *
   * Constructor of a fixed-size header used to transmit pre-defined information about
   * frames that the receiver does not need to know anything about.
   *
   * This constructores receives a flag `next` indicating whether the next message the
   * receiver should expect is another header (in case the number of frames is larger than
   * the pre-defined size), the number of frames `nframes` it contains information for,
   * and pointers to `nframes` arrays of whether each frame is CUDA (`isCUDA == true`) or
   * host (`isCUDA == false`) and the size `size` of each frame in bytes.
   *
   * @param[in] next    whether the receiver should expect a next header.
   * @param[in] nframes the number of frames the header contains information for (must be
   *                    lower or equal than `HeaderFramesSize`).
   * @param[in] isCUDA  array with length `nframes` containing flag of whether each of the
   *                    frames being transferred are CUDA (`true`) or host (`false`).
   * @param[in] size    array with length `nframes` containing the size in bytes of each
   *                    frame.
   */
  Header(bool next, size_t nframes, int* isCUDA, size_t* size);

  /**
   * @brief Constructor of a fixed-size header from serialized data.
   *
   * Reconstruct (i.e., deserialize) a fixed-size header from serialized data.
   *
   * @param[in] serializedHeader  the header in serialized format.
   */
  explicit Header(const std::string& serializedHeader);

  /**
   * @brief Get the size of the underlying data.
   *
   * Get the size of the underlying data, in other words, the size of a serialized
   * `ucxx::Header` ready for transfer.
   *
   * @returns the size of the underlying data.
   */
  [[nodiscard]] static size_t dataSize();

  /**
   * @brief Get the serialized data.
   *
   * Get the serialized data ready for transfer.
   *
   * @returns the serialized data.
   */
  [[nodiscard]] const std::string serialize() const;

  /**
   * @brief Convenience method to build headers given arbitrary-sized input.
   *
   * Convenience method to build one or more headers given arbitrary-sized input `size` and
   * `isCUDA` vectors.
   *
   * @param[in] isCUDA  vector containing flag of whether each frame being transferred are
   *                    CUDA (`1`) or host (`0`).
   * @param[in] size    vector containing the size in bytes of eachf frame.
   *
   * @returns A vector of one or more `ucxx::Header` objects.
   */
  [[nodiscard]] static std::vector<Header> buildHeaders(const std::vector<size_t>& size,
                                                        const std::vector<int>& isCUDA);
};

}  // namespace ucxx
