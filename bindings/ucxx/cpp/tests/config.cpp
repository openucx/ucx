/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

#include <ucxx/api.h>

namespace {
TEST(ConfigTest, HandleIsValid)
{
  ucxx::ConfigMap configMap{};
  ucxx::Config config{configMap};

  ASSERT_TRUE(config.getHandle() != nullptr);
}

// TEST(ConfigTest, ConfigMapDefault) {
//     ucxx::Config config{};

//     auto configMapOut = config.get();

//     // ASSERT_GT(configMapOut.size(), 1u);
//     // ASSERT_NE(configMapOut.find("TLS"), configMapOut.end());
//     // ASSERT_EQ(configMapOut["TLS"], "all");
//     for (const auto it : configMapOut) {
//         std::cout << it.first << ": " << it.second << std::endl;
//     }
// }

TEST(ConfigTest, ConfigMapTLS)
{
  ucxx::ConfigMap configMap{{"TLS", "tcp"}};
  ucxx::Config config{configMap};

  auto configMapOut = config.get();

  ASSERT_GT(configMapOut.size(), 1u);
  ASSERT_NE(configMapOut.find("TLS"), configMapOut.end());
  ASSERT_EQ(configMapOut["TLS"], "tcp");
  // auto configMapOut = config.get();
  // for (const auto it : configMapOut) {
  //     std::cout << it.first << ": " << it.second << std::endl;
  // }
}

}  // namespace
