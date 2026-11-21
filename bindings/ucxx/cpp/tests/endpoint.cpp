/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <ucxx/api.h>

namespace {

class EndpointTest : public ::testing::Test {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build()};
  std::shared_ptr<ucxx::Context> _remoteContext{
    ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build()};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Worker> _remoteWorker{nullptr};

  virtual void SetUp()
  {
    _worker       = _context->workerBuilder().build();
    _remoteWorker = _remoteContext->workerBuilder().build();
  }
};

TEST_F(EndpointTest, HandleIsValid)
{
  auto ep = _worker->endpointBuilder(_worker->addressBuilder().build()).build();
  _worker->progress();

  ASSERT_TRUE(ep->getHandle() != nullptr);
}

TEST_F(EndpointTest, IsAlive)
{
  GTEST_SKIP()
    << "Connecting to worker via its UCX address doesn't seem to call endpoint error handler";
  auto ep = _worker->endpointBuilder(_remoteWorker->addressBuilder().build()).build();
  _worker->progress();
  _remoteWorker->progress();

  ASSERT_TRUE(ep->isAlive());

  std::vector<int> buf{123};
  std::shared_ptr<ucxx::Request> send_req =
    ep->tagSendBuilder(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0}).build();
  while (!send_req->isCompleted())
    _worker->progress();

  _remoteWorker  = nullptr;
  _remoteContext = nullptr;
  _worker->progress();
  ASSERT_FALSE(ep->isAlive());
}

TEST(AddressTest, EmptyAddressRejected)
{
  EXPECT_THROW(std::ignore = ucxx::AddressBuilder("").build(), std::invalid_argument);
}

}  // namespace
