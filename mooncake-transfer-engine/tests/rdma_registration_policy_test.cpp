// Copyright 2026 KVCache.AI
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <gtest/gtest.h>
#include <sys/mman.h>

#include <atomic>
#include <cerrno>
#include <memory>

#include "config.h"
#include "rdma_test_peers.h"
#include "transport/rdma_twosided/rdma_twosided_transport.h"

namespace {
std::atomic<int> calls{0}, live_mrs{0};
ibv_pd domains[3]{};
}  // namespace

// Only verbs and NUMA are simulated. The production
// transport, chunk handling, MR maps, metadata, selectors and cleanup execute.
#undef ibv_reg_mr
extern "C" ibv_mr *ibv_reg_mr(ibv_pd *pd, void *addr, size_t length, int) {
    ++calls;
    auto *mr = new ibv_mr{};
    mr->addr = addr;
    mr->length = length;
    mr->lkey = 1 + static_cast<uint32_t>(pd - domains);
    mr->rkey = mr->lkey + 10;
    ++live_mrs;
    return mr;
}
extern "C" ibv_mr *ibv_reg_mr_iova2(ibv_pd *pd, void *addr, size_t length,
                                    uint64_t, unsigned int access) {
    return ibv_reg_mr(pd, addr, length, access);
}
extern "C" int ibv_dereg_mr(ibv_mr *mr) {
    --live_mrs;
    delete mr;
    return 0;
}
extern "C" int numa_move_pages(int, unsigned long count, void **, const int *,
                               int *status, int) {
    for (unsigned long i = 0; i < count; ++i)
        status[i] = live_mrs > 0 ? 1 : -ENOENT;
    return 0;
}

namespace mooncake {
namespace {
class RdmaRegistrationPolicyTest : public ::testing::TestWithParam<bool> {
   protected:
    void SetUp() override {
        calls = live_mrs = 0;
        old_limit = globalConfig().max_mr_size;
        old_parallel = globalConfig().parallel_reg_mr;
        globalConfig().max_mr_size = pagesize;
        globalConfig().parallel_reg_mr = 0;
        metadata = std::make_shared<TransferMetadata>(P2PHANDSHAKE);
        auto desc = std::make_shared<TransferMetadata::SegmentDesc>();
        desc->name = "registration-policy-test";
        desc->protocol = "rdma";
        topology = std::make_shared<Topology>();
        ASSERT_EQ(topology->parse(R"({"cuda:0":[["nic0"],["nic1","nic2"]],)"
                                  R"("cpu:1":[["nic0","nic1","nic2"],[]]})"),
                  0);
        desc->topology = *topology;
        for (const auto &name : topology->getHcaList())
            desc->devices.push_back({name, 0, "", ""});
        ASSERT_EQ(
            metadata->addLocalSegment(
                LOCAL_SEGMENT_ID, "registration-policy-test", std::move(desc)),
            0);
        if (GetParam())
            transport = std::make_unique<RdmaTwoSidedTransport>();
        else
            transport = std::make_unique<RdmaTransport>();
        RdmaTransportTestPeer::bindMetadata(*transport, metadata,
                                            "registration-policy-test");
        RdmaTransportTestPeer::bindTopology(*transport, topology);
        size_t i = 0;
        for (const auto &name : topology->getHcaList()) {
            auto context = std::make_shared<RdmaContext>(*transport, name);
            RdmaContextTestPeer::bindProtectionDomain(*context, &domains[i++]);
            RdmaTransportTestPeer::addContext(*transport, context);
        }
        for (int i = 0; i < 2; ++i) {
            auto *addr = mmap(nullptr, 2 * pagesize, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            ASSERT_NE(addr, MAP_FAILED);
            buffers.push_back(addr);
        }
    }
    void TearDown() override {
        for (void *addr : buffers) {
            transport->unregisterLocalMemory(addr, false);
            munmap(addr, 2 * pagesize);
        }
        EXPECT_EQ(live_mrs, 0);
        transport.reset();
        globalConfig().max_mr_size = old_limit;
        globalConfig().parallel_reg_mr = old_parallel;
    }
    int reg(size_t index, size_t size = pagesize) {
        return transport->registerLocalMemory(buffers[index], size,
                                              kWildcardLocation, true, false);
    }
    int index(const std::string &name) {
        return topology->getDeviceIndex(kWildcardLocation, name);
    }
    auto desc() { return metadata->getSegmentDescByID(LOCAL_SEGMENT_ID); }
    std::vector<void *> buffers;
    std::unique_ptr<RdmaTransport> transport;
    std::shared_ptr<TransferMetadata> metadata;
    std::shared_ptr<Topology> topology;
    uint64_t old_limit;
    int old_parallel;
};

TEST_P(RdmaRegistrationPolicyTest, SparseContextsKeepKeyIndices) {
    const int missing = index("nic0");
    RdmaContextTestPeer::bindProtectionDomain(
        *transport->getContextList()[missing], nullptr);
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(live_mrs, 2);
    const auto &buffer = desc()->buffers[0];
    ASSERT_EQ(buffer.lkey.size(), 3);
    ASSERT_EQ(buffer.rkey.size(), 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(buffer.lkey[i] != 0, i != missing);
        EXPECT_EQ(buffer.rkey[i] != 0, i != missing);
    }
}

TEST_P(RdmaRegistrationPolicyTest, SelectionRejectsZeroCachedKeys) {
    ASSERT_EQ(reg(0), 0);
    auto segment = desc();
    const int zero = index("nic0");
    segment->buffers[0].lkey[zero] = 0;
    segment->buffers[0].rkey[zero] = 0;
    int buffer_id = -1, device_id = -1;
    ASSERT_EQ(RdmaTransport::selectDevice(segment.get(), (uint64_t)buffers[0],
                                          16, buffer_id, device_id, 0, 0, zero),
              0);
    EXPECT_NE(device_id, zero);
}

TEST_P(RdmaRegistrationPolicyTest, AllZeroRemoteKeysFailBoundedly) {
    ASSERT_EQ(reg(0), 0);
    auto segment = desc();
    segment->buffers[0].lkey.assign(3, 0);
    segment->buffers[0].rkey.assign(3, 0);
    int buffer_id = -1, device_id = -1;
    EXPECT_EQ(RdmaTransport::selectDevice(segment.get(), (uint64_t)buffers[0],
                                          16, buffer_id, device_id),
              ERR_ADDRESS_NOT_REGISTERED);
}

TEST_P(RdmaRegistrationPolicyTest, LocalOnlyBufferNeedsLkeyNotRkey) {
    ASSERT_EQ(transport->registerLocalMemory(buffers[0], pagesize,
                                             kWildcardLocation, false, false),
              0);
    auto segment = desc();
    int buffer_id = -1, device_id = -1;
    ASSERT_TRUE(segment->buffers[0].rkey.empty());
    ASSERT_EQ(RdmaTransport::selectDevice(segment.get(), (uint64_t)buffers[0],
                                          16, buffer_id, device_id),
              0);
    ASSERT_LT(static_cast<size_t>(device_id), segment->buffers[0].lkey.size());
    EXPECT_NE(segment->buffers[0].lkey[device_id], 0);
}

TEST_P(RdmaRegistrationPolicyTest,
       RemoteSelectionRejectsZeroCacheAndUsesLastKey) {
    ASSERT_EQ(reg(0), 0);
    auto segment = desc();
    auto &buffer = segment->buffers[0];
    buffer.lkey.assign(3, 0);
    buffer.rkey.assign(3, 0);
    const int last = index("nic2");
    buffer.lkey[last] = 7;
    buffer.rkey[last] = 42;
    int buffer_id = -1, device_id = -1;
    ASSERT_EQ(
        RdmaTransport::selectDevice(segment.get(), (uint64_t)buffers[0], 16,
                                    buffer_id, device_id, 0, 0, index("nic0")),
        0);
    EXPECT_EQ(device_id, last);
    // Peer-affinity selection uses the same per-buffer registration mask.
    ASSERT_EQ(RdmaTransport::selectDeviceByLocalHca(
                  segment.get(), (uint64_t)buffers[0], 16, "unknown-local-hca",
                  buffer_id, device_id),
              0);
    EXPECT_EQ(device_id, last);
    EXPECT_NE(RdmaTransport::selectDevice(segment.get(), (uint64_t)buffers[0],
                                          16, "nic1", buffer_id, device_id),
              0);
}

INSTANTIATE_TEST_SUITE_P(RdmaAndTwoSided, RdmaRegistrationPolicyTest,
                         ::testing::Bool());
}  // namespace
}  // namespace mooncake
