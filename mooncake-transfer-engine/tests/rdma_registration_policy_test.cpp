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
#include <unordered_set>

#include "config.h"
#include "memory_location.h"
#include "rdma_test_peers.h"
#include "transport/rdma_twosided/rdma_twosided_transport.h"

namespace {
std::atomic<int> calls{0}, live_mrs{0}, location_queries{0}, numa_probes{0};
std::atomic<int> early_location_queries{0};
int fail_call = 0;
int initial_numa_node = -ENOENT;
ibv_pd domains[3]{};
std::unordered_set<void *> device_buffers;
}  // namespace

// Only pointer classification and verbs/NUMA are simulated. The production
// transport, chunk handling, MR maps, metadata, selectors and cleanup execute.
const std::vector<mooncake::MemoryLocationEntry> realMemoryLocation(
    void *addr, size_t length,
    bool only_first_page) asm("__real__ZN8mooncake17getMemoryLocationEPvmb");
const std::vector<mooncake::MemoryLocationEntry> wrapMemoryLocation(
    void *addr, size_t length,
    bool only_first_page) asm("__wrap__ZN8mooncake17getMemoryLocationEPvmb");
const std::vector<mooncake::MemoryLocationEntry> wrapMemoryLocation(
    void *addr, size_t length, bool only_first_page) {
    ++location_queries;
    // This counter is used only by single-buffer registration tests.
    if (calls == 0) ++early_location_queries;
    if (device_buffers.count(addr))
        return {{reinterpret_cast<uint64_t>(addr), length, "cuda:0"}};
    return realMemoryLocation(addr, length, only_first_page);
}

#undef ibv_reg_mr
extern "C" ibv_mr *ibv_reg_mr(ibv_pd *pd, void *addr, size_t length, int) {
    const int call_number = ++calls;
    if (call_number == fail_call) {
        errno = ENOMEM;
        return nullptr;
    }
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
    ++numa_probes;
    for (unsigned long i = 0; i < count; ++i)
        status[i] = live_mrs > 0 ? 1 : initial_numa_node;
    return 0;
}

namespace mooncake {
namespace {
class RdmaRegistrationPolicyTest : public ::testing::TestWithParam<bool> {
   protected:
    void SetUp() override {
        calls = live_mrs = location_queries = numa_probes = 0;
        early_location_queries = 0;
        fail_call = 0;
        initial_numa_node = -ENOENT;
        device_buffers.clear();
        old_limit = globalConfig().max_mr_size;
        old_parallel = globalConfig().parallel_reg_mr;
        old_selection = globalConfig().rdma_nic_selection;
        globalConfig().max_mr_size = pagesize;
        globalConfig().parallel_reg_mr = 0;
        globalConfig().rdma_nic_selection = RdmaNicSelection::ALL;
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
        RdmaTransportTestPeer::initializeLocalNicMap(*transport);
        for (int i = 0; i < 2; ++i) {
            auto *addr = mmap(nullptr, 2 * pagesize, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            ASSERT_NE(addr, MAP_FAILED);
            buffers.push_back(addr);
        }
        device_buffers.insert(buffers[0]);
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
        globalConfig().rdma_nic_selection = old_selection;
    }
    int reg(size_t index, size_t size = pagesize) {
        return transport->registerLocalMemory(buffers[index], size,
                                              kWildcardLocation, true, false);
    }
    int index(const std::string &name) {
        return topology->getDeviceIndex(kWildcardLocation, name);
    }
    auto desc() { return metadata->getSegmentDescByID(LOCAL_SEGMENT_ID); }
    void useLocal() {
        globalConfig().rdma_nic_selection = RdmaNicSelection::LOCAL;
    }
    std::vector<void *> buffers;
    std::unique_ptr<RdmaTransport> transport;
    std::shared_ptr<TransferMetadata> metadata;
    std::shared_ptr<Topology> topology;
    uint64_t old_limit;
    int old_parallel;
    RdmaNicSelection old_selection;
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

TEST_P(RdmaRegistrationPolicyTest,
       DefaultAllDoesNotQueryLocationBeforeRegistration) {
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(location_queries, 1);
    EXPECT_EQ(early_location_queries, 0);
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(desc()->buffers[0].name, "cuda:0");
    EXPECT_EQ(numa_probes, 0);
}
TEST_P(RdmaRegistrationPolicyTest, LocalDeviceKeysKeepGlobalIndices) {
    useLocal();
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(location_queries, 1);
    EXPECT_EQ(early_location_queries, 1);
    EXPECT_EQ(numa_probes, 0);
    const auto &buffer = desc()->buffers[0];
    EXPECT_EQ(buffer.name, "cuda:0");
    ASSERT_EQ(buffer.lkey.size(), 3);
    ASSERT_EQ(buffer.rkey.size(), 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(buffer.lkey[i] != 0, i == index("nic0"));
        EXPECT_EQ(buffer.rkey[i] != 0, i == index("nic0"));
    }
}
TEST_P(RdmaRegistrationPolicyTest,
       HostFinalNumaLocationIsResolvedAfterPinning) {
    useLocal();
    ASSERT_EQ(reg(1), 0);
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(location_queries, 2);
    EXPECT_EQ(early_location_queries, 1);
    EXPECT_EQ(numa_probes, 2);
    EXPECT_EQ(desc()->buffers[0].name, "cpu:1");
}
TEST_P(RdmaRegistrationPolicyTest, EarlyCpuLocationDoesNotNarrowRegistration) {
    useLocal();
    initial_numa_node = 0;
    ASSERT_EQ(topology->parse(R"({"cuda:0":[["nic0"],["nic1","nic2"]],)"
                              R"("cpu:0":[["nic0"],["nic1","nic2"]],)"
                              R"("cpu:1":[["nic0","nic1","nic2"],[]]})"),
              0);
    RdmaTransportTestPeer::initializeLocalNicMap(*transport);
    ASSERT_EQ(reg(1), 0);
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(location_queries, 2);
    EXPECT_EQ(numa_probes, 2);
    EXPECT_EQ(desc()->buffers[0].name, "cpu:1");
}
TEST_P(RdmaRegistrationPolicyTest, HostWithGpuLabelStillRegistersEverywhere) {
    useLocal();
    ASSERT_EQ(transport->registerLocalMemory(buffers[1], pagesize, "cuda:0",
                                             true, false),
              0);
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(numa_probes, 1);
}
TEST_P(RdmaRegistrationPolicyTest, MissingPreferredFallsBackToAllNotAvail) {
    useLocal();
    ASSERT_EQ(topology->parse(R"({"cuda:0":[[],["nic1","nic2"]],)"
                              R"("cpu:1":[["nic0","nic1","nic2"],[]]})"),
              0);
    RdmaTransportTestPeer::initializeLocalNicMap(*transport);
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 3);
}
TEST_P(RdmaRegistrationPolicyTest, UnknownPreferredFallsBackToAllNotAvail) {
    useLocal();
    ASSERT_EQ(topology->parse(R"({"cuda:0":[["absent"],["nic1","nic2"]],)"
                              R"("cpu:1":[["nic0","nic1","nic2"],[]]})"),
              0);
    RdmaTransportTestPeer::initializeLocalNicMap(*transport);
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(live_mrs, 3);
}
TEST_P(RdmaRegistrationPolicyTest, LocalPreferredKeepsPlaceholderIndices) {
    useLocal();
    const auto &contexts = transport->getContextList();
    ASSERT_EQ(contexts.size(), 3);
    const size_t preferred = contexts.size() - 1;
    contexts.front()->set_active(false);
    RdmaContextTestPeer::bindProtectionDomain(*contexts.front(), nullptr);
    ASSERT_EQ(topology->parse(R"({"cuda:0":[[")" +
                              contexts[preferred]->deviceName() + R"("],[]]})"),
              0);
    RdmaTransportTestPeer::initializeLocalNicMap(*transport);
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(live_mrs, 1);
    const auto &buffer = desc()->buffers[0];
    ASSERT_EQ(buffer.lkey.size(), contexts.size());
    ASSERT_EQ(buffer.rkey.size(), contexts.size());
    for (size_t i = 0; i < contexts.size(); ++i) {
        EXPECT_EQ(buffer.lkey[i] != 0, i == preferred);
        EXPECT_EQ(buffer.rkey[i] != 0, i == preferred);
    }
}

TEST_P(RdmaRegistrationPolicyTest, InactivePreferredFallsBackToAll) {
    useLocal();
    transport->getContextList()[index("nic0")]->set_active(false);
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 3);
}
TEST_P(RdmaRegistrationPolicyTest, RecoveredPreferredIsUsedOnNextRegistration) {
    useLocal();
    auto &preferred = transport->getContextList()[index("nic0")];
    preferred->set_active(false);
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 3);
    ASSERT_EQ(transport->unregisterLocalMemory(buffers[0], false), 0);
    preferred->set_active(true);
    calls = 0;
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 1);
    const auto &buffer = desc()->buffers[0];
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(buffer.lkey[i] != 0, i == index("nic0"));
}
TEST_P(RdmaRegistrationPolicyTest, LocalNicCachesArePerTransport) {
    useLocal();
    auto other_metadata = std::make_shared<TransferMetadata>(P2PHANDSHAKE);
    auto other_topology = std::make_shared<Topology>();
    ASSERT_EQ(other_topology->parse(R"({"cuda:0":[["nic2"],["nic0","nic1"]],)"
                                    R"("cpu:1":[["nic0","nic1","nic2"],[]]})"),
              0);
    auto other_desc = std::make_shared<TransferMetadata::SegmentDesc>();
    other_desc->name = "other-registration-policy-test";
    other_desc->protocol = "rdma";
    other_desc->topology = *other_topology;
    for (const auto &name : other_topology->getHcaList())
        other_desc->devices.push_back({name, 0, "", ""});
    ASSERT_EQ(other_metadata->addLocalSegment(LOCAL_SEGMENT_ID,
                                              "other-registration-policy-test",
                                              std::move(other_desc)),
              0);
    RdmaTransport other;
    RdmaTransportTestPeer::bindMetadata(other, other_metadata,
                                        "other-registration-policy-test");
    RdmaTransportTestPeer::bindTopology(other, other_topology);
    size_t i = 0;
    for (const auto &name : other_topology->getHcaList()) {
        auto context = std::make_shared<RdmaContext>(other, name);
        RdmaContextTestPeer::bindProtectionDomain(*context, &domains[i++]);
        RdmaTransportTestPeer::addContext(other, context);
    }
    const int other_preferred =
        other_topology->getDeviceIndex(kWildcardLocation, "nic2");
    RdmaTransportTestPeer::initializeLocalNicMap(other);
    device_buffers.insert(buffers[1]);
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(other.registerLocalMemory(buffers[1], pagesize, kWildcardLocation,
                                        true, false),
              0);
    EXPECT_EQ(calls, 2);
    const auto registered_desc =
        other_metadata->getSegmentDescByID(LOCAL_SEGMENT_ID);
    EXPECT_EQ(other.unregisterLocalMemory(buffers[1], false), 0);
    ASSERT_NE(registered_desc, nullptr);
    ASSERT_EQ(registered_desc->buffers.size(), 1);
    const auto &other_keys = registered_desc->buffers[0].lkey;
    ASSERT_EQ(other_keys.size(), 3);
    const auto &local_keys = desc()->buffers[0].lkey;
    ASSERT_EQ(local_keys.size(), 3);
    for (int j = 0; j < 3; ++j) {
        EXPECT_EQ(local_keys[j] != 0, j == index("nic0"));
        EXPECT_EQ(other_keys[j] != 0, j == other_preferred);
    }
}
TEST_P(RdmaRegistrationPolicyTest, PartialPreferredDoesNotExpandToAvail) {
    useLocal();
    ASSERT_EQ(topology->parse(R"({"cuda:0":[["nic0","nic1"],["nic2"]],)"
                              R"("cpu:1":[["nic0","nic1","nic2"],[]]})"),
              0);
    RdmaTransportTestPeer::initializeLocalNicMap(*transport);
    transport->getContextList()[index("nic1")]->set_active(false);
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 1);
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

TEST_P(RdmaRegistrationPolicyTest, MissingLocationFallsBackToAll) {
    useLocal();
    ASSERT_EQ(transport->registerLocalMemory(buffers[0], pagesize, "cuda:99",
                                             true, false),
              0);
    EXPECT_EQ(calls, 3);
}
TEST_P(RdmaRegistrationPolicyTest, SequentialAndParallelChunksUseSameMask) {
    useLocal();
    for (int parallel : {0, 1}) {
        globalConfig().parallel_reg_mr = parallel;
        calls = 0;
        ASSERT_EQ(reg(0, 2 * pagesize), 0);
        EXPECT_EQ(calls, 2);
        ASSERT_EQ(desc()->buffers.size(), 2);
        for (const auto &buffer : desc()->buffers)
            EXPECT_EQ(buffer.lkey[index("nic1")], 0);
        ASSERT_EQ(transport->unregisterLocalMemory(buffers[0], false), 0);
        EXPECT_EQ(live_mrs, 0);
    }
}
TEST_P(RdmaRegistrationPolicyTest, FailureDoesNotRetryOtherHcasAndRollsBack) {
    useLocal();
    fail_call = 2;
    EXPECT_NE(reg(0, 2 * pagesize), 0);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(live_mrs, 0);
    EXPECT_TRUE(desc()->buffers.empty());
}
TEST_P(RdmaRegistrationPolicyTest, BatchMixesHostAndDeviceWithoutPolicyLeak) {
    useLocal();
    ASSERT_EQ(transport->registerLocalMemoryBatch(
                  {{buffers[0], pagesize}, {buffers[1], pagesize}},
                  kWildcardLocation),
              0);
    EXPECT_EQ(calls, 4);
    EXPECT_EQ(desc()->buffers.size(), 2);
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
