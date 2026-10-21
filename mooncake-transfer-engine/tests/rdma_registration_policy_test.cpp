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
#include <future>
#include <memory>
#include <unordered_set>

#include "config.h"
#include "rdma_test_peers.h"
#include "transfer_engine.h"
#include "transfer_engine_impl.h"
#include "transport/rdma_twosided/rdma_twosided_transport.h"

namespace {
std::atomic<int> calls{0}, live_mrs{0}, device_probes{0}, numa_probes{0};
int fail_call = 0;
ibv_pd domains[3]{};
std::unordered_set<void *> device_buffers;
}  // namespace

// Only pointer classification and verbs/NUMA are simulated. The production
// transport, chunk handling, MR maps, metadata, selectors and cleanup execute.
#if _GLIBCXX_USE_CXX11_ABI
std::string wrapDeviceMemoryLocation(void *addr) asm(
    "__wrap__ZN8mooncake23getDeviceMemoryLocationB5cxx11EPv");
#else
std::string wrapDeviceMemoryLocation(void *addr) asm(
    "__wrap__ZN8mooncake23getDeviceMemoryLocationEPv");
#endif
std::string wrapDeviceMemoryLocation(void *addr) {
    ++device_probes;
    return device_buffers.count(addr) ? "cuda:0" : "";
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
        status[i] = live_mrs > 0 ? 1 : -ENOENT;
    return 0;
}

namespace mooncake {
class TransferEngineImplTestPeer {
   public:
    static void prepare(TransferEngineImpl &engine,
                        std::shared_ptr<TransferMetadata> metadata,
                        std::shared_ptr<Topology> topology,
                        std::shared_ptr<Transport> transport = nullptr) {
        engine.local_server_name_ = "registration-policy-test";
        engine.metadata_ = std::move(metadata);
        engine.local_topology_ = std::move(topology);
        engine.multi_transports_ = std::make_shared<MultiTransport>(
            engine.metadata_, engine.local_server_name_);
        if (transport) attach(engine, std::move(transport));
    }
    static void attach(TransferEngineImpl &engine,
                       std::shared_ptr<Transport> transport) {
        const std::string name = transport->getName();
        engine.multi_transports_->transport_map_[name] = std::move(transport);
    }
    static RdmaRegistrationPolicy policy(TransferEngineImpl &engine,
                                         void *addr) {
        return engine.local_memory_regions_.at((uintptr_t)addr)
            .rdma_registration_policy;
    }
    static size_t records(TransferEngineImpl &engine) {
        return engine.local_memory_regions_.size();
    }
    static size_t pending(TransferEngineImpl &engine) {
        return engine.registering_memory_regions_.size();
    }
    static int replay(TransferEngineImpl &engine, Transport *transport) {
        return engine.replayLocalMemory(transport);
    }
};

namespace {
using Policy = RdmaRegistrationPolicy;
class RdmaRegistrationPolicyTest : public ::testing::TestWithParam<bool> {
   protected:
    void SetUp() override {
        calls = live_mrs = device_probes = numa_probes = 0;
        fail_call = 0;
        device_buffers.clear();
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
            transport = std::make_shared<RdmaTwoSidedTransport>();
        else
            transport = std::make_shared<RdmaTransport>();
        RdmaTransportTestPeer::bindMetadata(*transport, metadata,
                                            "registration-policy-test");
        RdmaTransportTestPeer::bindTopology(*transport, topology);
        size_t i = 0;
        for (const auto &name : topology->getHcaList()) {
            auto context = std::make_shared<RdmaContext>(*transport, name);
            RdmaContextTestPeer::bindProtectionDomain(*context, &domains[i++]);
            RdmaTransportTestPeer::addContext(*transport, context);
        }
        RdmaTransportTestPeer::initializePreferredDevices(*transport);
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
        RdmaTransportTestPeer::setRegistrationPolicy(*transport,
                                                     Policy::TopologyLocal);
    }
    std::vector<void *> buffers;
    std::shared_ptr<RdmaTransport> transport;
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

TEST_P(RdmaRegistrationPolicyTest, DefaultAllDoesNotProbeBeforeRegistration) {
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(device_probes, 0);
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(desc()->buffers[0].name, "cpu:1");
    EXPECT_EQ(numa_probes, 1);
}
TEST_P(RdmaRegistrationPolicyTest, LocalDeviceKeysKeepGlobalIndices) {
    useLocal();
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 1);
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
TEST_P(RdmaRegistrationPolicyTest, HostResolvesNumaOnlyAfterPinning) {
    useLocal();
    ASSERT_EQ(reg(1), 0);
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(numa_probes, 1);
    EXPECT_EQ(desc()->buffers[0].name, "cpu:1");
}
TEST_P(RdmaRegistrationPolicyTest, HostWithGpuLabelStillRegistersEverywhere) {
    useLocal();
    ASSERT_EQ(transport->registerLocalMemory(buffers[1], pagesize, "cuda:0",
                                             true, false),
              0);
    EXPECT_EQ(calls, 3);
    EXPECT_EQ(numa_probes, 0);
}
TEST_P(RdmaRegistrationPolicyTest, MissingPreferredFallsBackToAllNotAvail) {
    useLocal();
    ASSERT_EQ(topology->parse(R"({"cuda:0":[[],["nic1","nic2"]],)"
                              R"("cpu:1":[["nic0","nic1","nic2"],[]]})"),
              0);
    RdmaTransportTestPeer::initializePreferredDevices(*transport);
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 3);
}
TEST_P(RdmaRegistrationPolicyTest, InactivePreferredFallsBackToAll) {
    useLocal();
    transport->getContextList()[index("nic0")]->set_active(false);
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 3);
}
TEST_P(RdmaRegistrationPolicyTest, PartialPreferredDoesNotExpandToAvail) {
    useLocal();
    ASSERT_EQ(topology->parse(R"({"cuda:0":[["nic0","nic1"],["nic2"]],)"
                              R"("cpu:1":[["nic0","nic1","nic2"],[]]})"),
              0);
    RdmaTransportTestPeer::initializePreferredDevices(*transport);
    transport->getContextList()[index("nic1")]->set_active(false);
    ASSERT_EQ(reg(0), 0);
    EXPECT_EQ(calls, 1);
}

TEST_P(RdmaRegistrationPolicyTest, LocalSelectionRejectsZeroCachedLkey) {
    ASSERT_EQ(reg(0), 0);
    auto segment = desc();
    const int zero = index("nic0");
    segment->buffers[0].lkey[zero] = 0;
    int buffer_id = -1, device_id = -1;
    ASSERT_EQ(RdmaTransportTestPeer::selectLocalDevice(
                  *transport, segment.get(), (uint64_t)buffers[0], 16,
                  buffer_id, device_id, 0, zero),
              0);
    EXPECT_NE(device_id, zero);
}

TEST_P(RdmaRegistrationPolicyTest, AllZeroRemoteKeysFailBoundedly) {
    ASSERT_EQ(reg(0), 0);
    auto segment = desc();
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
    ASSERT_EQ(RdmaTransportTestPeer::selectLocalDevice(
                  *transport, segment.get(), (uint64_t)buffers[0], 16,
                  buffer_id, device_id),
              0);
    EXPECT_TRUE(RdmaTransport::hasLocalKey(segment->buffers[0], device_id));
    EXPECT_NE(RdmaTransport::selectDevice(segment.get(), (uint64_t)buffers[0],
                                          16, buffer_id, device_id),
              0);
}
TEST_P(RdmaRegistrationPolicyTest,
       RemoteSelectionRejectsZeroCacheAndUsesLastKey) {
    useLocal();
    ASSERT_EQ(reg(0), 0);
    auto segment = desc();
    auto &buffer = segment->buffers[0];
    buffer.rkey.assign(3, 0);
    const int last = index("nic2");
    buffer.rkey[last] = 42;
    int buffer_id = -1, device_id = -1;
    ASSERT_EQ(
        RdmaTransport::selectDevice(segment.get(), (uint64_t)buffers[0], 16,
                                    buffer_id, device_id, 0, 0, index("nic0")),
        0);
    EXPECT_EQ(device_id, last);
    // Peer-affinity selection also needs the remote rkey, not the local lkey.
    ASSERT_EQ(RdmaTransport::selectDeviceByLocalHca(
                  segment.get(), (uint64_t)buffers[0], 16, "unknown-local-hca",
                  buffer_id, device_id),
              0);
    EXPECT_EQ(device_id, last);
    EXPECT_NE(RdmaTransport::selectDevice(segment.get(), (uint64_t)buffers[0],
                                          16, "nic1", buffer_id, device_id),
              0);
}
TEST_P(RdmaRegistrationPolicyTest, LocalSelectionSkipsInactiveCachedDevice) {
    ASSERT_EQ(reg(0), 0);
    auto segment = desc();
    const int inactive = index("nic0");
    transport->getContextList()[inactive]->set_active(false);
    int buffer_id = -1, device_id = -1;
    ASSERT_EQ(RdmaTransportTestPeer::selectLocalDevice(
                  *transport, segment.get(), (uint64_t)buffers[0], 16,
                  buffer_id, device_id, 0, inactive),
              0);
    EXPECT_NE(device_id, inactive);
}

TEST_P(RdmaRegistrationPolicyTest, PublicOverridesDoNotChangeTransportDefault) {
    useLocal();
    device_buffers.insert(buffers[1]);
    ASSERT_EQ(transport->registerLocalMemory(buffers[0], pagesize, Policy::All,
                                             kWildcardLocation, true, false),
              0);
    ASSERT_EQ(reg(1), 0);
    EXPECT_EQ(calls, 4);
    ASSERT_EQ(transport->unregisterLocalMemory(buffers[0], false), 0);
    ASSERT_EQ(transport->unregisterLocalMemory(buffers[1], false), 0);
    ASSERT_EQ(transport->registerLocalMemoryBatch(
                  {{buffers[0], pagesize}, {buffers[1], pagesize}}, Policy::All,
                  kWildcardLocation),
              0);
    EXPECT_EQ(calls, 10);
}

TEST_P(RdmaRegistrationPolicyTest, EngineRecordsConcurrentSingleOverrides) {
    useLocal();
    device_buffers.insert(buffers[1]);
    TransferEngineImpl engine;
    ASSERT_EQ(engine.setRdmaRegistrationPolicy(Policy::TopologyLocal), 0);
    TransferEngineImplTestPeer::prepare(engine, metadata, topology, transport);
    auto first = std::async(std::launch::async, [&] {
        return engine.registerLocalMemory(buffers[0], pagesize, Policy::All,
                                          kWildcardLocation, true, false);
    });
    ASSERT_EQ(engine.registerLocalMemory(buffers[1], pagesize,
                                         kWildcardLocation, true, false),
              0);
    ASSERT_EQ(first.get(), 0);
    EXPECT_EQ(calls, 4);
    EXPECT_EQ(TransferEngineImplTestPeer::policy(engine, buffers[0]),
              Policy::All);
    EXPECT_EQ(TransferEngineImplTestPeer::policy(engine, buffers[1]),
              Policy::TopologyLocal);
    ASSERT_EQ(engine.unregisterLocalMemory(buffers[0], false), 0);
    ASSERT_EQ(engine.unregisterLocalMemory(buffers[1], false), 0);
    EXPECT_EQ(TransferEngineImplTestPeer::records(engine), 0);
}

TEST_P(RdmaRegistrationPolicyTest, ReplayUsesSnapshotsInBothDefaultDirections) {
    // The transport default intentionally differs from one of the snapshots.
    useLocal();
    device_buffers.insert(buffers[1]);
    for (Policy default_policy : {Policy::All, Policy::TopologyLocal}) {
        TransferEngineImpl engine;
        ASSERT_EQ(engine.setRdmaRegistrationPolicy(default_policy), 0);
        TransferEngineImplTestPeer::prepare(engine, metadata, topology);
        Policy override_policy =
            default_policy == Policy::All ? Policy::TopologyLocal : Policy::All;
        ASSERT_EQ(
            engine.registerLocalMemory(buffers[0], pagesize, override_policy),
            0);
        ASSERT_EQ(engine.registerLocalMemoryBatch({{buffers[1], pagesize}},
                                                  kWildcardLocation),
                  0);
        EXPECT_EQ(live_mrs, 0);
        EXPECT_EQ(TransferEngineImplTestPeer::policy(engine, buffers[0]),
                  override_policy);
        EXPECT_EQ(TransferEngineImplTestPeer::policy(engine, buffers[1]),
                  default_policy);
        const int previous_calls = calls.load();
        TransferEngineImplTestPeer::attach(engine, transport);
        ASSERT_EQ(TransferEngineImplTestPeer::replay(engine, transport.get()),
                  0);
        EXPECT_EQ(calls - previous_calls, 4);
        EXPECT_EQ(live_mrs, 4);
        ASSERT_EQ(engine.unregisterLocalMemory(buffers[0], false), 0);
        ASSERT_EQ(engine.unregisterLocalMemory(buffers[1], false), 0);
        EXPECT_EQ(live_mrs, 0);
    }
}

TEST_P(RdmaRegistrationPolicyTest,
       BatchFailureReleasesSnapshotsAndAllowsRetry) {
    device_buffers.insert(buffers[1]);
    TransferEngineImpl engine;
    ASSERT_EQ(engine.setRdmaRegistrationPolicy(Policy::TopologyLocal), 0);
    TransferEngineImplTestPeer::prepare(engine, metadata, topology, transport);
    const std::vector<BufferEntry> batch = {{buffers[0], pagesize},
                                            {buffers[1], pagesize}};
    fail_call = 4;
    EXPECT_NE(engine.registerLocalMemoryBatch(batch, Policy::All), 0);
    EXPECT_EQ(live_mrs, 0);
    EXPECT_EQ(TransferEngineImplTestPeer::records(engine), 0);
    EXPECT_EQ(TransferEngineImplTestPeer::pending(engine), 0);
    fail_call = 0;
    ASSERT_EQ(engine.registerLocalMemoryBatch(batch, Policy::All), 0);
    EXPECT_EQ(live_mrs, 6);
    for (void *addr : buffers)
        EXPECT_EQ(TransferEngineImplTestPeer::policy(engine, addr),
                  Policy::All);
    EXPECT_EQ(
        engine.registerLocalMemory(buffers[0], pagesize, Policy::TopologyLocal),
        ERR_ADDRESS_OVERLAPPED);
    ASSERT_EQ(engine.unregisterLocalMemoryBatch({buffers[0], buffers[1]}), 0);
    ASSERT_EQ(engine.registerLocalMemory(buffers[0], pagesize), 0);
    EXPECT_EQ(live_mrs, 1);
    EXPECT_EQ(TransferEngineImplTestPeer::policy(engine, buffers[0]),
              Policy::TopologyLocal);
    ASSERT_EQ(engine.unregisterLocalMemory(buffers[0], false), 0);
}

TEST_P(RdmaRegistrationPolicyTest,
       InvalidOverrideHasNoRegistrationSideEffects) {
    TransferEngineImpl engine;
    TransferEngineImplTestPeer::prepare(engine, metadata, topology, transport);
    const Policy invalid = static_cast<Policy>(99);
    EXPECT_EQ(engine.registerLocalMemory(buffers[0], pagesize, invalid),
              ERR_INVALID_ARGUMENT);
    EXPECT_EQ(
        engine.registerLocalMemoryBatch({{buffers[0], pagesize}}, invalid),
        ERR_INVALID_ARGUMENT);
    EXPECT_EQ(transport->registerLocalMemory(buffers[0], pagesize, invalid,
                                             kWildcardLocation, true, false),
              ERR_INVALID_ARGUMENT);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(TransferEngineImplTestPeer::records(engine), 0);
    EXPECT_EQ(TransferEngineImplTestPeer::pending(engine), 0);
}

#ifdef ENABLE_MULTI_PROTOCOL
TEST_P(RdmaRegistrationPolicyTest, MultiProtocolSnapshotKeepsExplicitPolicy) {
    TransferEngineImpl engine;
    ASSERT_EQ(engine.setRdmaRegistrationPolicy(Policy::TopologyLocal), 0);
    TransferEngineImplTestPeer::prepare(engine, metadata, topology, transport);
    std::unordered_map<std::string, std::vector<RegisteredBuffer>> request;
    request[transport->getName()].emplace_back(buffers[0], pagesize,
                                               kWildcardLocation, true, false);
    ASSERT_EQ(engine.mp_registerLocalMemory(request, Policy::All), 0);
    EXPECT_EQ(live_mrs, 3);
    EXPECT_EQ(TransferEngineImplTestPeer::policy(engine, buffers[0]),
              Policy::All);
    ASSERT_EQ(engine.mp_unregisterLocalMemory(request), 0);
    EXPECT_EQ(TransferEngineImplTestPeer::records(engine), 0);
    ASSERT_EQ(engine.mp_registerLocalMemory(request), 0);
    EXPECT_EQ(live_mrs, 1);
    EXPECT_EQ(TransferEngineImplTestPeer::policy(engine, buffers[0]),
              Policy::TopologyLocal);
    ASSERT_EQ(engine.mp_unregisterLocalMemory(request), 0);
}
#endif

INSTANTIATE_TEST_SUITE_P(RdmaAndTwoSided, RdmaRegistrationPolicyTest,
                         ::testing::Bool());

TEST(RdmaRegistrationPolicy, EngineFreezesConfigurationAtInitAndAfterMove) {
    TransferEngine first(false), second(false);
    if (first.isUsingTent()) {
        EXPECT_EQ(first.setRdmaRegistrationPolicy(Policy::TopologyLocal),
                  ERR_NOT_IMPLEMENTED);
        EXPECT_EQ(
            first.registerLocalMemory(nullptr, pagesize, Policy::TopologyLocal),
            ERR_NOT_IMPLEMENTED);
        EXPECT_EQ(first.registerLocalMemoryBatch({}, Policy::TopologyLocal),
                  ERR_NOT_IMPLEMENTED);
        EXPECT_EQ(first.registerLocalMemory(nullptr, pagesize,
                                            static_cast<Policy>(99)),
                  ERR_INVALID_ARGUMENT);
        return;
    }
    ASSERT_EQ(first.setRdmaRegistrationPolicy(Policy::TopologyLocal), 0);
    EXPECT_EQ(second.setRdmaRegistrationPolicy(static_cast<Policy>(99)),
              ERR_INVALID_ARGUMENT);
    ASSERT_EQ(first.init(P2PHANDSHAKE, "127.0.0.1:0", "127.0.0.1", 0), 0);
    EXPECT_EQ(first.setRdmaRegistrationPolicy(Policy::All),
              ERR_INVALID_ARGUMENT);
    TransferEngine moved(std::move(first));
    EXPECT_EQ(moved.setRdmaRegistrationPolicy(Policy::All),
              ERR_INVALID_ARGUMENT);
    EXPECT_EQ(second.setRdmaRegistrationPolicy(Policy::All), 0);
}
}  // namespace
}  // namespace mooncake
