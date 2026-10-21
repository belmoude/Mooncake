import ctypes
import os
import unittest
from unittest import mock

import engine


class RdmaRegistrationPolicyTest(unittest.TestCase):
    def test_invalid_override_is_rejected_before_initialization(self):
        te = engine.TransferEngine()
        for value in ("", "avail", "LOCAL"):
            with self.subTest(policy=value):
                self.assertEqual(
                    te.register_memory(0, 4096, rdma_registration_policy=value), -1
                )
                self.assertEqual(
                    te.batch_register_memory(
                        [0], [4096], rdma_registration_policy=value
                    ),
                    -1,
                )

    def test_override_is_keyword_only(self):
        te = engine.TransferEngine()
        with self.assertRaises(TypeError):
            te.register_memory(0, 4096, "*", True, "local")
        with self.assertRaises(TypeError):
            te.batch_register_memory([0], [4096], "*", "local")

    def test_cpu_calls_support_default_and_explicit_overrides(self):
        # Isolate the classic API from runtime TENT and discovery preferences.
        with mock.patch.dict(
            os.environ, {"MC_USE_TENT": "0", "MC_USE_TEV1": "0", "MC_FORCE_TCP": "1"}
        ):
            te = engine.TransferEngine()
            self.assertEqual(
                te.initialize(
                    "127.0.0.1",
                    "P2PHANDSHAKE",
                    "tcp",
                    "",
                    rdma_registration_policy="local",
                ),
                0,
            )
            buffers = [ctypes.create_string_buffer(128) for _ in range(2)]
            addresses = [ctypes.addressof(buffer) for buffer in buffers]
            # Existing positional arguments keep their meaning.
            self.assertEqual(te.register_memory(addresses[0], 128, "*", True), 0)
            self.assertEqual(te.unregister_memory(addresses[0]), 0)
            for policy in (None, "all", "local"):
                with self.subTest(policy=policy):
                    self.assertEqual(
                        te.register_memory(
                            addresses[0], 128, rdma_registration_policy=policy
                        ),
                        0,
                    )
                    self.assertEqual(te.unregister_memory(addresses[0]), 0)
                    self.assertEqual(
                        te.batch_register_memory(
                            addresses, [128, 128], "*", rdma_registration_policy=policy
                        ),
                        0,
                    )
                    self.assertEqual(te.batch_unregister_memory(addresses), 0)


if __name__ == "__main__":
    unittest.main()
