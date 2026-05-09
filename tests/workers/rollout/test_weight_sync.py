# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for weight synchronization using real interfaces including send_weights/receive_weights.

Run with: python tests/workers/rollout/test_weight_sync.py
"""

import asyncio
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "verl_old"))


class SimpleTestModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 64)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class MockCheckpointEngine:
    """Mock checkpoint engine that simulates send/receive weights."""

    def __init__(self, is_master=False):
        self.is_master = is_master
        self.received_weights = []

    async def send_weights(self, weights):
        """Simulate sending weights."""
        print("    [Mock] send_weights called with generator")
        self.received_weights = []
        async for name, param in weights:
            self.received_weights.append((name, param.clone()))
            print(f"      Sent: {name} - {param.shape}")
        print(f"    [Mock] send_weights completed. Total {len(self.received_weights)} weights")

    async def receive_weights(self):
        """Simulate receiving weights as async generator."""
        print("    [Mock] receive_weights called")
        for name, param in self.received_weights:
            yield name, param
        print("    [Mock] receive_weights completed")


class AsyncWeightIterator:
    """Async iterator for weights."""

    def __init__(self, weights_dict):
        self.weights = list(weights_dict.items())
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.weights):
            raise StopAsyncIteration
        name, param = self.weights[self.index]
        self.index += 1
        await asyncio.sleep(0)
        return name, param


def test_import_hf_rollout():
    """Test importing real HFRollout class."""
    print("\n" + "=" * 60)
    print("TEST: Import HFRollout")
    print("=" * 60)

    try:
        from verl_vla.workers.rollout.hf_rollout import HFRollout

        print(f"  ✅ Successfully imported HFRollout: {HFRollout}")
        return HFRollout
    except ImportError as e:
        print(f"  ❌ Failed to import HFRollout: {e}")
        return None


async def test_real_update_weights(HFRolloutClass):
    """Test calling real HFRollout.update_weights()."""
    print("\n" + "=" * 60)
    print("TEST: Call Real HFRollout.update_weights()")
    print("=" * 60)

    if HFRolloutClass is None:
        print("  ❌ Skipped: HFRollout not imported")
        return False

    try:
        source_model = SimpleTestModel()
        target_model = SimpleTestModel()

        for param in target_model.parameters():
            nn.init.zeros_(param)

        print("\n  [Before] Target model weights (zeros):")
        for name, param in target_model.named_parameters():
            print(f"    {name}: {param.sum().item():.6f}")

        prefix = "_fsdp_wrapped_module."
        weights = {f"{prefix}{name}": param.clone() for name, param in source_model.state_dict().items()}
        weights[f"{prefix}critic_backend.target_network.weight"] = torch.randn(128, 128)

        print(f"\n  [Data] Created {len(weights)} weights with FSDP prefix")

        rollout = object.__new__(HFRolloutClass)
        rollout.module = target_model
        rollout.engine = None
        rollout.tokenizer = None
        rollout.output_critic_value = False

        print("\n  [Execute] Calling rollout.update_weights()...")
        result = await rollout.update_weights(AsyncWeightIterator(weights))
        print(f"        Result: {result}")

        print("\n  [After] Target model weights (should match source):")
        target_state = target_model.state_dict()
        source_state = source_model.state_dict()

        for name in source_state:
            if name in target_state:
                assert torch.allclose(target_state[name], source_state[name], rtol=1e-3), (
                    f"Assertion failed: Weight mismatch for {name}"
                )
                print(f"    ✓ {name}: MATCH (assert passed)")

        assert "critic_backend.target_network.weight" not in target_state, (
            "Assertion failed: Critic weight should be filtered!"
        )
        print("    ✓ critic_backend.target_network.weight: correctly filtered (assert passed)")

        print("\n  ✅ All assertions passed!")
        return True

    except Exception as e:
        print(f"  ❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_send_receive_weights_interface():
    """Test send_weights and receive_weights interface simulation."""
    print("\n" + "=" * 60)
    print("TEST: send_weights / receive_weights Interface")
    print("=" * 60)

    try:
        source_model = SimpleTestModel()
        target_model = SimpleTestModel()

        for param in target_model.parameters():
            nn.init.zeros_(param)

        prefix = "_fsdp_wrapped_module."
        source_weights = {f"{prefix}{name}": param.clone() for name, param in source_model.state_dict().items()}

        print("\n  [Step 1] Create mock checkpoint engine")
        engine = MockCheckpointEngine(is_master=True)

        print("\n  [Step 2] Simulate send_weights (trainer side)")

        async def weight_generator():
            for name, param in source_weights.items():
                yield name, param

        await engine.send_weights(weight_generator())

        print("\n  [Step 3] Simulate receive_weights (rollout side)")
        received_weights = {}
        async for name, param in engine.receive_weights():
            received_weights[name] = param
            print(f"      Received: {name} - {param.shape}")

        print("\n  [Step 4] Verify weights match after send/receive")
        assert len(received_weights) == len(source_weights), (
            f"Assertion failed: Expected {len(source_weights)} weights, got {len(received_weights)}"
        )

        for name in source_weights:
            assert name in received_weights, f"Assertion failed: Weight '{name}' not received"
            assert torch.allclose(source_weights[name], received_weights[name], rtol=1e-3), (
                f"Assertion failed: Weight mismatch for {name}"
            )
            print(f"    ✓ {name}: MATCH (assert passed)")

        print("\n  ✅ All assertions passed!")
        return True

    except Exception as e:
        print(f"  ❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_checkpoint_engine_registry():
    """Test checkpoint engine registry and import."""
    print("\n" + "=" * 60)
    print("TEST: Checkpoint Engine Registry")
    print("=" * 60)

    try:
        from verl.checkpoint_engine import CheckpointEngineRegistry

        print("\n  [Available backends]:")
        for backend in CheckpointEngineRegistry._registry.keys():
            print(f"    ✓ {backend}")

        if "nccl" in CheckpointEngineRegistry._registry:
            print("\n  ✅ NCCL backend is registered!")
            return True
        else:
            print("\n  ❌ NCCL backend NOT found in registry")
            return False

    except Exception as e:
        print(f"  ❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Weight Synchronization Tests")
    print("#" * 60)

    HFRolloutClass = test_import_hf_rollout()
    test1_passed = HFRolloutClass is not None

    test2_passed = await test_real_update_weights(HFRolloutClass)

    test3_passed = await test_send_receive_weights_interface()

    test4_passed = await test_checkpoint_engine_registry()

    print("\n" + "#" * 60)
    print("# TEST SUMMARY")
    print("#" * 60)
    print(f"  Test 1 (Import HFRollout):           {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Test 2 (update_weights):             {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"  Test 3 (send/receive_weights):       {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"  Test 4 (Checkpoint Engine Registry): {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print("#" * 60)

    if all([test1_passed, test2_passed, test3_passed, test4_passed]):
        print("# ALL TESTS PASSED!")
    else:
        print("# SOME TESTS FAILED")
    print("#" * 60)

    return all([test1_passed, test2_passed, test3_passed, test4_passed])


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
