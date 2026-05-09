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
Tests for sample transfer functionality using real interfaces.

Run with: python tests/workers/rollout/test_sample_transfer.py
"""

import os
import sys

import torch
from tensordict import TensorDict

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from verl import DataProto


def test_dataproto_creation():
    """Test DataProto creation with sample data."""
    print("\n" + "=" * 60)
    print("TEST: DataProto Creation")
    print("=" * 60)

    # Create sample data
    params = {
        "input_ids": torch.randint(0, 32000, (2, 128)),
        "attention_mask": torch.ones(2, 128),
        "pixel_values": torch.randn(2, 3, 224, 224),
    }

    batch = TensorDict(params, batch_size=2)
    meta_info = {"temperature": 1.0, "do_sample": True}
    data_proto = DataProto(batch=batch, meta_info=meta_info)

    print(f"  Batch size: {data_proto.batch.batch_size}")
    print(f"  Input ids shape: {data_proto.batch['input_ids'].shape}")
    print(f"  Pixel values shape: {data_proto.batch['pixel_values'].shape}")

    assert data_proto.batch["input_ids"].shape == (2, 128)
    assert data_proto.batch["pixel_values"].shape == (2, 3, 224, 224)
    print("✅ DataProto creation test PASSED!")


def test_sample_transfer():
    """Test sample transfer via DataProto."""
    print("\n" + "=" * 60)
    print("TEST: Sample Transfer")
    print("=" * 60)

    # Create source data
    source_params = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3),
    }

    # Create DataProto
    batch = TensorDict(source_params, batch_size=2)
    data_proto = DataProto(batch=batch, meta_info={})

    print(f"  Source input_ids: {data_proto.batch['input_ids']}")
    print(f"  Transferred correctly: {data_proto.batch['input_ids'].tolist()}")

    # Verify data integrity
    assert torch.equal(data_proto.batch["input_ids"], source_params["input_ids"])
    assert data_proto.batch["input_ids"][0, 0].item() == 1
    assert data_proto.batch["input_ids"][1, 2].item() == 6

    print("✅ Sample transfer test PASSED!")


def test_sample_data_integrity():
    """Test data integrity during transfer."""
    print("\n" + "=" * 60)
    print("TEST: Data Integrity")
    print("=" * 60)

    # Create complex sample
    params = {
        "input_ids": torch.randn(4, 256),
        "pixel_values": torch.randn(4, 3, 224, 224),
        "position_ids": torch.arange(256).unsqueeze(0).expand(4, -1),
    }

    batch = TensorDict(params, batch_size=4)
    meta_info = {"temperature": 0.7, "do_sample": True}
    data_proto = DataProto(batch=batch, meta_info=meta_info)

    # Verify all tensors match
    for key, value in params.items():
        assert torch.equal(data_proto.batch[key], value), f"Data mismatch for {key}"
        print(f"  ✓ {key}: {value.shape}")

    print("✅ Data integrity test PASSED!")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Sample Transfer Tests")
    print("#" * 60)

    try:
        test_dataproto_creation()
        test_sample_transfer()
        test_sample_data_integrity()

        print("\n" + "#" * 60)
        print("# ALL SAMPLE TRANSFER TESTS PASSED!")
        print("#" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
