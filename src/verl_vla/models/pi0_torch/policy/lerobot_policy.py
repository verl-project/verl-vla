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

import os

import torch
from typing_extensions import override
from verl.protocol import DataProto

from verl_vla.utils.image import image_to_float01, is_int8_image_tensor, preprocess_image_batch_to_uint8

from .base import Pi0Input, Pi0Output

PI0_MAX_STATE_DIM = 32
PI0_ACTION_CHUNK_SIZE = 10
LEROBOT_ACTION_DIM = 6
LEROBOT_IMAGE_CROP_SIZE = 480
LEROBOT_IMAGE_RESIZE_SIZE = (224, 224)


class LerobotPi0Input(Pi0Input):
    _debug_dump_count = 0

    @staticmethod
    def _save_debug_image(image: torch.Tensor, image_path: str):
        image = image.detach().cpu().float()
        if image.ndim == 4:
            image = image[0]

        if image.max() > 1.0:
            image = image / 255.0
        image = image.clamp(0.0, 1.0)
        image_uint8 = (image * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()

        try:
            from PIL import Image

            Image.fromarray(image_uint8).save(image_path)
        except Exception:
            torch.save(image, image_path + ".pt")

    @classmethod
    def _debug_dump_inputs(
        cls,
        cam_high: torch.Tensor,
        left_wrist: torch.Tensor,
        state: torch.Tensor,
        task: list,
    ):
        debug_max_dumps = int(os.getenv("VERL_PI0_DEBUG_MAX_DUMPS", "200"))
        if cls._debug_dump_count >= debug_max_dumps:
            return

        debug_dir = os.getenv("VERL_PI0_DEBUG_DIR", "./outputs/pi0_input_debug")
        os.makedirs(debug_dir, exist_ok=True)

        dump_id = cls._debug_dump_count
        cls._debug_dump_count += 1

        print(
            f"[Pi0Input Debug] dump_id={dump_id}, cam_high={tuple(cam_high.shape)}, "
            f"left_wrist={tuple(left_wrist.shape)}, state={tuple(state.shape)}"
        )
        if len(task) > 0:
            print(f"[Pi0Input Debug] task[0]={task[0]}")
            print(f"[Pi0Input Debug] state[0]={state[0].detach().cpu().tolist()}")

        cls._save_debug_image(cam_high[0], os.path.join(debug_dir, f"dump_{dump_id:06d}_cam_high.png"))
        cls._save_debug_image(left_wrist[0], os.path.join(debug_dir, f"dump_{dump_id:06d}_left_wrist.png"))

    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> "LerobotPi0Input":
        input = cls()

        images = env_obs.batch["observation.images.top"]
        wrist_images = env_obs.batch["observation.images.wrist"]
        device = images.device

        if is_int8_image_tensor(images) and is_int8_image_tensor(wrist_images):
            images = image_to_float01(images).to(device=device)
            wrist_images = image_to_float01(wrist_images).to(device=device)
        else:
            images = preprocess_image_batch_to_uint8(
                images,
                crop_size=LEROBOT_IMAGE_CROP_SIZE,
                resize_size=LEROBOT_IMAGE_RESIZE_SIZE,
            )
            wrist_images = preprocess_image_batch_to_uint8(
                wrist_images,
                crop_size=LEROBOT_IMAGE_CROP_SIZE,
                resize_size=LEROBOT_IMAGE_RESIZE_SIZE,
            )
            images = image_to_float01(images).to(device=device)
            wrist_images = image_to_float01(wrist_images).to(device=device)

        batch_size = images.shape[0]
        cam_high = images
        left_wrist = wrist_images
        empty_images = torch.zeros(
            (batch_size, 3, cam_high.shape[2], cam_high.shape[3]),
            device=device,
            dtype=torch.bfloat16,
        )

        input.images = {
            "observation.images.cam_high": cam_high.to(torch.bfloat16),
            "observation.images.cam_left_wrist": left_wrist.to(torch.bfloat16),
            "observation.images.cam_right_wrist": empty_images,
        }
        input.img_masks = [
            torch.ones((batch_size,), device=device, dtype=torch.bool),
            torch.ones((batch_size,), device=device, dtype=torch.bool),
            torch.zeros((batch_size,), device=device, dtype=torch.bool),
        ]

        task_key = "task_descriptions" if "task_descriptions" in env_obs.non_tensor_batch else "task"
        input.task = list(env_obs.non_tensor_batch[task_key])

        state = env_obs.batch["observation.state"]
        input.state = torch.nn.functional.pad(
            state, (0, max(0, PI0_MAX_STATE_DIM - state.shape[-1])), "constant", 0
        ).to(device=device, dtype=torch.float32)

        return input


class LerobotPi0Output(Pi0Output):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> "LerobotPi0Output":
        output = cls()
        output.action = model_output["full_action"][:, :PI0_ACTION_CHUNK_SIZE, :LEROBOT_ACTION_DIM]
        output.log_prob = model_output["log_probs"]
        return output
