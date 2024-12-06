# Copyright 2024 the LlamaFactory team.
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

from dataclasses import dataclass, field
from typing import List, Literal, Optional

@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )

@dataclass
class FinetuningArguments(LoraArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze vision tower in MLLM training."},
    )
    train_mm_proj_only: bool = field(
        default=False,
        metadata={"help": "Whether or not to train the multimodal projector for MLLM only."},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: List[str] = split_arg(self.lora_target)
        self.additional_target: Optional[List[str]] = split_arg(self.additional_target)
        self.freeze_vision_tower = self.freeze_vision_tower or self.train_mm_proj_only

        assert self.finetuning_type in ["lora", "full"], "Invalid fine-tuning method."

        if self.train_mm_proj_only and self.finetuning_type != "full":
            raise ValueError("`train_mm_proj_only` is only valid for full training.")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")
