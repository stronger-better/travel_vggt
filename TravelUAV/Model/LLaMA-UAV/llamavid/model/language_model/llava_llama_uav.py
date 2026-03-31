#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import inspect
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from llamavid.constants import WAYPOINT_LABEL_TOKEN
from llamavid.model.language_model.llama_uav import (
    CausalLMOutputWithPastUAV,
    CausalLMOutputWithPastUAVMulLoss,
    LlamaUAVForCausalLM,
    LlamaUAVModel,
)
from llamavid.model.llamavid_arch import LLaMAVIDMetaForCausalLM, LLaMAVIDMetaModel
from vggt.models.vggt import VGGT

try:
    from utils.logger import logger
except Exception:  # pragma: no cover - fallback for standalone model imports
    import logging

    logger = logging.getLogger(__name__)


def _extract_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module", "weights"):
            maybe_state_dict = checkpoint.get(key)
            if isinstance(maybe_state_dict, dict):
                return maybe_state_dict
    return checkpoint


def _strip_state_dict_prefix(state_dict, prefix):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _load_vggt_state_dict(vggt_model, checkpoint_or_state_dict):
    vggt_state_dict = _extract_checkpoint_state_dict(checkpoint_or_state_dict)
    vggt_state_dict = _strip_state_dict_prefix(vggt_state_dict, "module.")
    vggt_state_dict = _strip_state_dict_prefix(vggt_state_dict, "model.")
    return vggt_model.load_state_dict(vggt_state_dict, strict=False)


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaAttLlamaModel(LLaMAVIDMetaModel, LlamaUAVModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaAttLlamaModel, self).__init__(config)


class CosineDirectionLoss(nn.Module):
    def __init__(self):
        super(CosineDirectionLoss, self).__init__()

    def forward(self, vec1, vec2):
        cosine_sim = F.cosine_similarity(vec1, vec2, dim=-1)
        loss = 1 - cosine_sim
        return loss.mean()


class LlavaLlamaAttForCausalLM(LlamaUAVForCausalLM, LLaMAVIDMetaForCausalLM):
    config_class = LlavaConfig

    @staticmethod
    def _build_vggt_model():
        return VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
        )

    def __init__(self, config, **model_args):
        super(LlamaUAVForCausalLM, self).__init__(config)
        self.model = LlavaAttLlamaModel(config)
        self.use_angle_and_norm_loss = model_args.get("use_angle_and_norm_loss", True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.waypoint_emb = nn.Embedding(1, config.hidden_size)
        self.waypoints_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 64),
        )
        self.waypoints_output = nn.Linear(64, 4)

        self.history_preprocessor = nn.Sequential(
            nn.Linear(3, 4096 // 2),
            nn.ReLU(),
            nn.Linear(4096 // 2, 4096),
        )

        self.vggt_model = self._build_vggt_model()
        try:
            logger.info(
                "WAYPOINT_FLOW vggt_import module=%s file=%s",
                VGGT.__module__,
                inspect.getfile(VGGT),
            )
        except Exception:
            logger.info("WAYPOINT_FLOW vggt_import module=%s", VGGT.__module__)

        self.config.vggt_model_path = (
            model_args.get("vggt_model_path")
            or getattr(config, "vggt_model_path", None)
            or os.environ.get("VGGT_MODEL_PATH")
        )
        self.config.vggt_model_repo = (
            model_args.get("vggt_model_repo")
            or getattr(config, "vggt_model_repo", None)
            or os.environ.get("VGGT_MODEL_REPO")
            or "facebook/VGGT-1B"
        )
        self.config.vggt_model_url = (
            model_args.get("vggt_model_url")
            or getattr(config, "vggt_model_url", None)
            or os.environ.get("VGGT_MODEL_URL")
            or "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        )
        self.config.vggt_auto_download = model_args.get("vggt_auto_download", None)
        self._vggt_weights_loaded = False
        self._freeze_vggt_model()

        self.vggt_latent_projector = nn.Sequential(
            nn.Conv2d(2048, config.hidden_size, kernel_size=2, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=1),
        )

        self.waypoints_loss_func = torch.nn.L1Loss()
        self.angle_loss_func = CosineDirectionLoss()
        self.waypoint_loss_scale = 1.0
        self.special_token_dict = None

        self.post_init()

    def get_special_token_id(self, special_token_dict):
        self.special_token_dict = special_token_dict

    def _freeze_vggt_model(self):
        self.vggt_model.eval()
        for param in self.vggt_model.parameters():
            param.requires_grad = False

    def initialize_vggt_weights(self, force_reload: bool = False):
        has_meta_params = any(getattr(param, "is_meta", False) for param in self.vggt_model.parameters())
        if self._vggt_weights_loaded and not force_reload and not has_meta_params:
            return True

        requested_vggt_model_path = (
            getattr(self.config, "vggt_model_path", None)
            or os.environ.get("VGGT_MODEL_PATH")
        )
        requested_vggt_model_repo = (
            getattr(self.config, "vggt_model_repo", None)
            or os.environ.get("VGGT_MODEL_REPO")
            or "facebook/VGGT-1B"
        )
        requested_vggt_model_url = (
            getattr(self.config, "vggt_model_url", None)
            or os.environ.get("VGGT_MODEL_URL")
            or "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        )

        config_auto_download = getattr(self.config, "vggt_auto_download", None)
        if config_auto_download is None:
            should_auto_download_vggt = os.environ.get("VGGT_AUTO_DOWNLOAD", "1") == "1"
        else:
            should_auto_download_vggt = bool(config_auto_download)

        loaded_vggt_model = None

        if requested_vggt_model_path:
            resolved_vggt_model_path = os.path.expanduser(requested_vggt_model_path)
            if os.path.isfile(resolved_vggt_model_path):
                loaded_vggt_model = self._build_vggt_model()
                vggt_checkpoint = torch.load(resolved_vggt_model_path, map_location="cpu")
                load_result = _load_vggt_state_dict(loaded_vggt_model, vggt_checkpoint)
                logger.info(
                    "WAYPOINT_FLOW vggt_weights loaded path=%s missing=%d unexpected=%d",
                    resolved_vggt_model_path,
                    len(load_result.missing_keys),
                    len(load_result.unexpected_keys),
                )
                if load_result.missing_keys or load_result.unexpected_keys:
                    logger.warning(
                        "WAYPOINT_FLOW vggt_weights mismatch missing=%s unexpected=%s",
                        load_result.missing_keys[:8],
                        load_result.unexpected_keys[:8],
                    )
                self.config.vggt_model_path = resolved_vggt_model_path
            else:
                logger.warning(
                    "WAYPOINT_FLOW vggt_weights path_not_found path=%s",
                    resolved_vggt_model_path,
                )

        if loaded_vggt_model is None and should_auto_download_vggt:
            try:
                logger.info(
                    "WAYPOINT_FLOW vggt_weights from_pretrained repo=%s",
                    requested_vggt_model_repo,
                )
                loaded_vggt_model = VGGT.from_pretrained(
                    requested_vggt_model_repo,
                    enable_camera=False,
                    enable_point=False,
                    enable_depth=False,
                    enable_track=False,
                )
                logger.info(
                    "WAYPOINT_FLOW vggt_weights loaded_from_repo repo=%s",
                    requested_vggt_model_repo,
                )
                self.config.vggt_model_repo = requested_vggt_model_repo
            except Exception as exc:
                logger.warning(
                    "WAYPOINT_FLOW vggt_weights from_pretrained_failed repo=%s error=%s",
                    requested_vggt_model_repo,
                    exc,
                )

        if loaded_vggt_model is None and should_auto_download_vggt:
            try:
                logger.info(
                    "WAYPOINT_FLOW vggt_weights downloading_fallback url=%s",
                    requested_vggt_model_url,
                )
                loaded_vggt_model = self._build_vggt_model()
                vggt_checkpoint = torch.hub.load_state_dict_from_url(
                    requested_vggt_model_url,
                    map_location="cpu",
                )
                load_result = _load_vggt_state_dict(loaded_vggt_model, vggt_checkpoint)
                logger.info(
                    "WAYPOINT_FLOW vggt_weights downloaded_fallback missing=%d unexpected=%d",
                    len(load_result.missing_keys),
                    len(load_result.unexpected_keys),
                )
                if load_result.missing_keys or load_result.unexpected_keys:
                    logger.warning(
                        "WAYPOINT_FLOW vggt_weights mismatch missing=%s unexpected=%s",
                        load_result.missing_keys[:8],
                        load_result.unexpected_keys[:8],
                    )
                self.config.vggt_model_url = requested_vggt_model_url
            except Exception as exc:
                logger.warning(
                    "WAYPOINT_FLOW vggt_weights fallback_failed url=%s error=%s",
                    requested_vggt_model_url,
                    exc,
                )

        if loaded_vggt_model is None:
            logger.warning("WAYPOINT_FLOW vggt_weights unavailable using_random_init=True")
            self._vggt_weights_loaded = False
            self._freeze_vggt_model()
            return False

        self.vggt_model = loaded_vggt_model
        self._freeze_vggt_model()
        self._vggt_weights_loaded = True
        return True

    def get_model(self):
        return self.model

    def forward_waypoint(self, hidden_states):
        _, hidden_size = hidden_states.size()
        waypoints_feature = self.waypoints_fc(hidden_states.reshape(-1, hidden_size))
        predicted_waypoints = self.waypoints_output(waypoints_feature)
        return predicted_waypoints

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        vggt_images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        waypoints: Optional[torch.FloatTensor] = None,
        orientations: Optional[torch.FloatTensor] = None,
        historys: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        return_waypoints: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPastUAV]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not self.training:
            if images[0].device != self.device:
                if type(images) is not list:
                    images = images.to(device=self.device)
                else:
                    images = [image.to(device=self.device) for image in images]
            if input_ids.device != self.device:
                input_ids = input_ids.to(device=self.device)
            if attention_mask.device != self.device:
                attention_mask = attention_mask.to(device=self.device)
            if labels is not None and labels.device != self.device:
                labels = labels.to(device=self.device)
            if vggt_images is not None and vggt_images.device != self.device:
                vggt_images = vggt_images.to(device=self.device)

        if type(images) is not list:
            images = images.to(dtype=self.dtype)
        else:
            images = [image.to(dtype=self.dtype) for image in images]

        if vggt_images is not None:
            vggt_images = vggt_images.to(device=self.device, dtype=torch.float32)

        history_embeds = []
        for idx in range(len(historys)):
            history = historys[idx]
            info = history.view(-1, 3)
            history_embed = self.history_preprocessor(info)
            history_embeds.append(history_embed)

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            prompts=prompts,
            historys=history_embeds,
            special_token_dict=self.special_token_dict,
            vggt_images=vggt_images,
        )

        inputs_embeds = inputs_embeds.to(dtype=self.waypoint_emb.weight.dtype)
        if not torch.isfinite(inputs_embeds).all():
            inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=0.0, neginf=0.0)
        inputs_embeds[labels == WAYPOINT_LABEL_TOKEN] = self.waypoint_emb.weight

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        waypoints_feat = hidden_states[labels == WAYPOINT_LABEL_TOKEN]
        predicted_waypoints = self.forward_waypoint(waypoints_feat)

        if waypoints is None and return_waypoints:
            return predicted_waypoints

        loss = None

        assert len(torch.where(labels == WAYPOINT_LABEL_TOKEN)[0]) == waypoints.shape[0]
        if waypoints is not None:
            if self.use_angle_and_norm_loss:
                waypoint_loss = self.waypoint_loss_scale * self.waypoints_loss_func(predicted_waypoints[:, 3], waypoints[:, 3])
                angle_loss = self.waypoint_loss_scale * self.angle_loss_func(predicted_waypoints[:, :3], waypoints[:, :3])
                loss = waypoint_loss + angle_loss
            else:
                loss = self.waypoint_loss_scale * self.waypoints_loss_func(predicted_waypoints, waypoints)

        if self.training and loss is not None:
            dummy_input = torch.zeros((1, 2048, 2, 2), dtype=self.dtype, device=self.device)
            dummy_out = self.vggt_latent_projector(dummy_input)
            loss = loss + dummy_out.sum() * 0.0

        if return_waypoints:
            return loss, predicted_waypoints

        if not return_dict:
            output = (waypoints_feat,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastUAVMulLoss(
            loss=loss,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "vggt_images": kwargs.get("vggt_images", None),
            }
        )
        return model_inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaAttForCausalLM)
