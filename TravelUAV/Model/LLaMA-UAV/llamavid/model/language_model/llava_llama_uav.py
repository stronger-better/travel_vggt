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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from llamavid.constants import WAYPOINT_LABEL_TOKEN
from llamavid.model.feature_fusion import (
    FeatureFusionConfig,
    FeatureFusionModule,
    GeometryFeatureMerger,
    VGGTGeometryEncoder,
)
from llamavid.model.language_model.llama_uav import (
    CausalLMOutputWithPastUAV,
    CausalLMOutputWithPastUAVMulLoss,
    LlamaUAVForCausalLM,
    LlamaUAVModel,
)
from llamavid.model.llamavid_arch import LLaMAVIDMetaForCausalLM, LLaMAVIDMetaModel


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
            nn.Linear(3, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )

        fusion_method = model_args.get("feature_fusion_method", "gated")
        fusion_heads = model_args.get("fusion_attention_heads", max(1, config.num_attention_heads // 4))
        fusion_layers = model_args.get("fusion_num_layers", 1)
        fusion_dropout = model_args.get("fusion_dropout", 0.1)
        self.geometry_merge_size = int(model_args.get("geometry_merge_size", 4))

        self.geometry_encoder = VGGTGeometryEncoder()
        self.geometry_merger = GeometryFeatureMerger(
            output_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            context_dim=self.geometry_encoder.feature_dim,
            spatial_merge_size=self.geometry_merge_size,
            merger_type="mlp",
        )
        self.feature_fusion = FeatureFusionModule(
            FeatureFusionConfig(
                fusion_method=fusion_method,
                hidden_size=config.hidden_size,
                num_heads=fusion_heads,
                dropout=fusion_dropout,
                num_layers=fusion_layers,
            )
        )
        self.vggt_model = None
        self.vggt_latent_projector = None

        self.waypoints_loss_func = torch.nn.L1Loss()
        self.angle_loss_func = CosineDirectionLoss()
        self.waypoint_loss_scale = 1.0
        self.special_token_dict = None

        self.post_init()

    def get_special_token_id(self, special_token_dict):
        self.special_token_dict = special_token_dict

    def get_model(self):
        return self.model

    @staticmethod
    def _infer_image_token_layout(total_tokens: int):
        side_no_ctx = int(total_tokens ** 0.5)
        if side_no_ctx * side_no_ctx == total_tokens:
            return side_no_ctx, False

        side_with_ctx = int((total_tokens - 1) ** 0.5)
        if total_tokens > 1 and side_with_ctx * side_with_ctx == (total_tokens - 1):
            return side_with_ctx, True

        return None, None

    def fuse_current_image_features_with_geometry(self, current_image_feature, sample_vggt_images):
        if sample_vggt_images is None:
            return current_image_feature

        if sample_vggt_images.dim() == 3:
            sample_vggt_images = sample_vggt_images.unsqueeze(0)

        if sample_vggt_images.shape[0] == 0 or current_image_feature.shape[0] == 0:
            return current_image_feature

        n_image = sample_vggt_images.shape[0]
        total_tokens = current_image_feature.shape[0]
        if total_tokens % n_image != 0:
            return current_image_feature

        token_per_image = total_tokens // n_image
        vis_side, has_ctx = self._infer_image_token_layout(token_per_image)
        if vis_side is None:
            return current_image_feature

        image_tokens = current_image_feature.reshape(n_image, token_per_image, -1)
        if has_ctx:
            ctx_tokens = image_tokens[:, :1, :]
            vis_tokens = image_tokens[:, 1:, :]
        else:
            ctx_tokens = None
            vis_tokens = image_tokens

        geometry_tokens = self.geometry_encoder.encode(sample_vggt_images)
        geometry_tokens = geometry_tokens.to(device=current_image_feature.device, dtype=current_image_feature.dtype)
        geo_token_num = geometry_tokens.shape[1]
        geo_side = int(geo_token_num ** 0.5)
        if geo_side * geo_side != geo_token_num:
            return current_image_feature

        geometry_grid = geometry_tokens.reshape(n_image, geo_side, geo_side, -1)
        if geo_side // vis_side != self.geometry_merge_size:
            return current_image_feature

        merged_geometry = self.geometry_merger(geometry_grid, target_hw=(vis_side, vis_side))
        vis_grid = vis_tokens.reshape(n_image, vis_side, vis_side, -1)
        fused_vis_grid = self.feature_fusion(vis_grid, merged_geometry)
        fused_vis_tokens = fused_vis_grid.reshape(n_image, vis_side * vis_side, -1)

        if has_ctx:
            fused_tokens = torch.cat([ctx_tokens, fused_vis_tokens], dim=1)
        else:
            fused_tokens = fused_vis_tokens
        return fused_tokens.reshape(total_tokens, -1)

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
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not self.training:
            if isinstance(images, list):
                images = [image.to(device=self.device) for image in images]
            elif images is not None:
                images = images.to(device=self.device)

            if input_ids is not None and input_ids.device != self.device:
                input_ids = input_ids.to(device=self.device)
            if attention_mask is not None and attention_mask.device != self.device:
                attention_mask = attention_mask.to(device=self.device)
            if labels is not None and labels.device != self.device:
                labels = labels.to(device=self.device)
            if vggt_images is not None and vggt_images.device != self.device:
                vggt_images = vggt_images.to(device=self.device)

        if isinstance(images, list):
            images = [image.to(dtype=self.dtype) for image in images]
        elif images is not None:
            images = images.to(dtype=self.dtype)

        if vggt_images is not None:
            vggt_images = vggt_images.to(dtype=self.dtype)

        history_embeds = []
        for history in historys:
            info = history.view(-1, 3)
            history_embeds.append(self.history_preprocessor(info))

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

        if return_waypoints:
            return loss, predicted_waypoints

        if not return_dict:
            output = (waypoints_feat,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastUAVMulLoss(loss=loss)

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
