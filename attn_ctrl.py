import abc

import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.utils import logging

logger = logging.get_logger(__name__)


class AttentionControl(abc.ABC):
    def __init__(self):
        self.num_cross_att_layers = 0
        self.num_self_att_layers = 0
        self.cur_step = 0
        self.cur_attn_layer = 0
        self.cur_edit = 0

    def init_att_layers_count(self, unet: UNet2DConditionModel):
        for m in unet.modules():
            if isinstance(m, Attention):
                if m.is_cross_attention:
                    self.num_cross_att_layers += 1
                else:
                    self.num_self_att_layers += 1

    def reset(self):
        self.cur_step = 0
        self.cur_attn_layer = 0

    def step_callback(self, *args, **kwargs):
        return

    def between_steps(self):
        return

    def between_edits(self):
        self.reset()
        self.cur_edit += 1

    @property
    def num_att_layers(self):
        if self.num_cross_att_layers == 0 and self.num_self_att_layers == 0:
            logger.warning(
                "No attention layers found in the UNet."
                f"Please call `{self.__class__.__name__}.init_att_layers_count` or set `num_cross_att_layers` and `num_self_att_layers` manually."
            )
        return self.num_cross_att_layers + self.num_self_att_layers

    @abc.abstractmethod
    def forward(
        self,
        tensors: dict[str, torch.Tensor],
        is_cross: bool,
        attn_processor_name: str,
    ):
        raise NotImplementedError

    def __call__(
        self,
        tensors: dict[str, torch.Tensor],
        is_cross: bool,
        attn_processor_name: str,
    ):
        return self.forward(tensors, is_cross, attn_processor_name)

    def next_attn_layer(self):
        if self.cur_attn_layer < self.num_att_layers:  # inside the unet
            self.cur_attn_layer += 1
        if self.cur_attn_layer == self.num_att_layers:  # after the unet
            self.between_steps()
            self.cur_attn_layer = 0
            self.cur_step += 1


class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, attn_processor_name: str, **kwargs):
        return attn


class AttnControlProcessor:
    def __init__(self, attn_ctrl: AttentionControl, attn_processor_name: str):
        self.attn_ctrl = attn_ctrl
        self.attn_processor_name = attn_processor_name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        key, value = self.attn_ctrl(
            tensors={"key": key, "value": value},
            is_cross=attn.is_cross_attention,
            attn_processor_name=self.attn_processor_name,
        )

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        attention_probs = self.attn_ctrl(
            tensors={"attn": attention_probs},
            is_cross=attn.is_cross_attention,
            attn_processor_name=self.attn_processor_name,
        )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        self.attn_ctrl.next_attn_layer()

        return hidden_states


def register_attention_controller(
    unet: UNet2DConditionModel, controller: AttentionControl
):
    attn_processors = {
        name: AttnControlProcessor(attn_ctrl=controller, attn_processor_name=name)
        for name in unet.attn_processors.keys()
    }

    unet.set_attn_processor(attn_processors)  # type: ignore
    controller.init_att_layers_count(unet)
