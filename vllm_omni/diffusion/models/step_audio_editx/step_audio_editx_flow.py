import torch.nn as nn
from torch.nn import functional as F
import onnxruntime
import torch
from stepvocoder.cosyvoice2.flow.decoder_dit import DiT
from stepvocoder.cosyvoice2.flow.flow_matching import CausalConditionalCFM
from typing import Tuple, List
from stepvocoder.cosyvoice2.transformer.encoder_layer import ConformerEncoderLayer
from stepvocoder.cosyvoice2.transformer.positionwise_feed_forward import PositionwiseFeedForward
from stepvocoder.cosyvoice2.utils.class_utils import (
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_SUBSAMPLE_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_ACTIVATION_CLASSES,
)
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 128 

class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2, scale_factor: float = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        # In this mode, first repeat interpolate, than conv with stride=1
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)
        self.scale_factor = float(self.stride) if scale_factor is None else float(scale_factor)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride
    
    def forward_chunk(self, inputs: torch.Tensor, input_lengths: torch.Tensor, cache: torch.Tensor = torch.zeros((0, 0, 0))):
        """
        Args:
            inputs(torch.Tensor): shape (b, c, t)
            input_length(torch.Tensor): shape (b), can be None 
            cache(torch.Tensor): shape (b, c, cache_t), where cache_t = stride * 2
        """
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")
        
        if cache is None:
            cache = inputs.new_zeros(inputs.shape[0], inputs.shape[1], self.stride * 2)
        outputs = torch.cat([cache, outputs], dim=2)
        new_cache = outputs[..., -self.stride*2:]
        outputs = self.conv(outputs)

        if input_lengths is not None:
            input_lengths = input_lengths * self.stride
        return outputs, input_lengths, new_cache


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        # look ahead
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (2, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # residual connection
        outputs = outputs + inputs
        return outputs
    
    def forward_chunk(self, inputs: torch.Tensor, cache: torch.Tensor = None):
        """
        Args:
            inputs(torch.Tensor): shape (b, t, c)
            cache(torch.Tensor): shape (b, c, cache_t=2), c = channels
        """
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.leaky_relu(self.conv1(outputs))
        # the length of outputs is input length - pre_lookahead_len
        if cache is None:
            cache = outputs.new_zeros(outputs.shape[0], outputs.shape[1], 2)
        # NOTE 
        new_cache = outputs[..., -2:]
        outputs = torch.cat([cache, outputs], dim=2)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        # residual connection
        outputs = outputs + inputs[:, :-self.pre_lookahead_len]
        return outputs, new_cache


"""Customize each sample's chunk attention mask
"""
class UpsampleConformerEncoderV2(torch.nn.Module):

    def __init__(
        self,
        # input & output
        input_size: int,
        output_size: int = 256,
        input_layer: str = "linear",
        pre_lookahead_len: int = 3,
        # size
        num_blocks: int = 6,
        num_up_blocks: int = 4,
        # upsampling
        up_stride: int = 2,
        up_scale_factor: float = 2,
        # attention
        attention_heads: int = 4,
        pos_enc_layer_type: str = "rel_pos_espnet",
        selfattention_layer_type: str = "rel_selfattn",
        key_bias: bool = True,
        # mlp
        linear_units: int = 2048,
        # dropouts
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        # other
        normalize_before: bool = True,
        activation_type: str = "swish",
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size
        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](
                output_size,
                positional_dropout_rate
            ),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        self.pre_lookahead_layer = PreLookaheadLayer(
            channels=output_size, 
            pre_lookahead_len=pre_lookahead_len
        )
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args
                ),
                PositionwiseFeedForward(*positionwise_layer_args),
                None,
                None,
                dropout_rate,
                normalize_before,
            ) for _ in range(num_blocks)
        ]) 
        self.up_layer = Upsample1D(
            channels=output_size, 
            out_channels=output_size, 
            stride=up_stride, 
            scale_factor=up_scale_factor
        )
        self.up_embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](
                output_size,
                positional_dropout_rate
            ),
        )
        self.up_encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args
                ),
                PositionwiseFeedForward(*positionwise_layer_args),
                None,
                None,
                dropout_rate,
                normalize_before,
            ) for _ in range(num_up_blocks)
        ])

        self.enable_cuda_graph = False
        self.use_cuda_graph = False
        self.graph_encoder = {}
        self.graph_up_encoder = {}
        self.inference_buffers_encoder = {}
        self.inference_buffers_up_encoder = {}
        self.max_static_time = 1500
    
    # FIXME(sfy) revert hard-coded bfloat16 
    # this method is skipped in CausalMaskedDiffWithXvec.scatter_cuda_graph
    def scatter_cuda_graph(self, enable_cuda_graph: bool):
        self.enable_cuda_graph = enable_cuda_graph
        if self.enable_cuda_graph:
            self._init_cuda_graph()
    
    def _init_cuda_graph(self):
        """初始化 CUDA Graph"""

        for l in range(100, 1500, 10):
            static_x = torch.zeros((1, l, 512), 
                                dtype=torch.float32, device=torch.device('cuda'))
            static_mask = torch.ones((1, 1, l), 
                                    dtype=torch.bool, device=torch.device('cuda'))
            static_pos_emb = torch.zeros((1, 2*l-1, 512), 
                                        dtype=torch.float32, device=torch.device('cuda'))
            
            static_inputs = [
                static_x,
                static_mask,
                static_pos_emb,
            ]
            
            self._forward_impl_encoder(
                static_inputs[0],
                static_inputs[1],
                static_inputs[2],
            )
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(graph):
                    static_out_x = self._forward_impl_encoder(
                        static_inputs[0],
                        static_inputs[1],
                        static_inputs[2]
                    )
            self.graph_encoder[l] = graph
            static_outputs = [
                static_out_x,
            ]
            self.inference_buffers_encoder[l] = {
                'static_inputs': static_inputs,
                'static_outputs': static_outputs
            }

        for l in range(100, 1500, 10):
            static_x = torch.zeros((1, l, 512), 
                                dtype=torch.float32, device=torch.device('cuda'))
            static_mask = torch.ones((1, 1, l), 
                                    dtype=torch.bool, device=torch.device('cuda'))
            static_pos_emb = torch.zeros((1, 2*l-1, 512), 
                                        dtype=torch.float32, device=torch.device('cuda'))
            
            static_inputs = [
                static_x,
                static_mask,
                static_pos_emb,
            ]
            
            self._forward_impl_up_encoder(
                static_inputs[0],
                static_inputs[1],
                static_inputs[2],
            )
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(graph):
                    static_out_x = self._forward_impl_up_encoder(
                        static_inputs[0],
                        static_inputs[1],
                        static_inputs[2]
                    )
            self.graph_up_encoder[l] = graph
            static_outputs = [
                static_out_x,
            ]
            self.inference_buffers_up_encoder[l] = {
                'static_inputs': static_inputs,
                'static_outputs': static_outputs
            }

        self.use_cuda_graph = True
        print("CUDA Graph initialized successfully for encoder and up_encoder")

    # @torch.compile(dynamic=True,backend="eager")
    def _forward_impl_encoder(self,
                             x: torch.Tensor,
                             mask: torch.Tensor,
                             pos_emb: torch.Tensor):
        for layer in self.encoders:
            x, _, _, _ = layer(x, mask, pos_emb)
        return x
    
    # @torch.compile(dynamic=True,backend="eager")
    def _forward_impl_up_encoder(self,
                             x: torch.Tensor,
                             mask: torch.Tensor,
                             pos_emb: torch.Tensor):
        for layer in self.up_encoders:
            x, _, _, _ = layer(x, mask, pos_emb)
        return x
    
    def output_size(self) -> int:
        return self._output_size
    
    # @torch.compile(dynamic=True,backend="eager")
    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (sfy) chunk training strategy should not be open-sourced
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.embed(xs, masks)

        # lookahead
        xs = self.pre_lookahead_layer(xs)
        # conformer block
        if self.enable_cuda_graph and xs.shape[1] in self.graph_encoder:
            self.inference_buffers_encoder[xs.shape[1]]['static_inputs'][0].copy_(xs)
            self.inference_buffers_encoder[xs.shape[1]]['static_inputs'][1].copy_(masks)
            self.inference_buffers_encoder[xs.shape[1]]['static_inputs'][2].copy_(pos_emb)
            self.graph_encoder[xs.shape[1]].replay()    
            xs = self.inference_buffers_encoder[xs.shape[1]]['static_outputs'][0]
        else:
            xs = self._forward_impl_encoder(xs, masks, pos_emb)
        # upsample
        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        
        # 2nd conformer block
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        if self.enable_cuda_graph and xs.shape[1] in self.graph_up_encoder:
            self.inference_buffers_up_encoder[xs.shape[1]]['static_inputs'][0].copy_(xs)
            self.inference_buffers_up_encoder[xs.shape[1]]['static_inputs'][1].copy_(masks)
            self.inference_buffers_up_encoder[xs.shape[1]]['static_inputs'][2].copy_(pos_emb)
            self.graph_up_encoder[xs.shape[1]].replay()
            xs = self.inference_buffers_up_encoder[xs.shape[1]]['static_outputs'][0]
        else:
            xs = self._forward_impl_up_encoder(xs, masks, pos_emb)
        # post norm
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    @torch.compile(dynamic=True,backend="eager")
    def forward_chunk(self,
                      xs: torch.Tensor,
                      last_chunk: bool = False,
                      cnn_cache: torch.Tensor = None,
                      att_cache: torch.Tensor = None,
                      ):
        """
        Args:
            xs: shape (b, dt, c)
            last_chunk: bool. If last chunk, will pad input with lookaheads
            att_cache: shape (depth1+depth2, b, nh, 2*t1, c).
            cnn_cache: shape (b, c, t1+t2). Where t1=2 (pre_lookahead_layer), t2=4 (up_layer)
        """ 
        if att_cache is not None:
            assert att_cache.shape[3] % 2 == 0, att_cache.shape
        if cnn_cache is not None:
            assert cnn_cache.shape[2] == 2+self.up_layer.stride*2, cnn_cache.shape

        # unpack caches
        offset1 = att_cache.shape[3] // 2 if att_cache is not None else 0
        att_cache1 = att_cache[:len(self.encoders), :, :, :offset1] if att_cache is not None else [None] * len(self.encoders)
        att_cache2 = att_cache[len(self.encoders):] if att_cache is not None else [None] * len(self.encoders)
        cnn_cache1 = cnn_cache[:, :, :2] if cnn_cache is not None else None
        cnn_cache2 = cnn_cache[:, :, 2:] if cnn_cache is not None else None
        xs, _, _ = self.embed(xs, None)
        if last_chunk:
            xs = F.pad(xs, (0, 0, 0, self.pre_lookahead_layer.pre_lookahead_len))
        
        # this_cnn_cache: shape (b=1, c=512, t=2)
        xs, new_cnn_cache1 = self.pre_lookahead_layer.forward_chunk(xs, cache=cnn_cache1)

        # remake pos_emb, offset param is ignored by position_encoding
        pos_emb = self.embed.position_encoding(offset=None, size=offset1 + xs.shape[1])

        # first conformer 
        chunk_masks = torch.zeros((0, 0, 0))
        new_att_cache1 = []

        for idx, layer in enumerate(self.encoders):
            # this_att_cache: shape (b, nh, t, c * 2)
            xs, _, this_new_att_cache1, _ = layer(xs, chunk_masks, pos_emb, att_cache=att_cache1[idx])
            new_att_cache1.append(this_new_att_cache1)
        new_att_cache1 = torch.stack(new_att_cache1, dim=0)

        # upsample + conformer encoder, xs: (b, t, c) -> (b, c, t)
        xs = xs.transpose(1, 2).contiguous()
        # this_cnn_cache: shape (b=1, c=512, t=2*2)
        xs, _, new_cnn_cache2 = self.up_layer.forward_chunk(xs, None, cache=cnn_cache2)
        xs = xs.transpose(1, 2).contiguous()

        # at this time, xs are doubled in length
        xs, _, _ = self.up_embed(xs, None)

        # remake pos_emb
        pos_emb = self.embed.position_encoding(offset=None, size=offset1 * self.up_layer.stride + xs.shape[1])

        # second conformer
        chunk_masks = torch.zeros((0, 0, 0),dtype=torch.bfloat16)
        new_att_cache2 = []

        for idx, layer in enumerate(self.up_encoders):
            xs, _, this_new_att_cache2, _ = layer(xs, chunk_masks, pos_emb, att_cache=att_cache2[idx])
            new_att_cache2.append(this_new_att_cache2)
        new_att_cache2 = torch.stack(new_att_cache2, dim=0)

        if self.normalize_before:
            xs = self.after_norm(xs)
        
        # pack new cache
        new_att_cache = torch.cat([new_att_cache1.repeat(1, 1, 1, 2, 1), new_att_cache2], dim=0)
        new_cnn_cache = torch.cat([new_cnn_cache1, new_cnn_cache2], dim=2)

        return xs, new_cnn_cache, new_att_cache



class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 5121,
                 encoder: UpsampleConformerEncoderV2 = None,
                 decoder: CausalConditionalCFM = None,
                 input_embedding: torch.nn.Module = None,
                 ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.pre_lookahead_len = int(encoder.pre_lookahead_layer.pre_lookahead_len)
        self.up_rate = int(encoder.up_layer.stride)
        if input_embedding is None:
            self.input_embedding = nn.Embedding(vocab_size, input_size)
        else:
            self.input_embedding = input_embedding
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder

        # xvec projection with CUDA Graph optimization
        # 初始化 CUDA Graph 相关变量
        self.enable_cuda_graph = False
        self.static_embedding = None
        self.static_output = None
        self.graph = None
        self.embedding_shape = None

    def scatter_cuda_graph(self, enable_cuda_graph: bool):
        self.enable_cuda_graph = enable_cuda_graph
        if self.enable_cuda_graph:
            # self.encoder.scatter_cuda_graph(enable_cuda_graph)
            self.decoder.scatter_cuda_graph(enable_cuda_graph)

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  n_timesteps: int = 10,
                  ):
        assert token.shape[0] == 1

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
    
        # concat text and prompt_text
        token_len = prompt_token_len + token_len
        token = torch.concat([prompt_token, token], dim=1)
        
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # token encode
        h, _ = self.encoder.forward(token, token_len)
        h = self.encoder_proj(h)

        # condition
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]

        conds = torch.zeros_like(h)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2).contiguous()

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)

        feat = self.decoder.forward(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
        )

        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat

    @torch.inference_mode()
    def setup_cache(self, 
                    token: torch.Tensor, 
                    mel: torch.Tensor, 
                    spk: torch.Tensor, 
                    n_timesteps: int = 10,
                    ):
        """
        Args:
            token: shape (b, t), with look ahead tokens
            mel: shape (b, t, c), groundtruth mel
            spk: shape (b, 192), speaker embedding
        Returns:
            cache: dict {
                'conformer': {'cnn_cache': xxx, 'att_cache': xxx}, 
                'estimator': {'cnn_cache': xxx, 'att_cache': xxx}
            }
        """
        # check if look ahead token included
        assert (token.shape[1] - self.pre_lookahead_len) * self.up_rate == mel.shape[1], (token.shape, mel.shape)

        # xvec projection
        spk = F.normalize(spk, dim=1)
        spk = self.spk_embed_affine_layer(spk)

        token = self.input_embedding(token)
        # NOTE encoder.forward_chunk will strip the look ahead part
        h, conformer_cnn_cache, conformer_att_cache = self.encoder.forward_chunk(
            xs = token,
            last_chunk = False,
            cnn_cache = None,
            att_cache = None,
        )
        h = self.encoder_proj(h)

        feat, estimator_cnn_cache, estimator_att_cache = self.decoder.forward_chunk(
            mu = h.transpose(1, 2).contiguous(),
            spks = spk,
            cond = mel.transpose(1, 2).contiguous(),
            n_timesteps = n_timesteps,
            temperature = 1.0,
            cnn_cache = None,
            att_cache = None,
        )

        cache = {
            'conformer_cnn_cache': conformer_cnn_cache,
            'conformer_att_cache': conformer_att_cache,
            'estimator_cnn_cache': estimator_cnn_cache,
            'estimator_att_cache': estimator_att_cache,
        }
        return cache

    @torch.inference_mode()
    def inference_chunk(self, 
                        token: torch.Tensor,
                        spk: torch.Tensor, 
                        cache: dict,
                        last_chunk: bool = False,
                        n_timesteps: int = 10,
                        ):
        """
        Args:
            token: shape (b, t), with look ahead tokens
            spk: shape (b, 192), speaker embedding
            cache: dict {
                'conformer_cnn_cache': xxx,
                ...
            }
        """
        # unpack cache
        conformer_cnn_cache = cache['conformer_cnn_cache']
        conformer_att_cache = cache['conformer_att_cache']
        estimator_cnn_cache = cache['estimator_cnn_cache']
        estimator_att_cache = cache['estimator_att_cache']

        # xvec projection
        spk = F.normalize(spk, dim=1)
        spk = self.spk_embed_affine_layer(spk)

        token = self.input_embedding(token)
        # if not the last chunk, h is shorter than xs for a length of lookahead_length * stride (6)
        h, conformer_cnn_cache, conformer_att_cache = self.encoder.forward_chunk(
            xs = token,
            last_chunk = last_chunk,
            cnn_cache = conformer_cnn_cache,
            att_cache = conformer_att_cache,
        )
        h = self.encoder_proj(h)

        cond = torch.zeros_like(h)
        # forward estimator
        feat, estimator_cnn_cache, estimator_att_cache = self.decoder.forward_chunk(
            mu = h.transpose(1, 2).contiguous(),
            spks = spk,
            cond = cond.transpose(1, 2).contiguous(),
            n_timesteps = n_timesteps,
            temperature = 1.0,
            cnn_cache = estimator_cnn_cache,
            att_cache = estimator_att_cache,
        )


        new_cache = {
            'conformer_cnn_cache': conformer_cnn_cache,
            'conformer_att_cache': conformer_att_cache,
            'estimator_cnn_cache': estimator_cnn_cache,
            'estimator_att_cache': estimator_att_cache,
        }

        return feat, new_cache



def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask



"""
Inference wrapper
"""
class CausalConditionalCFM(torch.nn.Module):
    def __init__(self, estimator: DiT, inference_cfg_rate:float=0.7):
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.out_channels = estimator.out_channels
         # a maximum of 600s
        self.register_buffer('rand_noise', torch.randn([1, self.out_channels, 50 * 600]), persistent=False)

        self.register_buffer('cnn_cache_buffer', torch.zeros(16, 16, 2, 1024, 2), persistent=False)
        self.register_buffer('att_cache_buffer', torch.zeros(16, 16, 2, 8, 1000, 128), persistent=False)

    def scatter_cuda_graph(self, enable_cuda_graph: bool):
        if enable_cuda_graph:
            self.estimator._init_cuda_graph_all()

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        assert self.inference_cfg_rate > 0, 'inference_cfg_rate better > 0'

        # constant during denoising
        mask_in = torch.cat([mask, mask], dim=0)
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        
        for step in range(1, len(t_span)):

            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)

            dphi_dt = self.estimator.forward(
                x_in,
                mask_in,
                mu_in,
                t_in,
                spks_in,
                cond_in,
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x

    @torch.inference_mode()
    def forward(self, mu, mask, spks, cond, n_timesteps=10, temperature=1.0):
        z = self.rand_noise[:, :, :mu.size(2)] * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        # cosine scheduling
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span, mu, mask, spks, cond)

    def solve_euler_chunk(self, 
                          x:torch.Tensor, 
                          t_span:torch.Tensor, 
                          mu:torch.Tensor, 
                          spks:torch.Tensor, 
                          cond:torch.Tensor, 
                          cnn_cache:torch.Tensor=None,
                          att_cache:torch.Tensor=None,
                          ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
            cnn_cache: shape (n_time, depth, b, c1+c2, 2)
            att_cache: shape (n_time, depth, b, nh, t, c * 2)
        """
        assert self.inference_cfg_rate > 0, 'cfg rate should be > 0'
        
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)  # (b,)

        # setup initial cache
        if cnn_cache is None:
            cnn_cache = [None for _ in range(len(t_span)-1)]
        if att_cache is None:
            att_cache = [None for _ in range(len(t_span)-1)]
        # next chunk's cache at each timestep

        if att_cache[0] is not None:
            last_att_len = att_cache.shape[4]
        else:
            last_att_len = 0

        # constant during denoising
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        for step in range(1, len(t_span)):
            # torch.cuda.memory._record_memory_history(max_entries=100000)
            # torch.cuda.memory._record_memory_history(max_entries=100000)
            this_att_cache = att_cache[step-1]
            this_cnn_cache = cnn_cache[step-1]

            dphi_dt, this_new_cnn_cache, this_new_att_cache = self.estimator.forward_chunk(
                x = x.repeat(2, 1, 1),
                mu = mu_in,
                t = t.repeat(2),
                spks = spks_in,
                cond = cond_in,
                cnn_cache = this_cnn_cache,
                att_cache = this_att_cache,
            )
            dphi_dt, cfg_dphi_dt = dphi_dt.chunk(2, dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

            self.cnn_cache_buffer[step-1] = this_new_cnn_cache
            self.att_cache_buffer[step-1][:, :, :, :x.shape[2]+last_att_len, :] = this_new_att_cache
        
        cnn_cache = self.cnn_cache_buffer
        att_cache = self.att_cache_buffer[:, :, :, :, :x.shape[2]+last_att_len, :]
        return x, cnn_cache, att_cache
    
    @torch.inference_mode()
    def forward_chunk(self, 
                      mu:torch.Tensor, 
                      spks:torch.Tensor, 
                      cond:torch.Tensor,
                      n_timesteps:int=10, 
                      temperature:float=1.0, 
                      cnn_cache:torch.Tensor=None,
                      att_cache:torch.Tensor=None,
                      ):
        """
        Args:
            mu(torch.Tensor): shape (b, c, t)
            spks(torch.Tensor): shape (b, 192)
            cond(torch.Tensor): shape (b, c, t)
            cnn_cache: shape (n_time, depth, b, c1+c2, 2)
            att_cache: shape (n_time, depth, b, nh, t, c * 2)
        """
        # get offset from att_cache
        offset = att_cache.shape[4] if att_cache is not None else 0
        z = self.rand_noise[:, :, offset:offset+mu.size(2)] * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        # cosine scheduling
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        x, new_cnn_cache, new_att_cache = self.solve_euler_chunk(
            x=z,
            t_span=t_span,
            mu=mu,
            spks=spks,
            cond=cond,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
        )
        return x, new_cnn_cache, new_att_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 5121,
                 encoder: UpsampleConformerEncoderV2 = None,
                 decoder: CausalConditionalCFM = None,
                 input_embedding: torch.nn.Module = None,
                 ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.pre_lookahead_len = int(encoder.pre_lookahead_layer.pre_lookahead_len)
        self.up_rate = int(encoder.up_layer.stride)
        if input_embedding is None:
            self.input_embedding = nn.Embedding(vocab_size, input_size)
        else:
            self.input_embedding = input_embedding
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder

        # xvec projection with CUDA Graph optimization
        # 初始化 CUDA Graph 相关变量
        self.enable_cuda_graph = False
        self.static_embedding = None
        self.static_output = None
        self.graph = None
        self.embedding_shape = None

    def scatter_cuda_graph(self, enable_cuda_graph: bool):
        self.enable_cuda_graph = enable_cuda_graph
        if self.enable_cuda_graph:
            # self.encoder.scatter_cuda_graph(enable_cuda_graph)
            self.decoder.scatter_cuda_graph(enable_cuda_graph)

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  n_timesteps: int = 10,
                  ):
        assert token.shape[0] == 1

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
    
        # concat text and prompt_text
        token_len = prompt_token_len + token_len
        token = torch.concat([prompt_token, token], dim=1)
        
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # token encode
        h, _ = self.encoder.forward(token, token_len)
        h = self.encoder_proj(h)

        # condition
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]

        conds = torch.zeros_like(h)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2).contiguous()

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)

        feat = self.decoder.forward(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
        )

        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat

    @torch.inference_mode()
    def setup_cache(self, 
                    token: torch.Tensor, 
                    mel: torch.Tensor, 
                    spk: torch.Tensor, 
                    n_timesteps: int = 10,
                    ):
        """
        Args:
            token: shape (b, t), with look ahead tokens
            mel: shape (b, t, c), groundtruth mel
            spk: shape (b, 192), speaker embedding
        Returns:
            cache: dict {
                'conformer': {'cnn_cache': xxx, 'att_cache': xxx}, 
                'estimator': {'cnn_cache': xxx, 'att_cache': xxx}
            }
        """
        # check if look ahead token included
        assert (token.shape[1] - self.pre_lookahead_len) * self.up_rate == mel.shape[1], (token.shape, mel.shape)

        # xvec projection
        spk = F.normalize(spk, dim=1)
        spk = self.spk_embed_affine_layer(spk)

        token = self.input_embedding(token)
        # NOTE encoder.forward_chunk will strip the look ahead part
        h, conformer_cnn_cache, conformer_att_cache = self.encoder.forward_chunk(
            xs = token,
            last_chunk = False,
            cnn_cache = None,
            att_cache = None,
        )
        h = self.encoder_proj(h)

        feat, estimator_cnn_cache, estimator_att_cache = self.decoder.forward_chunk(
            mu = h.transpose(1, 2).contiguous(),
            spks = spk,
            cond = mel.transpose(1, 2).contiguous(),
            n_timesteps = n_timesteps,
            temperature = 1.0,
            cnn_cache = None,
            att_cache = None,
        )

        cache = {
            'conformer_cnn_cache': conformer_cnn_cache,
            'conformer_att_cache': conformer_att_cache,
            'estimator_cnn_cache': estimator_cnn_cache,
            'estimator_att_cache': estimator_att_cache,
        }
        return cache

    @torch.inference_mode()
    def inference_chunk(self, 
                        token: torch.Tensor,
                        spk: torch.Tensor, 
                        cache: dict,
                        last_chunk: bool = False,
                        n_timesteps: int = 10,
                        ):
        """
        Args:
            token: shape (b, t), with look ahead tokens
            spk: shape (b, 192), speaker embedding
            cache: dict {
                'conformer_cnn_cache': xxx,
                ...
            }
        """
        # unpack cache
        conformer_cnn_cache = cache['conformer_cnn_cache']
        conformer_att_cache = cache['conformer_att_cache']
        estimator_cnn_cache = cache['estimator_cnn_cache']
        estimator_att_cache = cache['estimator_att_cache']

        # xvec projection
        spk = F.normalize(spk, dim=1)
        spk = self.spk_embed_affine_layer(spk)

        token = self.input_embedding(token)
        # if not the last chunk, h is shorter than xs for a length of lookahead_length * stride (6)
        h, conformer_cnn_cache, conformer_att_cache = self.encoder.forward_chunk(
            xs = token,
            last_chunk = last_chunk,
            cnn_cache = conformer_cnn_cache,
            att_cache = conformer_att_cache,
        )
        h = self.encoder_proj(h)

        cond = torch.zeros_like(h)
        # forward estimator
        feat, estimator_cnn_cache, estimator_att_cache = self.decoder.forward_chunk(
            mu = h.transpose(1, 2).contiguous(),
            spks = spk,
            cond = cond.transpose(1, 2).contiguous(),
            n_timesteps = n_timesteps,
            temperature = 1.0,
            cnn_cache = estimator_cnn_cache,
            att_cache = estimator_att_cache,
        )


        new_cache = {
            'conformer_cnn_cache': conformer_cnn_cache,
            'conformer_att_cache': conformer_att_cache,
            'estimator_cnn_cache': estimator_cnn_cache,
            'estimator_att_cache': estimator_att_cache,
        }

        return feat, new_cache
