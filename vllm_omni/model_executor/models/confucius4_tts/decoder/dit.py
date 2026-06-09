from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fused_add_tanh_sigmoid_multiply(input_a: torch.Tensor, input_b: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Fused gated activation: tanh(x) * sigmoid(x).

    Args:
        input_a: First input tensor
        input_b: Second input tensor (added to input_a)
        n_channels: Number of channels (split point for tanh/sigmoid)

    Returns:
        Gated activation output
    """
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act


class WeightNormConv1d(nn.Module):
    """1D convolution with weight normalization.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        padding: Padding size
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply weight-normalized convolution.

        Args:
            x: Input tensor, shape (B, C, T)

        Returns:
            Output tensor, shape (B, out_channels, T)
        """
        return self.conv(x)


class WN(nn.Module):
    """WaveNet-style gated residual network with dilated convolutions.

    Stacks dilated convolution layers with gated activations and residual connections.
    Used as the final layer in DiT for mel-spectrogram prediction.

    Args:
        hidden_channels: Number of hidden channels
        kernel_size: Convolution kernel size
        dilation_rate: Base dilation rate (exponentially increases per layer)
        n_layers: Number of WaveNet layers
        gin_channels: Global conditioning dimension (timestep embedding)
        p_dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.drop = nn.Dropout(p_dropout)
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.cond_layer = WeightNormConv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.in_layers.append(
                WeightNormConv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding)
            )
            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(WeightNormConv1d(hidden_channels, res_skip_channels, 1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """WaveNet forward pass with gated convolutions and residual connections.

        Args:
            x: Input tensor, shape (B, hidden_channels, T)
            x_mask: Padding mask, shape (B, 1, T)
            g: Global conditioning (timestep embedding), shape (B, gin_channels, 1)

        Returns:
            Output tensor, shape (B, hidden_channels, T)
        """
        output = torch.zeros_like(x)
        g = self.cond_layer(g)  # Project conditioning to all layers
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            cond_offset = i * 2 * self.hidden_channels
            g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            # Gated activation: tanh(x) * sigmoid(x)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                # Residual connection + skip connection
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                # Final layer: skip connection only
                output = output + res_skip_acts
        return output * x_mask


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient alternative to LayerNorm without mean centering.

    Args:
        dim: Input dimension
        eps: Small epsilon for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization with learnable scale.

        Args:
            x: Input tensor, shape (..., dim)

        Returns:
            Normalized tensor, shape (..., dim)
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embedding.

    Applies RMSNorm then modulates with learned scale and shift from conditioning.

    Args:
        hidden_size: Hidden dimension
        eps: Epsilon for normalization
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.modulation = nn.Linear(hidden_size, 2 * hidden_size)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply adaptive normalization.

        Args:
            x: Input tensor, shape (B, T, hidden_size)
            cond: Conditioning vector (timestep embedding), shape (B, hidden_size)

        Returns:
            Modulated tensor, shape (B, T, hidden_size)
        """
        weight, bias = torch.split(self.modulation(cond), self.norm.weight.shape[0], dim=-1)
        return self.norm(x) * weight.unsqueeze(1) + bias.unsqueeze(1)


class FeedForward(nn.Module):
    """SwiGLU feedforward network.

    Uses gated linear units with SiLU activation for improved expressiveness.

    Args:
        dim: Input/output dimension
        intermediate_size: Hidden layer dimension
    """

    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: (SiLU(W1(x)) * W3(x)) @ W2.

        Args:
            x: Input tensor, shape (..., dim)

        Returns:
            Output tensor, shape (..., dim)
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Precompute rotary position embedding frequencies.

    Args:
        seq_len: Maximum sequence length
        n_elem: Dimension per head
        base: Base for frequency computation
        dtype: Output data type

    Returns:
        Precomputed frequencies, shape (seq_len, n_elem // 2, 2)
    """
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to query/key tensors.

    Args:
        x: Input tensor, shape (B, T, num_heads, head_dim)
        freqs_cis: Precomputed frequencies, shape (T, head_dim // 2, 2)

    Returns:
        Tensor with rotary embeddings applied, shape (B, T, num_heads, head_dim)
    """
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class Attention(nn.Module):
    """Multi-head self-attention with rotary position embeddings.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wqkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Multi-head self-attention with rotary embeddings.

        Args:
            x: Input tensor, shape (B, T, dim)
            attn_mask: Attention mask, shape (B, 1, 1, T)
            freqs_cis: Rotary frequencies, shape (T, head_dim // 2, 2)

        Returns:
            Attention output, shape (B, T, dim)
        """
        bsz, seqlen, dim = x.shape
        q, k, v = self.wqkv(x).chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim)

        # Apply rotary position embeddings to Q and K
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.wo(y)


class DiTBlock(nn.Module):
    """DiT transformer block with adaptive normalization and optional skip connections.

    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        intermediate_size: Feedforward intermediate dimension
    """

    def __init__(self, dim: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attention = Attention(dim, num_heads)
        self.feed_forward = FeedForward(dim, intermediate_size)
        self.attention_norm = AdaptiveLayerNorm(dim)
        self.ffn_norm = AdaptiveLayerNorm(dim)
        self.skip_in_linear = nn.Linear(dim * 2, dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        skip_in_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DiT block forward with adaptive norm and optional skip connection.

        Args:
            x: Input tensor, shape (B, T, dim)
            cond: Timestep conditioning, shape (B, dim)
            attn_mask: Attention mask, shape (B, 1, 1, T)
            freqs_cis: Rotary frequencies, shape (T, head_dim // 2, 2)
            skip_in_x: Optional U-Net skip input, shape (B, T, dim)

        Returns:
            Output tensor, shape (B, T, dim)
        """
        if skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention(self.attention_norm(x, cond), attn_mask, freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h, cond))


class FinalLayer(nn.Module):
    """Final output layer with adaptive normalization.

    Args:
        hidden_size: Hidden dimension
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply adaptive normalization and final projection.

        Args:
            x: Input tensor, shape (B, T, hidden_size)
            c: Conditioning vector (timestep embedding), shape (B, hidden_size)

        Returns:
            Output tensor, shape (B, T, hidden_size)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class SinusPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timestep encoding.

    Args:
        dim: Embedding dimension
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        """Generate sinusoidal embeddings.

        Args:
            x: Input timestamps, shape (B,)
            scale: Scaling factor for frequencies

        Returns:
            Position embeddings, shape (B, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding with sinusoidal encoding and MLP projection.

    Args:
        dim: Output embedding dimension
        freq_embed_dim: Frequency embedding dimension
    """

    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float[b]):  # noqa: F821
        """Embed diffusion timestep.

        Args:
            timestep: Diffusion time, shape (B,)

        Returns:
            Timestep embedding, shape (B, dim)
        """
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # (B, dim)
        return time


class InputEmbedding(nn.Module):
    """Input embedding layer combining mel, conditioning, and speaker features.

    Projects and concatenates:
    - Noisy mel-spectrogram
    - Reference mel conditioning
    - Semantic/text conditioning (mu)
    - Speaker style embedding (optional)

    Args:
        mel_dim: Mel-spectrogram dimension
        cond_dim: Conditioning vector dimension
        out_dim: Output embedding dimension
        spk_dim: Speaker embedding dimension (0 to disable)
    """

    def __init__(self, mel_dim: int, cond_dim: int, out_dim: int, spk_dim: int = 0):
        super().__init__()
        self.spk_dim = spk_dim
        self.mu_projection = nn.Linear(cond_dim, out_dim, bias=True)
        self.proj = nn.Linear(out_dim + mel_dim * 2 + spk_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
    ) -> torch.Tensor:
        """Combine and project input features.

        Args:
            x: Noisy mel-spectrogram, shape (B, T, mel_dim)
            cond: Reference mel conditioning, shape (B, T, mel_dim)
            mu: Semantic/text conditioning, shape (B, T, cond_dim)
            spks: Speaker embedding, shape (B, spk_dim)

        Returns:
            Combined embeddings, shape (B, T, out_dim)
        """
        mu_proj = self.mu_projection(mu)
        to_cat = [x, cond, mu_proj]
        if self.spk_dim > 0:
            spks_seq = spks.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, T, spk_dim)
            to_cat.append(spks_seq)
        return self.proj(torch.cat(to_cat, dim=-1))


class DiT(nn.Module):
    """Diffusion Transformer for velocity prediction in flow matching.

    Architecture:
    1. Input embedding (mel + conditioning + speaker)
    2. Timestep embedding for diffusion time
    3. Transformer blocks with rotary position embeddings
    4. Optional U-Net-style skip connections
    5. Final layer (WaveNet or MLP)

    Args:
        hidden_dim: Transformer hidden dimension
        num_heads: Number of attention heads
        depth: Number of transformer blocks
        mel_dim: Mel-spectrogram dimension
        mu_dim: Conditioning vector dimension
        spk_dim: Speaker embedding dimension
        long_skip_connection: Enable U-Net skip connections
        max_seq_len: Maximum sequence length for rotary embeddings
        ff_intermediate_size: Feedforward intermediate size (default: hidden_dim * 4)
        final_layer: "wavenet" or "mlp"
        wavenet_*: WaveNet final layer parameters
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        depth: int = 13,
        mel_dim: int = 80,
        mu_dim: int = 512,
        spk_dim: int = 192,
        long_skip_connection: bool = True,
        max_seq_len: int = 4096,
        ff_intermediate_size: int | None = None,
        final_layer: str = "wavenet",
        wavenet_hidden_dim: int = 512,
        wavenet_kernel_size: int = 5,
        wavenet_dilation_rate: int = 1,
        wavenet_num_layers: int = 8,
        wavenet_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_channels = mel_dim
        self.mu_dim = mu_dim
        self.spk_dim = spk_dim
        self.final_layer_type = final_layer
        self.depth = depth
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_embed = InputEmbedding(mel_dim, mu_dim, hidden_dim, spk_dim)
        self.t_embedder = TimestepEmbedding(hidden_dim)
        self.skip_linear = nn.Linear(hidden_dim + mel_dim, hidden_dim) if long_skip_connection else None

        # Transformer blocks with U-Net skip connections
        intermediate_size = ff_intermediate_size if ff_intermediate_size is not None else hidden_dim * 4
        self._emit_skip = set(i for i in range(depth) if i < depth // 2) if long_skip_connection else set()
        self._receive_skip = set(i for i in range(depth) if i > depth // 2) if long_skip_connection else set()
        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, intermediate_size) for _ in range(depth)]
        )
        self.transformer_norm = AdaptiveLayerNorm(hidden_dim)

        # Rotary position embeddings
        head_dim = hidden_dim // num_heads
        freqs_cis = precompute_freqs_cis(max_seq_len, head_dim, base=10000, dtype=torch.float32)
        self.register_buffer("freqs_cis", freqs_cis)

        # Final projection layer
        if self.final_layer_type == "wavenet":
            self.t_embedder2 = TimestepEmbedding(wavenet_hidden_dim)
            self.conv1 = nn.Linear(hidden_dim, wavenet_hidden_dim)
            self.wavenet = WN(
                hidden_channels=wavenet_hidden_dim,
                kernel_size=wavenet_kernel_size,
                dilation_rate=wavenet_dilation_rate,
                n_layers=wavenet_num_layers,
                gin_channels=wavenet_hidden_dim,
                p_dropout=wavenet_dropout,
            )
            self.final_layer = FinalLayer(wavenet_hidden_dim)
            self.res_projection = nn.Linear(hidden_dim, wavenet_hidden_dim)
            self.conv2 = nn.Conv1d(wavenet_hidden_dim, mel_dim, 1)
        else:
            self.final_mlp = nn.Linear(hidden_dim, mel_dim)

    def initialize_weights(self):
        """Initialize model weights with Kaiming normal for Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass predicting velocity field for flow matching.

        Args:
            x: Noisy mel-spectrogram, shape (B, mel_dim, T)
            mask: Padding mask, shape (B, 1, T)
            mu: Semantic/text conditioning, shape (B, T, cond_dim)
            t: Diffusion timestep, shape (B,)
            spks: Speaker embedding, shape (B, spk_dim) or None
            cond: Reference mel conditioning, shape (B, mel_dim, T) or None

        Returns:
            Predicted velocity, shape (B, mel_dim, T)
        """
        if cond is None:
            cond = torch.zeros_like(x)
        if spks is None:
            spks = torch.zeros(x.size(0), self.spk_dim, device=x.device, dtype=x.dtype)

        assert x.dim() == 3, f"Expected 3D tensor, got {x.dim()}D"
        assert x.size(1) == self.in_channels, f"Expected {self.in_channels} channels, got {x.size(1)}"
        assert x.shape == cond.shape, "x and cond must have the same shape"

        # Transpose to (B, T, C) for transformer
        x = x.transpose(1, 2)
        cond = cond.transpose(1, 2)
        bsz, seq_len, _ = x.shape
        t1 = self.t_embedder(t)  # Timestep embedding
        x_in = self.input_embed(x, cond, mu, spks)

        attn_mask = mask.view(bsz, 1, 1, seq_len)

        freqs_cis = self.freqs_cis[:seq_len]

        # Transformer blocks with U-Net skip connections
        skip_stack: list[torch.Tensor] = []
        h = x_in
        for idx, block in enumerate(self.transformer_blocks):
            skip_in = skip_stack.pop(-1) if idx in self._receive_skip and skip_stack else None
            h = block(h, t1, attn_mask, freqs_cis, skip_in)
            if idx in self._emit_skip:
                skip_stack.append(h)
        x_res = self.transformer_norm(h, t1)

        # Long skip connection from input
        if self.skip_linear is not None:
            x_res = self.skip_linear(torch.cat([x_res, x], dim=-1))

        # Final projection layer
        if self.final_layer_type == "wavenet":
            # WaveNet-based final layer
            x_out = self.conv1(x_res).transpose(1, 2)
            t2 = self.t_embedder2(t)
            x_mask = mask.unsqueeze(1).to(x_out.dtype)
            x_out = self.wavenet(x_out, x_mask, g=t2.unsqueeze(2))
            x_out = x_out.transpose(1, 2) + self.res_projection(x_res)
            x_out = self.final_layer(x_out, t1).transpose(1, 2)
            x_out = self.conv2(x_out)
            return x_out

        # MLP-based final layer
        x_out = self.final_mlp(x_res)
        return x_out.transpose(1, 2)
