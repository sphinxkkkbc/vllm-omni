from dataclasses import dataclass

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from transformers import AutoTokenizer, GPT2Config, GPT2Model, SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel

from ..qwen3_tts.qwen3_tts_talker import Qwen3TTSSpeakerEncoder, Qwen3TTSSpeakerEncoderConfig
from .text_normalizer import TextNormalizer
from .utils import LANGUAGE_TOKEN_MAP


class DummyPositionEmbedding(nn.Module):
    """Placeholder position embedding that returns zeros.
    Args:
        embedding_dim: Dimension of embeddings
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Return zero embeddings.

        Args:
            position_ids: Position indices, shape (B, T)

        Returns:
            Zero tensor, shape (B, T, embedding_dim)
        """
        batch_size, seq_len = position_ids.shape
        return torch.zeros(batch_size, seq_len, self.embedding_dim, dtype=torch.float32, device=position_ids.device)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding with additive combination.

    Args:
        max_seq_len: Maximum sequence length
        embedding_dim: Dimension of embeddings
        init_std: Standard deviation for weight initialization
    """

    def __init__(self, max_seq_len: int, embedding_dim: int, init_std: float = 0.02):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.embedding.weight.data.normal_(mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input.

        Args:
            x: Input tensor, shape (B, T, D)

        Returns:
            Input with positional embeddings added, shape (B, T, D)
        """
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions).unsqueeze(0).expand(x.shape[0], -1, -1)

    def get_fixed_embedding(self, position: int, device: torch.device) -> torch.Tensor:
        """Get embedding for a specific position (used in KV-cached generation).

        Args:
            position: Position index
            device: Target device

        Returns:
            Position embedding, shape (1, 1, D)
        """
        pos_tensor = torch.tensor([position], device=device)
        return self.embedding(pos_tensor).unsqueeze(0)


class TextEmbeddingProjector(nn.Module):
    """
    MLP for resizing text embedding dimension.
    Reference: Qwen3TTSTalkerResizeMLP from https://github.com/QwenLM/Qwen3-TTS.

    Structure: Embedding -> Linear(input_size, intermediate_size) -> Act -> Linear(intermediate_size, output_size)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        output_size: int,
        hidden_act: str = "silu",
        bias: bool = True,
    ):
        super().__init__()

        # Initialize embedding layer and load pretrained weights
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Freeze embedding weights
        self.embed.weight.requires_grad = False
        self.embed.eval()

        # Text projection MLP (following Qwen3TTSTalkerResizeMLP structure)
        self.text_projection_fc1 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.text_projection_fc2 = nn.Linear(embed_dim, output_size, bias=bias)

        # Activation function
        if hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif hidden_act == "relu":
            self.act_fn = nn.ReLU()
        else:
            self.act_fn = nn.SiLU()  # Default to SiLU

        # Initialize projection layers
        nn.init.normal_(self.text_projection_fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.text_projection_fc2.weight, mean=0.0, std=0.02)
        if bias:
            nn.init.zeros_(self.text_projection_fc1.bias)
            nn.init.zeros_(self.text_projection_fc2.bias)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Resize text embeddings through MLP projection."""
        with torch.no_grad():
            text_embeds = self.embed(text_ids)
        # MLP projection: fc1 -> act -> fc2
        return self.text_projection_fc2(self.act_fn(self.text_projection_fc1(text_embeds)))

    def load_pretrained_embeddings(self, pretrained_weights: torch.Tensor):
        """Load pretrained embedding weights."""
        self.embed.weight.data.copy_(pretrained_weights)
        self.embed.weight.requires_grad = False


@dataclass
class Text2SemanticConfig(PretrainedConfig):
    """Configuration for Text2Semantic model.

    Args:
        num_layers: Number of transformer layers
        model_dim: Hidden dimension size
        num_heads: Number of attention heads
        max_text_seq_lens: Maximum text sequence length
        max_semantic_seq_lens: Maximum semantic sequence length
        vocab_size: Size of text vocabulary
        semantic_vocab_size: Size of semantic token vocabulary
        text_embedding_dim: Dimension of input text embeddings
        speaker_embedding_dim: Dimension of speaker/style conditioning vector
        start_semantic_token: BOS token ID for semantic sequence
        stop_semantic_token: EOS token ID for semantic sequence
    """

    model_type = "text2semantic"

    num_layers: int = 24
    model_dim: int = 1280
    num_heads: int = 20
    max_text_seq_lens: int = 520
    max_semantic_seq_lens: int = 1520
    vocab_size: int = 32000
    semantic_vocab_size: int = 8194
    text_embedding_dim: int = 4096
    speaker_embedding_dim: int = 1024
    start_semantic_token: int = 8192
    stop_semantic_token: int = 8193

    def __init__(
        self,
        num_layers: int = 24,
        model_dim: int = 1280,
        num_heads: int = 20,
        max_text_seq_lens: int = 520,
        max_semantic_seq_lens: int = 1520,
        vocab_size: int = 32000,
        semantic_vocab_size: int = 8194,
        text_embedding_dim: int = 4096,
        speaker_embedding_dim: int = 1024,
        start_semantic_token: int = 8192,
        stop_semantic_token: int = 8193,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.max_text_seq_lens = max_text_seq_lens
        self.max_semantic_seq_lens = max_semantic_seq_lens
        self.vocab_size = vocab_size
        self.semantic_vocab_size = semantic_vocab_size
        self.text_embedding_dim = text_embedding_dim
        self.speaker_embedding_dim = speaker_embedding_dim
        self.start_semantic_token = start_semantic_token
        self.stop_semantic_token = stop_semantic_token


class Text2Semantic(PreTrainedModel, GenerationMixin):
    config_class = Text2SemanticConfig

    def __init__(self, config: Text2SemanticConfig):
        super().__init__(config)

        self.config = config
        self.max_seq_len = config.max_text_seq_lens + config.max_semantic_seq_lens + 1

        self.text_projector = TextEmbeddingProjector(
            vocab_size=config.vocab_size,
            embed_dim=config.text_embedding_dim,
            output_size=config.model_dim,
        )

        self.semantic_embedding = nn.Embedding(config.semantic_vocab_size, config.model_dim)

        self.text_position_embedding = LearnedPositionalEmbedding(config.max_text_seq_lens, config.model_dim)
        self.semantic_position_embedding = LearnedPositionalEmbedding(config.max_semantic_seq_lens, config.model_dim)

        gpt_config = GPT2Config(
            vocab_size=config.semantic_vocab_size,
            n_positions=self.max_seq_len,
            n_ctx=self.max_seq_len,
            n_embd=config.model_dim,
            n_layer=config.num_layers,
            n_head=config.num_heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.transformer = GPT2Model(gpt_config)

        # Replace GPT2's position embedding with dummy (we use custom position embeddings)
        del self.transformer.wpe
        self.transformer.wpe = DummyPositionEmbedding(config.model_dim)

        # Remove GPT2's word embedding (we use custom embedding concatenation)
        del self.transformer.wte

        self.final_norm = nn.LayerNorm(config.model_dim)
        self.semantic_head = nn.Linear(config.model_dim, config.semantic_vocab_size)

        speaker_config = Qwen3TTSSpeakerEncoderConfig(
            mel_dim=config.speaker_embedding_dim,
            enc_dim=config.model_dim,
        )
        self.speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_config)

        # Caching for inference efficiency
        self.cached_condition_emb = None
        self.cached_text_emb = None

        self.post_init()

    def _prepare_embed_inputs(
        self,
        text_inputs: torch.Tensor | None = None,
        semantic_codes: torch.Tensor | None = None,
        condition_vector: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepare input embeddings by concatenating condition, text, and semantic embeddings.

        Training mode (text_inputs provided):
            Concatenates all three embedding types with their respective position encodings.

        Inference mode (input_ids provided):
            Uses cached condition and text embeddings, only computes new semantic embeddings.
            Supports both full sequence and single-token (KV-cached) inputs.

        Args:
            text_inputs: Text token IDs, shape (B, T_text)
            semantic_codes: Semantic token IDs, shape (B, T_sem)
            condition_vector: Speaker/style features, shape (B, T_cond, D_spk)
            input_ids: Combined token IDs for inference, shape (B, T) or (B, 1)
            attention_mask: Attention mask for inference, shape (B, T)

        Returns:
            Concatenated embeddings, shape (B, T_total, D)
        """
        if text_inputs is not None:
            # Training mode: full concatenation
            text_emb = self.text_projector(text_inputs)  # (B, T_text, D)
            text_emb = self.text_position_embedding(text_emb)

            semantic_emb = self.semantic_embedding(semantic_codes)  # (B, T_sem, D)
            semantic_emb = self.semantic_position_embedding(semantic_emb)

            condition_emb = self.speaker_encoder(condition_vector).unsqueeze(1)  # (B, 1, D)

            return torch.cat([condition_emb, text_emb, semantic_emb], dim=1)

        else:
            # Inference mode: use cached prefix
            condition_len = self.cached_condition_emb.shape[1]
            text_len = self.cached_text_emb.shape[1]
            prefix_len = condition_len + text_len

            if input_ids.shape[1] != 1:
                # First inference step: full sequence
                semantic_inputs = input_ids[:, prefix_len:]
                semantic_emb = self.semantic_embedding(semantic_inputs)
                semantic_emb = self.semantic_position_embedding(semantic_emb)

                # Handle beam search batch expansion
                repeat_factor = semantic_emb.shape[0] // self.cached_condition_emb.shape[0]
                condition_emb = self.cached_condition_emb.repeat_interleave(repeat_factor, 0)
                text_emb = self.cached_text_emb.repeat_interleave(repeat_factor, 0)

                return torch.cat([condition_emb, text_emb, semantic_emb], dim=1)

            else:
                # KV-cached step: single token
                semantic_emb = self.semantic_embedding(input_ids)  # (B, 1, D)

                # Compute position for this single token
                semantic_pos = attention_mask.shape[1] - prefix_len - 1
                pos_emb = self.semantic_position_embedding.get_fixed_embedding(semantic_pos, input_ids.device)
                return semantic_emb + pos_emb

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        text_inputs: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,
        semantic_codes: torch.Tensor | None = None,
        semantic_lengths: torch.Tensor | None = None,
        condition_vector: torch.Tensor | None = None,
        return_latent: bool = False,
    ):
        """Forward pass for T2S model.

        Training mode (text_inputs provided):
            Computes loss for semantic token prediction given text and condition.

        Inference mode (input_ids provided):
            Generates logits for next semantic token, uses cached KV and embeddings.

        Args:
            input_ids: Token IDs for inference mode
            attention_mask: Attention mask, shape (B, T)
            past_key_values: Cached key/value states from previous steps
            labels: Target semantic tokens for loss computation, shape (B, T_sem)
            text_inputs: Text token IDs, shape (B, T_text)
            text_lengths: Valid text lengths for each batch element, shape (B,)
            semantic_codes: Semantic token IDs with BOS/EOS, shape (B, T_sem)
            semantic_lengths: Valid semantic lengths (excluding BOS/EOS), shape (B,)
            condition_vector: Speaker/style features, shape (B, T_cond, D_spk)
            return_latent: If True, return hidden states instead of logits

        Returns:
            CausalLMOutputWithCrossAttentions with loss, logits, and hidden states
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if text_inputs is not None:
            # Training mode
            inputs_embeds = self._prepare_embed_inputs(
                text_inputs=text_inputs,
                semantic_codes=semantic_codes,
                condition_vector=condition_vector,
            )

            if attention_mask is None:
                # Auto-generate attention mask from lengths
                batch_size = text_inputs.shape[0]
                device = text_inputs.device
                cond_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
                text_mask = torch.arange(text_inputs.shape[1], device=device).unsqueeze(0) < text_lengths.unsqueeze(1)
                semantic_mask = torch.arange(semantic_codes.shape[1], device=device).unsqueeze(0) < (
                    semantic_lengths + 2
                ).unsqueeze(1)
                attention_mask = torch.cat([cond_mask, text_mask, semantic_mask], dim=1)

        else:
            # Inference mode
            inputs_embeds = self._prepare_embed_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs.last_hidden_state  # (B, T_total, D)

        if return_latent:
            # Return latent representations (excluding condition and text prefix, and BOS/EOS)
            return hidden_states[:, 1 + text_inputs.shape[1] : -2]

        if text_inputs is not None:
            # Training: extract semantic portion (skip condition token, then skip text)
            hidden_states = self.final_norm(hidden_states[:, 1:])[:, text_inputs.shape[1] :]

        else:
            # Inference
            hidden_states = self.final_norm(hidden_states)

        logits = self.semantic_head(hidden_states)  # (B, T_sem, vocab_size)

        loss = None
        if labels is not None:
            logits_for_loss = logits.permute(0, 2, 1)  # (B, vocab_size, T_sem)
            loss = F.cross_entropy(logits_for_loss, labels, ignore_index=-100)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def store_conditioning(self, condition_vector: torch.Tensor, text_inputs: torch.Tensor):
        """Cache condition and text embeddings for efficient inference.

        Avoids recomputing prefix embeddings during autoregressive generation.

        Args:
            condition_vector: Speaker/style features, shape (B, T_cond, D_spk)
            text_inputs: Text token IDs, shape (B, T_text)
        """
        with torch.no_grad():
            condition_emb = self.speaker_encoder(condition_vector).unsqueeze(1)
            text_emb = self.text_projector(text_inputs)
            text_emb = self.text_position_embedding(text_emb)

            self.cached_condition_emb = condition_emb
            self.cached_text_emb = text_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for HuggingFace generation interface.

        Args:
            input_ids: Current token sequence
            past_key_values: Cached KV states
            **kwargs: Additional generation arguments

        Returns:
            Dict with model inputs for next generation step
        """
        attention_mask = kwargs.get("attention_mask", None)

        if past_key_values:
            # Use only last token when KV cache is available
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cached KV states for beam search.

        Args:
            past_key_values: Cached states for all layers
            beam_idx: Beam indices to select

        Returns:
            Reordered cache
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

    @torch.no_grad()
    def generate(
        self,
        text_inputs: torch.Tensor,
        condition_vector: torch.Tensor,
        max_length: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: int | None = None,
        return_latent: bool = False,
        **kwargs,
    ):
        """Generate semantic tokens from text and condition autoregressively.

        Args:
            text_inputs: Text token IDs, shape (B, T_text)
            condition_vector: Speaker/style features, shape (B, T_cond, D_spk)
            max_length: Maximum total sequence length (prefix + generated tokens)
            temperature: Sampling temperature (higher = more diverse)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling probability threshold
            do_sample: Use sampling if True, greedy decoding if False
            eos_token_id: Stop token ID (defaults to config.stop_semantic_token)
            return_latent: If True, also return hidden state representations
            **kwargs: Additional generation arguments (num_beams, etc.)

        Returns:
            If return_latent=False: Semantic token IDs, shape (B, T_gen)
            If return_latent=True: Dict with "semantic_codes" and "latent" (hidden states)
        """
        self.store_conditioning(condition_vector, text_inputs)

        batch_size = text_inputs.shape[0]
        device = text_inputs.device
        prefix_len = 1 + text_inputs.shape[1]  # condition(1) + text

        bos = self.config.start_semantic_token
        eos = eos_token_id if eos_token_id is not None else self.config.stop_semantic_token

        # Initialize with BOS tokens for the semantic portion
        start_tokens = torch.full(
            (batch_size, prefix_len + 1),
            fill_value=bos,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones(batch_size, prefix_len + 1, dtype=torch.long, device=device)

        # Call HuggingFace generate
        generated = super().generate(
            start_tokens,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=eos,
            pad_token_id=self.config.stop_semantic_token,
            **kwargs,
        )

        # Extract semantic tokens (remove prefix and BOS)
        semantic_codes = generated[:, prefix_len + 1 :]

        if return_latent:
            # Compute latent representations by forward pass
            with torch.no_grad():
                bos_col = torch.full(
                    (batch_size, 1),
                    bos,
                    dtype=semantic_codes.dtype,
                    device=device,
                )
                semantic_codes = torch.cat([bos_col, semantic_codes], dim=1)
                inputs_embeds = self._prepare_embed_inputs(
                    text_inputs=text_inputs,
                    semantic_codes=semantic_codes,
                    condition_vector=condition_vector,
                )
                text_lengths = torch.tensor([text_inputs.shape[1]], device=device)
                semantic_lengths = torch.tensor([semantic_codes.shape[1] - 2], device=device)

                cond_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
                text_mask = torch.arange(text_inputs.shape[1], device=device).unsqueeze(0) < text_lengths.unsqueeze(1)
                semantic_mask = torch.arange(semantic_codes.shape[1], device=device).unsqueeze(0) < (
                    semantic_lengths + 2
                ).unsqueeze(1)
                attention_mask = torch.cat([cond_mask, text_mask, semantic_mask], dim=1)

                transformer_outputs = self.transformer(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )

                hidden_states = transformer_outputs.last_hidden_state
                latent = hidden_states[:, 1 + text_inputs.shape[1] : -2]  # Extract semantic portion
                semantic_codes = semantic_codes[:, 1:-1]  # Remove BOS/EOS

            return {"semantic_codes": semantic_codes, "latent": latent}

        return semantic_codes


class Confucius4TTS_AR(nn.Module):
    def __init__(self, device, config_path, t2s_checkpoint=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(device)

        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        paths = self.cfg["paths"]
        if t2s_checkpoint is not None:
            paths["t2s_checkpoint"] = t2s_checkpoint
        paths.setdefault("t2s_checkpoint", "checkpoints/model.safetensors")
        self.normalizer = TextNormalizer()

        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(paths["w2v_bert_path"])
        self.w2v_model = Wav2Vec2BertModel.from_pretrained(paths["w2v_bert_path"]).eval().to(self.device)
        stats = torch.load(paths["w2v_stat"], map_location="cpu")
        self.semantic_mean = stats["mean"].to(self.device)
        self.semantic_std = torch.sqrt(stats["var"]).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer_path"])
        t2s_config = Text2SemanticConfig(**self.cfg["t2s_model"])
        self.t2s_model = Text2Semantic(t2s_config)
        self.t2s_model.config.vocab_size = t2s_config.semantic_vocab_size
        self.t2s_model.load_state_dict(safetensors.torch.load_file(paths["t2s_checkpoint"], device="cpu"))
        self.t2s_model.eval().to(self.device)

    def _load_prompt(self, prompt_wav: str):
        """Load and resample reference audio to 16kHz and target sample rate.

        Args:
            prompt_wav: Path to reference audio file

        Returns:
            Tuple of (wav_16k, wav_tgt) resampled to 16kHz and target sample rate
        """
        wav, sr = torchaudio.load(prompt_wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav_16k = wav if sr == 16000 else torchaudio.functional.resample(wav, sr, 16000)
        wav_tgt = wav if sr == self.sample_rate else torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav_16k, wav_tgt

    def _extract_semantic(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Extract normalized semantic features from reference audio using Wav2Vec2-BERT.

        Args:
            wav_16k: Waveform at 16kHz, shape (1, T)

        Returns:
            Normalized hidden states from layer 17, shape (1, T_feat, D)
        """
        inputs = self.feature_extractor(wav_16k.squeeze(0).cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        outputs = self.w2v_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feats = outputs.hidden_states[17]  # Layer 17 hidden states
        return (feats - self.semantic_mean) / self.semantic_std

    def preprocess(self, input_ids, text, prompt_wav, lang, max_text_tokens_per_segment):
        # Normalize text (punctuation, numbers, etc.)
        text = self.normalizer.normalize(text, language=lang)
        wav_16k, wav_tgt = self._load_prompt(prompt_wav)
        # Extract conditioning from reference audio
        semantic_features = self._extract_semantic(wav_16k)

        segments = self.normalizer.segment_text(
            text,
            tokenize_fn=self.tokenizer.tokenize,
            language=lang,
            max_tokens=max_text_tokens_per_segment,
        )
        if not segments:
            segments = [text]

        return segments, semantic_features, wav_tgt

    def forward(self, segments, semantic_features):
        chunks = []
        for i, text in enumerate(segments):
            chunk = self.generate(text, semantic_features)
            chunks.append(chunk)
        return torch.cat(chunks, dim=1)

    def generate(
        self,
        lang,
        semantic_features,
        text,
        max_length=512,
        num_beams=1,
        top_p=0.9,
        top_k=50,
        temperature=1.0,
        repetition_penalty=1.0,
    ):
        lang_token = LANGUAGE_TOKEN_MAP.get(lang, f"请用{lang}朗读接下来的文字")
        formatted = f"You are a helpful assistant. {lang_token}:{text}"
        token_ids = self.tokenizer.encode(formatted, return_tensors="pt").to(self.device)

        t2s_out = self.t2s_model.generate(
            text_inputs=token_ids,
            condition_vector=semantic_features,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            early_stopping=True,
            return_latent=True,
        )
        return t2s_out
