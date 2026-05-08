import torch
from transformers import PreTrainedModel, LlamaConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .step_audio_editx_transformer import Step1Model, Step1Config


class Step1CausalLMConfig(LlamaConfig):
    """Step1 CausalLM 配置类 - 继承LlamaConfig以支持标准HF加载"""
    model_type = "step1"  # 保持自定义的model_type

    def __init__(
        self,
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=48,
        num_attention_groups=4,
        num_hidden_layers=32,
        vocab_size=74752,
        rms_norm_eps=1e-5,
        torch_dtype="bfloat16",
        max_seq_len=999999,
        **kwargs
    ):
        # 将参数映射到LlamaConfig的标准参数
        kwargs.update({
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'num_attention_heads': num_attention_heads,
            'num_hidden_layers': num_hidden_layers,
            'vocab_size': vocab_size,
            'rms_norm_eps': rms_norm_eps,
            'torch_dtype': torch_dtype,
            'max_position_embeddings': max_seq_len,
        })
        
        super().__init__(**kwargs)
        
        # 保存Step1特有的参数
        self.num_attention_groups = num_attention_groups
        self.max_seq_len = max_seq_len


class Step1ForCausalLM(PreTrainedModel, GenerationMixin):

    config_class = Step1CausalLMConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(self, config: Step1CausalLMConfig):
        super().__init__(config)
        

        if isinstance(config.torch_dtype, str):
            dtype = getattr(torch, config.torch_dtype)
        else:
            dtype = config.torch_dtype or torch.float32


        step1_cfg = Step1Config(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_attention_groups=config.num_attention_groups,
            num_hidden_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
        )


        self.model = Step1Model(step1_cfg, dtype=dtype)
        self.lm_head = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype
        )

        # 初始化权重
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs
    ):


        hidden_states = self.model.embed_tokens(input_ids)

        

        hidden_states = self.model(hidden_states)

        logits = self.lm_head(hidden_states)
        
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )

    def get_input_embeddings(self):

        return self.model.embed_tokens

    def set_input_embeddings(self, value):

        self.model.embed_tokens = value

    def get_output_embeddings(self):

        return self.lm_head

    def set_output_embeddings(self, new_embeddings):

        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": kwargs.get("past_key_values"),
        }

    def _reorder_cache(self, past_key_values, beam_idx):

        return past_key_values

    def _set_gradient_checkpointing(self, module, value=False):

        if hasattr(self.model, 'gradient_checkpointing'):
            self.model.gradient_checkpointing = value 