from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import CausalConvRNNF0Predictor, CausalConv1d
import torch.nn as nn
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

    
class StepAudioCausalConvRNNF0Predictor(CausalConvRNNF0Predictor):
    def __init__(
        self,
        num_class: int = 1,
        in_channels: int = 80,
        cond_channels: int = 512,
    ):
        nn.Module.__init__(self)

        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(
                CausalConv1d(
                    in_channels,
                    cond_channels,
                    kernel_size=3,
                    causal_type="right",
                )
            ),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

