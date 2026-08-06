import torch
from torch.cuda import CUDAGraph
from vllm.platforms import current_platform


class HiFTGraphWrapper:
    def __init__(self, token2wav, connector_config, capture_batch_size):
        self.decode_fn = token2wav.hift.inference
        self.connector_config = connector_config
        self.codec_chunk_frames = connector_config["codec_chunk_frames"]
        self.codec_left_context_frames = connector_config["codec_left_context_frames"]
        lookahead_layer = getattr(token2wav.flow.encoder, "pre_lookahead_layer", None)
        self.pre_lookahead_len = int(getattr(lookahead_layer, "pre_lookahead_len"))
        self.mel_cache_len = int(token2wav.mel_cache_len)
        source_cache_len = int(token2wav.source_cache_len)
        self.mel_frames = int(token2wav.hift.conv_pre.in_channels)
        self.upsample_factor = source_cache_len // self.mel_cache_len
        self.flow_upsample_rate = int(getattr(token2wav.flow, "token_mel_ratio", 2))
        self.capture_bucket_size, self.capture_source_cache_len = self.derive_capture_bucket_size()
        self.capture_batch_size = capture_batch_size
        self.graph: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.static_speech_inputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_speech_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_cache_source_inputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_cache_source_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.enabled = True
        self.captured = False
        parameter = next(token2wav.hift.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.max_lazy_graphs = 8
        self.lazy_graph_count = 0

    def derive_capture_bucket_size(self):
        chunk_mel_frames = (
            self.codec_chunk_frames + self.codec_left_context_frames - self.pre_lookahead_len
        ) * self.flow_upsample_rate

        uncached_source_cache_len = 0
        cached_source_cache_len = self.mel_cache_len * self.upsample_factor
        uncached_bucket_size = chunk_mel_frames
        cached_bucket_size = chunk_mel_frames + self.mel_cache_len

        return [uncached_bucket_size, cached_bucket_size], [uncached_source_cache_len, cached_source_cache_len]

    def capture(self):
        if not self.enabled:
            return None

        for batch_size in self.capture_batch_size:
            for mel_frames, source_cache_len in zip(
                self.capture_bucket_size,
                self.capture_source_cache_len,
                strict=True,
            ):
                self._capture(batch_size, mel_frames, source_cache_len)

        self.captured = True

    def _capture(
        self,
        batch_size: int,
        mel_frames: int,
        source_cache_len: int,
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot capture HiFT graph during an active stream capture")

        key = (batch_size, mel_frames, source_cache_len)

        if key in self.graph:
            return

        static_mel = torch.zeros(batch_size, self.mel_frames, mel_frames, device=self.device, dtype=self.dtype)
        static_source_cache = torch.zeros(batch_size, 1, source_cache_len, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            _ = self.decode_fn(static_mel, static_source_cache)

        torch.accelerator.synchronize(self.device)

        graph = CUDAGraph()
        with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
            static_speech_output, static_cache_source_output = self.decode_fn(static_mel, static_source_cache)

        self.graph[key] = graph
        self.static_speech_inputs[key] = static_mel
        self.static_cache_source_inputs[key] = static_source_cache

        self.static_speech_outputs[key] = static_speech_output
        self.static_cache_source_outputs[key] = static_cache_source_output

    def replay(self, speech_feat, cache_source):
        if not self.enabled or not self.captured or torch.cuda.is_current_stream_capturing():
            return self.decode_fn(speech_feat, cache_source)

        batch_size = speech_feat.shape[0]
        num_frames = speech_feat.shape[2]
        cache_source_len = cache_source.shape[2]
        target_b = next((b for b in sorted(self.capture_batch_size) if b >= batch_size), None)

        if target_b is None:
            return self.decode_fn(speech_feat, cache_source)

        key = (target_b, num_frames, cache_source_len)

        if key not in self.graph:
            if self.lazy_graph_count >= self.max_lazy_graphs:
                return self.decode_fn(speech_feat, cache_source)
            self._capture(*key)
            self.lazy_graph_count += 1

        static_speech_inputs = self.static_speech_inputs[key].zero_()
        static_speech_inputs[:batch_size].copy_(speech_feat)
        static_cache_sources = self.static_cache_source_inputs[key].zero_()
        static_cache_sources[:batch_size].copy_(cache_source)

        self.graph[key].replay()
        static_speech_output = self.static_speech_outputs[key]
        static_cache_source_output = self.static_cache_source_outputs[key]
        speech = static_speech_output[:batch_size].clone()
        cache_source = static_cache_source_output[:batch_size].clone()
        return speech, cache_source
