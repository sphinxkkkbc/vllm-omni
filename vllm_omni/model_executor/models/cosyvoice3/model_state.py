# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CosyVoice3's MRv2 sampling-to-audio lifecycle.

run_eager_mtp is the existing post-sampling entry point. It can move to
model_state.post_sample once #9173 introduces that common lifecycle hook.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import decode_streaming_batch
from vllm_omni.model_executor.models.cosyvoice3.pipeline_adapter import InnerStageAdapter
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class CosyVoice3ModelState(OmniModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        if vllm_config.speculative_config is not None:
            raise ValueError("CosyVoice3 single-stage does not yet support speculative decoding")
        super().__init__(vllm_config, model, encoder_cache, device)
        self.adapter = InnerStageAdapter(
            int(model.config.llm["speech_token_size"]),
            int(getattr(model.config, "codec_chunk_frames", 25)),
            int(model.config.flow["pre_lookahead_len"]),
        )
        self._max_model_len = int(vllm_config.model_config.max_model_len)

    def prepare_inputs(self, input_batch, req_states):
        # Borrow runner-owned state; do not copy limits into request/batch buffers.
        self._req_states = req_states
        return super().prepare_inputs(input_batch, req_states)

    def run_eager_mtp(
        self,
        input_batch,
        text_hidden,
        sampled_token_ids,
        multimodal_outputs,
        mtp_batch_descriptor_dispatcher=None,
    ):
        if sampled_token_ids.shape[1] != 1:
            raise ValueError("CosyVoice3 single-stage expects one sampled token per request")
        count = input_batch.num_reqs
        scheduled = input_batch.num_scheduled_tokens[:count]
        is_last_prefill = (
            input_batch.num_computed_prefill_tokens_np[:count] + scheduled >= input_batch.prefill_len_np[:count]
        )
        kept = ~input_batch.is_prefilling_np[:count] | is_last_prefill
        payloads = self.intermediate_buffer.gather(input_batch)
        limits = np.minimum(self._req_states.max_seq_len[input_batch.idx_mapping_np[:count]], self._max_model_len)
        end = input_batch.num_computed_tokens_np[:count] + scheduled
        positions = end - self._req_states.prompt_len.np[input_batch.idx_mapping_np[:count]]
        length_finished = end + 1 >= limits
        tokens = sampled_token_ids[:count, 0]
        speech_mask = (tokens >= 0) & (tokens < self.adapter.vocab_size)
        step_embed = multimodal_outputs.get("embed", {})
        items, bindings = [], []
        # Only speech/EOS classification is read back; validity and lengths
        # come from CPU scheduling metadata. No per-step tail candidates.
        speech = speech_mask.cpu().tolist()
        capacity = self._max_model_len
        for row, payload in enumerate(payloads):
            req_id = input_batch.req_ids[row]
            # Retain live conditioning even on an intermediate prefill step.
            embed = payload.setdefault("embed", {})
            if isinstance(step_embed, Mapping):
                for key in ("speech_token", "speech_feat", "embedding", "speech_token_len"):
                    values = step_embed.get(key)
                    if isinstance(values, (list, tuple)) and row < len(values) and values[row] is not None:
                        embed[key] = values[row]
            if not kept[row] or str(req_id).startswith("_warmup_"):
                continue
            self.adapter.append(
                req_id,
                sampled_token_ids[row, :1],
                embed,
                capacity=capacity,
                position=int(positions[row]),
                speech_mask=speech_mask[row],
            )
            is_speech = speech[row]
            length = int(positions[row]) + int(is_speech)
            final = not is_speech or bool(length_finished[row])
            item = self.adapter.prepare(req_id, length, final)
            if item is not None:
                items.append(item)
                bindings.append((row, req_id))
        empty = text_hidden.new_empty(0, dtype=torch.float32)
        audio = [empty] * count
        if items:
            self.model._maybe_enable_code2wav_trt()
            results = decode_streaming_batch(self.model.code2wav, items)
            for (row, req_id), (wav, cache) in zip(bindings, results):
                audio[row] = wav.reshape(-1).float()
                self.adapter.apply_result(req_id, cache)
        multimodal_outputs.clear()
        multimodal_outputs.update(
            audio=audio, sr=[torch.tensor(int(self.model.config.sample_rate), dtype=torch.int32)] * count
        )

    def on_requests_finished(self, req_ids: set[str]) -> None:
        self.adapter.finish_requests(req_ids)
        super().on_requests_finished(req_ids)
