export STEP_AUDIO_TOKENIZER_PATH="/root/autodl-tmp/Model/hub/models--stepfun-ai--Step-Audio-Tokenizer/snapshots/af7e5a3ec06175a7facae9d4100073d6e4dbb36c"

vllm-omni serve ~/autodl-tmp/Model/hub/models--stepfun-ai--Step-Audio-EditX/snapshots/5fe2f8a05c2353301ad47d3c1747b262115da138 \
    --deploy-config vllm-omni/vllm_omni/deploy/step_audio_editx.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --trust-remote-code \
    --omni