import os
import json
import random
import string
import logging
import threading
import contextlib
import tempfile
from io import BytesIO
from urllib.parse import urlparse
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Generator
import numpy as np
from omegaconf import OmegaConf
import torch
from huggingface_hub import snapshot_download
import inspect
import re
import requests
from pathlib import Path
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import AutoWeightsLoader
from transformers import AutoTokenizer
from vllm import LLM
from .utils import set_all_random_seed

_model_download_cache = {}
_download_cache_lock = threading.Lock()
default_model_path = "stepfun-ai/Step-Audio-EditX"
default_tokenizer_path = "stepfun-ai/Step-Audio-Tokenizer"

class AutoModelLoader:
    def __init__(self, **kwargs):
        pass

    def load_ar(model_path, **kwargs):
        return
    
    def load_decoder(model_path, **kwargs):
        return

    def load_tokenizer(model_path, **kwargs):
        return

class AutoModel:
    def __init__(self, **kwargs):
        self.tables = RegisterTables()
        if not kwargs.get("disable_log", False):
            self.tables.print()
        model, kwargs = self.build_model(**kwargs)
        self.kwargs = kwargs
        self.model = model
        self.model_path = kwargs.get("model_path")
        self.repo_path = kwargs.get("repo_path")


    def build_model(self, **kwargs):
        assert "model" in kwargs
        if "model_conf" not in kwargs:
            logging.info(
                "download models from model hub: {}".format(
                    kwargs.get("model_hub", "ms")
                )
            )
            kwargs = download_model(**kwargs)

        set_all_random_seed(kwargs.get("seed", 0))

        device = kwargs.get("device", "cuda")
        if not torch.cuda.is_available() or kwargs.get("ngpu", 1) == 0:
            device = "cpu"
            kwargs["batch_size"] = 1
        kwargs["device"] = device

        if kwargs.get("ncpu", None):
            torch.set_num_threads(kwargs.get("ncpu"))

        # build tokenizer
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            tokenizer_class = self.tables.tokenizer_classes.get(tokenizer)
            tokenizer = tokenizer_class(**kwargs["tokenizer_conf"])
            kwargs["tokenizer"] = tokenizer
            kwargs["token_list"] = tokenizer.token_list
            vocab_size = len(tokenizer.token_list)
        else:
            vocab_size = -1

        # build frontend
        frontend = kwargs.get("frontend", None)
        if frontend is not None:
            frontend_class = self.tables.frontend_classes.get(frontend)
            frontend = frontend_class(**kwargs["frontend_conf"])
            kwargs["frontend"] = frontend
            kwargs["input_size"] = frontend.output_size()

        # build model
        model_class = self.tables.model_classes.get(kwargs["model"])
        model = model_class(**kwargs, **kwargs["model_conf"], vocab_size=vocab_size)

        model.to(device)

        # init_param
        init_param = kwargs.get("init_param", None)
        if init_param is not None:
            logging.info(f"Loading pretrained params from {init_param}")
            load_pretrained_model(
                model=model,
                path=init_param,
                ignore_init_mismatch=kwargs.get("ignore_init_mismatch", False),
                oss_bucket=kwargs.get("oss_bucket", None),
                scope_map=kwargs.get("scope_map", None),
                excludes=kwargs.get("excludes", None),
            )

        return model, kwargs

    def __call__(self, *args, **cfg):
        kwargs = self.kwargs
        kwargs.update(cfg)
        res = self.model(*args, kwargs)
        return res


class ModelSource:
    """Model source enumeration"""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class UnifiedModelLoader:
    """Unified model loader using vLLM"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _cached_snapshot_download(self, model_path: str, source: str, **kwargs) -> str:
        """
        Cached version of snapshot_download to avoid repeated downloads
        """
        cache_key = (model_path, source, str(sorted(kwargs.items())))

        with _download_cache_lock:
            if cache_key in _model_download_cache:
                cached_path = _model_download_cache[cache_key]
                self.logger.info(f"Using cached download for {model_path} from {source}: {cached_path}")
                return cached_path
        try:
            local_path = snapshot_download(model_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to download model from {source}: {model_path}. Error: {e}")

        with _download_cache_lock:
            _model_download_cache[cache_key] = local_path

        self.logger.info(f"Downloaded and cached {model_path} from {source}: {local_path}")
        return local_path

    def load_model(
        self,
        model_path: str,
        source: str = ModelSource.HUGGINGFACE,
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        max_model_len: Optional[int] = None,
        enforce_eager: bool = False,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        kv_cache_dtype: Optional[str] = None,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        **kwargs
    ) -> tuple:
        """
        Load model using vLLM for high-performance inference

        Args:
            model_path: Model path or ID
            source: Model source (auto/local/modelscope/huggingface)
            quantization: Quantization method ('awq', 'gptq', 'fp8', or None)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0)
            max_model_len: Maximum sequence length
            dtype: Data type ('float16', 'bfloat16', 'float32')
            trust_remote_code: Whether to trust remote code
            kv_cache_dtype: KV cache dtype (None, 'auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3')
            max_num_seqs: Maximum number of concurrent sequences
            max_num_batched_tokens: Maximum tokens per batch
            **kwargs: Other vLLM parameters

        Returns:
            Tuple of (llm, tokenizer, model_path)

        Example:
            >>> loader = UnifiedModelLoader()
            >>> llm, tokenizer, path = loader.load_model(
            ...     model_path="/path/to/model",
            ...     quantization="awq",
            ...     tensor_parallel_size=2
            ... )
        """
        self.logger.info(f"🚀 Loading vLLM model from {source}: {model_path}")
        if quantization:
            self.logger.info(f"🔧 Quantization: {quantization}")

        try:
            # Resolve model path based on source
            resolved_path = model_path
            try:
                resolved_path = self._cached_snapshot_download(model_path, ModelSource.HUGGINGFACE)
            except Exception as e:
                self.logger.warning(f"Failed to download from Hugging Face: {e}. Trying local path.")
                if not os.path.exists(model_path):
                    raise ValueError(f"Model path does not exist: {model_path}")

            # Build vLLM arguments
            llm_kwargs = {
                "model": resolved_path,
                "trust_remote_code": trust_remote_code,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "dtype": dtype,
                "enforce_eager": enforce_eager,
            }

            if quantization:
                llm_kwargs["quantization"] = quantization

            if max_model_len is not None:
                llm_kwargs["max_model_len"] = max_model_len

            # Memory optimization parameters
            if kv_cache_dtype is not None:
                llm_kwargs["kv_cache_dtype"] = kv_cache_dtype

            if max_num_seqs is not None:
                llm_kwargs["max_num_seqs"] = max_num_seqs

            if max_num_batched_tokens is not None:
                llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

            llm_kwargs.update(kwargs)

            self.logger.info(f"🔧 vLLM config: {llm_kwargs}")

            # Create vLLM LLM instance
            llm = LLM(**llm_kwargs)

            # Load tokenizer separately (needed for encoding prompts)
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_path,
                trust_remote_code=True
            )

            self.logger.info(f"✅ Successfully loaded vLLM model")
            return llm, tokenizer, resolved_path

        except Exception as e:
            self.logger.error(f"❌ Failed to load vLLM model: {e}")
            raise

    def load_funasr_model(
        self,
        repo_path: str,
        model_path: str,
        source: str = ModelSource.HUGGINGFACE,
        **kwargs
    ) -> AutoModel:
        """
        Load FunASR model (for StepAudioTokenizer)

        Args:
            repo_path: Repository path
            model_path: Model path or ID
            source: Model source
            **kwargs: Other parameters

        Returns:
            FunASR AutoModel instance
        """
        self.logger.info(f"Loading FunASR model from {source}: {model_path}")

        try:
            model_revision = kwargs.pop("model_revision", "main")

            if source == ModelSource.LOCAL:
                model_hub = "local"
            elif source == ModelSource.HUGGINGFACE:
                model_hub = "hf"
            else:
                raise ValueError(f"Unsupported model source: {source}")

            model = AutoModel(
                repo_path=repo_path,
                model=model_path,
                model_hub=model_hub,
                model_revision=model_revision,
                **kwargs
            )

            self.logger.info(f"✅ Successfully loaded FunASR model")
            return model

        except Exception as e:
            self.logger.error(f"❌ Failed to load FunASR model: {e}")
            raise

def load_audio(file_path, target_sample_rate=16000):
    return 

def load_bytes(input):
    middle_data = np.frombuffer(input, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    if middle_data.dtype.kind not in "iu":
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype("float32")
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(middle_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    array = np.frombuffer(
        (middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32
    )
    return array

def prepare_data_iterator(data_in, input_len=None, data_type=None, key=None):
    """
    :param input:
    :param input_len:
    :param data_type:
    :param frontend:
    :return:
    """
    data_list = []
    key_list = []
    filelist = [".scp", ".txt", ".json", ".jsonl"]

    chars = string.ascii_letters + string.digits
    if isinstance(data_in, str) and data_in.startswith("http"):  # url
        data_in = download_from_url(data_in)
    if isinstance(data_in, str) and os.path.exists(
        data_in
    ):  # wav_path; filelist: wav.scp, file.jsonl;text.txt;
        _, file_extension = os.path.splitext(data_in)
        file_extension = file_extension.lower()
        if file_extension in filelist:  # filelist: wav.scp, file.jsonl;text.txt;
            with open(data_in, encoding="utf-8") as fin:
                for line in fin:
                    key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
                    if data_in.endswith(
                        ".jsonl"
                    ):  # file.jsonl: json.dumps({"source": data})
                        lines = json.loads(line.strip())
                        data = lines["source"]
                        key = data["key"] if "key" in data else key
                    else:  # filelist, wav.scp, text.txt: id \t data or data
                        lines = line.strip().split(maxsplit=1)
                        data = lines[1] if len(lines) > 1 else lines[0]
                        key = lines[0] if len(lines) > 1 else key

                    data_list.append(data)
                    key_list.append(key)
        else:
            key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
            data_list = [data_in]
            key_list = [key]
    elif isinstance(data_in, (list, tuple)):
        if data_type is not None and isinstance(
            data_type, (list, tuple)
        ):  # mutiple inputs
            data_list_tmp = []
            for data_in_i, data_type_i in zip(data_in, data_type):
                key_list, data_list_i = prepare_data_iterator(
                    data_in=data_in_i, data_type=data_type_i
                )
                data_list_tmp.append(data_list_i)
            data_list = []
            for item in zip(*data_list_tmp):
                data_list.append(item)
        else:
            # [audio sample point, fbank, text]
            data_list = data_in
            key_list = [
                "rand_key_" + "".join(random.choice(chars) for _ in range(13))
                for _ in range(len(data_in))
            ]
    else:  # raw text; audio sample point, fbank; bytes
        if isinstance(data_in, bytes):  # audio bytes
            data_in = load_bytes(data_in)
        if key is None:
            key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
        data_list = [data_in]
        key_list = [key]

    return key_list, data_list


def normalize_cache_path(cache_path):
    """Normalize cache path to ensure consistent format with snapshots/{commit_id}."""
    # Check if the cache_path directory contains a snapshots folder
    snapshots_dir = os.path.join(cache_path, "snapshots")
    if os.path.exists(snapshots_dir) and os.path.isdir(snapshots_dir):
        # Find the commit_id subdirectory in snapshots
        try:
            snapshot_items = os.listdir(snapshots_dir)
            # Look for the first directory (should be the commit_id)
            for item in snapshot_items:
                item_path = os.path.join(snapshots_dir, item)
                if os.path.isdir(item_path):
                    # Found commit_id directory, return the full path
                    return os.path.join(cache_path, "snapshots", item)
        except OSError:
            pass

    # If no snapshots directory found or error occurred, return original path
    return cache_path

def get_or_download_model_dir(
    model,
    model_revision=None,
    model_hub="hf",
):
    """Get local model directory or download model if necessary.

    Args:
        model (str): model id or path to local model directory.
                    For HF subfolders, use format: "repo_id/subfolder_path"
        model_revision  (str, optional): model version number.
        is_training (bool): Whether this is for training
        check_latest (bool): Whether to check for latest version
        model_hub (str): Model hub type ("hf" for HuggingFace)
    """
    # Extract repo_id for caching (handle subfolder case)
    if "/" in model and len(model.split("/")) > 2:
        parts = model.split("/")
        repo_id = "/".join(parts[:2])  # e.g., "organization/repo" or "stepfun-ai/Step-Audio-EditX"
        subfolder = "/".join(parts[2:])  # e.g., "subfolder/model"
    else:
        repo_id = model
        subfolder = None

    # Create cache key
    cache_key = (repo_id, model_revision, model_hub)

    # Check cache first
    with _download_cache_lock:
        if cache_key in _model_download_cache:
            cached_repo_dir = _model_download_cache[cache_key]
            print(f"Using cached model for {repo_id}: {cached_repo_dir}")

            # For subfolder case, construct the model_cache_dir from cached repo
            if subfolder:
                model_cache_dir = os.path.join(cached_repo_dir, subfolder)
                if not os.path.exists(model_cache_dir):
                    raise FileNotFoundError(f"Subfolder {subfolder} not found in cached repo {repo_id}")
            else:
                model_cache_dir = cached_repo_dir

            return cached_repo_dir, model_cache_dir
        
    if model_hub == "hf":
        # Download the repo (use repo_id, not the full model path with subfolder)
        repo_cache_dir = snapshot_download(
            repo_id=repo_id,
            revision=model_revision,
            allow_patterns=None,  # Download all files to ensure resource files are available
        )
        repo_cache_dir = normalize_cache_path(repo_cache_dir)

        # Construct model_cache_dir
        if subfolder:
            model_cache_dir = os.path.join(repo_cache_dir, subfolder)
            if not os.path.exists(model_cache_dir):
                raise FileNotFoundError(f"Subfolder {subfolder} not found in downloaded repo {repo_id}")
        else:
            model_cache_dir = normalize_cache_path(repo_cache_dir)
    else:
        raise ValueError(f"Unsupported model_hub: {model_hub}")

    # Cache the result before returning
    with _download_cache_lock:
        _model_download_cache[cache_key] = repo_cache_dir

    print(f"Model downloaded to: {model_cache_dir}")
    return repo_cache_dir, model_cache_dir


def load_pretrained_model(
    path: str,
    model: torch.nn.Module,
    ignore_init_mismatch: bool,
    map_location: str = "cpu",
    oss_bucket=None,
    scope_map=None,
    excludes=None,
):
    """Load a model state and set it to the model.

    Args:
            init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

    Examples:

    """

    obj = model
    dst_state = obj.state_dict()
    print(f"ckpt: {path}")
    if oss_bucket is None:
        src_state = torch.load(path, map_location=map_location)
    else:
        buffer = BytesIO(oss_bucket.get_object(path).read())
        src_state = torch.load(buffer, map_location=map_location)
    if "state_dict" in src_state:
        src_state = src_state["state_dict"]

    for k in dst_state.keys():
        if not k.startswith("module.") and "module." + k in src_state.keys():
            k_ddp = "module." + k
        else:
            k_ddp = k
        if k_ddp in src_state:
            dst_state[k] = src_state[k_ddp]
        else:
            print(f"Miss key in ckpt: model: {k}, ckpt: {k_ddp}")

    flag = obj.load_state_dict(dst_state, strict=True)

def add_file_root_path(model_or_path: str, file_path_metas: dict, cfg={}):
    if isinstance(file_path_metas, dict):
        for k, v in file_path_metas.items():
            if isinstance(v, str):
                p = os.path.join(model_or_path, v)
                if os.path.exists(p):
                    cfg[k] = p
            elif isinstance(v, dict):
                if k not in cfg:
                    cfg[k] = {}
                add_file_root_path(model_or_path, v, cfg[k])

    return cfg


def download_model(**kwargs):
    model_hub = kwargs.get("model_hub", "hf")
    model_or_path = kwargs.get("model")
    repo_path = kwargs.get("repo_path", "")

    model_revision = kwargs.get("model_revision")

    # Download model if it doesn't exist locally
    if not os.path.exists(model_or_path):
        if model_hub == "local":
            # For local models, the path should already exist
            raise FileNotFoundError(f"Local model path does not exist: {model_or_path}")
        elif model_hub  == "hf":
            repo_path, model_or_path = get_or_download_model_dir(
                model_or_path,
                model_revision,
                model_hub=model_hub,
            )
        else:
            raise ValueError(f"Unsupported model_hub: {model_hub}")

    print(f"Using model path: {model_or_path}")
    kwargs["model_path"] = model_or_path
    kwargs["repo_path"] = repo_path

    # Common logic for processing configuration files (same for all model hubs)
    if os.path.exists(os.path.join(model_or_path, "configuration.json")):
        with open(
            os.path.join(model_or_path, "configuration.json"), "r", encoding="utf-8"
        ) as f:
            conf_json = json.load(f)
            cfg = {}
            add_file_root_path(model_or_path, conf_json["file_path_metas"], cfg)
            cfg.update(kwargs)
            config = OmegaConf.load(cfg["config"])
            kwargs = OmegaConf.merge(config, cfg)
        kwargs["model"] = config["model"]
    elif os.path.exists(os.path.join(model_or_path, "config.yaml")) and os.path.exists(
        os.path.join(model_or_path, "model.pt")
    ):
        config = OmegaConf.load(os.path.join(model_or_path, "config.yaml"))
        kwargs = OmegaConf.merge(config, kwargs)
        init_param = os.path.join(model_or_path, "model.pb")
        kwargs["init_param"] = init_param
        if os.path.exists(os.path.join(model_or_path, "tokens.txt")):
            kwargs["tokenizer_conf"]["token_list"] = os.path.join(
                model_or_path, "tokens.txt"
            )
        if os.path.exists(os.path.join(model_or_path, "tokens.json")):
            kwargs["tokenizer_conf"]["token_list"] = os.path.join(
                model_or_path, "tokens.json"
            )
        if os.path.exists(os.path.join(model_or_path, "seg_dict")):
            kwargs["tokenizer_conf"]["seg_dict"] = os.path.join(
                model_or_path, "seg_dict"
            )
        if os.path.exists(os.path.join(model_or_path, "bpe.model")):
            kwargs["tokenizer_conf"]["bpemodel"] = os.path.join(
                model_or_path, "bpe.model"
            )
        kwargs["model"] = config["model"]
        if os.path.exists(os.path.join(model_or_path, "am.mvn")):
            kwargs["frontend_conf"]["cmvn_file"] = os.path.join(model_or_path, "am.mvn")
        if os.path.exists(os.path.join(model_or_path, "jieba_usr_dict")):
            kwargs["jieba_usr_dict"] = os.path.join(model_or_path, "jieba_usr_dict")

    return OmegaConf.to_container(kwargs, resolve=True)

def download_from_url(url):
    result = urlparse(url)
    file_path = None
    if result.scheme is not None and len(result.scheme) > 0:
        storage = HTTPStorage()
        # bytes
        data = storage.read(url)
        work_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        file_path = os.path.join(work_dir, os.path.basename(url))
        with open(file_path, "wb") as fb:
            fb.write(data)
    assert file_path is not None, f"failed to download: {url}"
    return file_path

class ModelSource:
    """Model source enumeration"""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class Storage(metaclass=ABCMeta):
    """Abstract class of storage.

    All backends need to implement two apis: ``read()`` and ``read_text()``.
    ``read()`` reads the file as a byte stream and ``read_text()`` reads
    the file as texts.
    """

    @abstractmethod
    def read(self, filepath: str):
        pass

    @abstractmethod
    def read_text(self, filepath: str):
        pass

    @abstractmethod
    def write(self, obj: bytes, filepath: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def write_text(
        self, obj: str, filepath: Union[str, Path], encoding: str = "utf-8"
    ) -> None:
        pass


class HTTPStorage(Storage):
    """HTTP and HTTPS storage."""

    def read(self, url):
        # TODO @wenmeng.zwm add progress bar if file is too large
        r = requests.get(url)
        r.raise_for_status()
        return r.content

    def read_text(self, url):
        r = requests.get(url)
        r.raise_for_status()
        return r.text

    @contextlib.contextmanager
    def as_local_path(self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath``.

        ``as_local_path`` is decorated by :meth:`contextlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Examples:
            >>> storage = HTTPStorage()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with storage.get_local_path('http://path/to/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.read(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def write(self, obj: bytes, url: Union[str, Path]) -> None:
        raise NotImplementedError("write is not supported by HTTP Storage")

    def write_text(
        self, obj: str, url: Union[str, Path], encoding: str = "utf-8"
    ) -> None:
        raise NotImplementedError("write_text is not supported by HTTP Storage")

@dataclass
class RegisterTables:
    model_classes = {}
    frontend_classes = {}
    specaug_classes = {}
    normalize_classes = {}
    encoder_classes = {}
    decoder_classes = {}
    joint_network_classes = {}
    predictor_classes = {}
    stride_conv_classes = {}
    tokenizer_classes = {}
    batch_sampler_classes = {}
    dataset_classes = {}
    index_ds_classes = {}

    def print(self, key=None):
        print("\ntables: \n")
        fields = vars(self)
        for classes_key, classes_dict in fields.items():

            flag = True
            if key is not None:
                flag = key in classes_key
            if classes_key.endswith("_meta") and flag:
                print(
                    f"-----------    ** {classes_key.replace('_meta', '')} **    --------------"
                )
                headers = ["class name", "class location"]
                metas = []
                for register_key, meta in classes_dict.items():
                    metas.append(meta)
                metas.sort(key=lambda x: x[0])
                data = [headers] + metas
                col_widths = [max(len(str(item)) for item in col) for col in zip(*data)]

                for row in data:
                    print(
                        "| "
                        + " | ".join(
                            str(item).ljust(width)
                            for item, width in zip(row, col_widths)
                        )
                        + " |"
                    )
        print("\n")

    def register(self, register_tables_key: str, key=None):
        def decorator(target_class):

            if not hasattr(self, register_tables_key):
                setattr(self, register_tables_key, {})
                logging.info(
                    "new registry table has been added: {}".format(register_tables_key)
                )

            registry = getattr(self, register_tables_key)
            registry_key = key if key is not None else target_class.__name__

            # assert not registry_key in registry, "(key: {} / class: {}) has been registered already，in {}".format(
            #     registry_key, target_class, register_tables_key)

            registry[registry_key] = target_class

            # meta， headers = ["class name", "register name", "class location"]
            register_tables_key_meta = register_tables_key + "_meta"
            if not hasattr(self, register_tables_key_meta):
                setattr(self, register_tables_key_meta, {})
            registry_meta = getattr(self, register_tables_key_meta)
            # doc = target_class.__doc__
            class_file = inspect.getfile(target_class)
            class_line = inspect.getsourcelines(target_class)[1]
            pattern = r"^.+/funasr/"
            class_file = re.sub(pattern, "funasr/", class_file)
            meata_data = [f"{target_class.__name__}", f"{class_file}:{class_line}"]
            # meata_data = [f"{target_class.__name__}", f"{registry_key}", f"{class_file}:{class_line}"]
            registry_meta[registry_key] = meata_data
            # print(f"Registering class: {class_file}:{class_line} - {target_class.__name__} as {registry_key}")
            return target_class

        return decorator