# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opaque runtime-only vocoder CUDA Graph Handle."""

from collections.abc import Callable
from typing import Any


class VocoderGraphHandle:
    """Forward one ordinary call to a Manager-assembled runtime callable."""

    __slots__ = ("_call",)

    def __init__(self, call: Callable[..., Any]) -> None:
        self._call = call

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*args, **kwargs)
