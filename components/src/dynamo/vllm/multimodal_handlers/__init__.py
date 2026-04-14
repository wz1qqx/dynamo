# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Patched for vllm 0.19.0 compatibility: multimodal handlers are optional

try:
    from dynamo.vllm.multimodal_handlers.encode_worker_handler import EncodeWorkerHandler
    from dynamo.vllm.multimodal_handlers.multimodal_pd_worker_handler import (
        MultimodalPDWorkerHandler,
    )
    from dynamo.vllm.multimodal_handlers.worker_handler import MultimodalDecodeWorkerHandler
except (ImportError, AttributeError):
    EncodeWorkerHandler = None
    MultimodalPDWorkerHandler = None
    MultimodalDecodeWorkerHandler = None

__all__ = [
    "EncodeWorkerHandler",
    "MultimodalPDWorkerHandler",
    "MultimodalDecodeWorkerHandler",
]
