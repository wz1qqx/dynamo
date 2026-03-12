# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM uses the shared Dynamo snapshot lifecycle helpers directly."""

from dynamo.common.utils.snapshot import CheckpointConfig, get_checkpoint_config

__all__ = ["CheckpointConfig", "get_checkpoint_config"]
