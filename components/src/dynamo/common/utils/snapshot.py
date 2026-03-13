# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Dynamo snapshot helpers for checkpoint lifecycle and discovery identity restoration."""

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_PODINFO_DIR = Path("/etc/podinfo")
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveryIdentityRestoration:
    namespace: str
    discovery_backend: str


class CheckpointConfig:
    """Parsed checkpoint configuration plus the watcher-driven lifecycle."""

    def __init__(self, ready_file: str, storage_type: str, location: str):
        self.ready_file = ready_file
        self.storage_type = storage_type
        self.location = location
        self._checkpoint_done = asyncio.Event()
        self._restore_done = asyncio.Event()

    @classmethod
    def from_env(cls) -> "CheckpointConfig | None":
        ready_file = os.environ.get("DYN_READY_FOR_CHECKPOINT_FILE")
        if not ready_file:
            return None

        location = os.environ.get("DYN_CHECKPOINT_LOCATION", "")
        if not location:
            checkpoint_path = os.environ.get("DYN_CHECKPOINT_PATH", "").rstrip("/")
            checkpoint_hash = os.environ.get("DYN_CHECKPOINT_HASH", "")
            if not checkpoint_path or not checkpoint_hash:
                raise EnvironmentError(
                    "Checkpoint mode requires either DYN_CHECKPOINT_LOCATION or both "
                    "DYN_CHECKPOINT_PATH and DYN_CHECKPOINT_HASH"
                )
            location = f"{checkpoint_path}/{checkpoint_hash}"

        return cls(
            ready_file=ready_file,
            storage_type=os.environ.get("DYN_CHECKPOINT_STORAGE_TYPE", "pvc"),
            location=location,
        )

    def checkpoint_exists(self) -> bool:
        if self.storage_type != "pvc":
            return False

        if os.path.isdir(self.location):
            _LOG.info("Existing checkpoint found at %s, skipping", self.location)
            return True

        _LOG.info("No checkpoint at %s, creating new one", self.location)
        return False

    async def run_lifecycle(self, engine_client: Any, sleep_level: int) -> bool:
        _LOG.info("Putting model to sleep (level=%s)", sleep_level)
        await engine_client.sleep(level=sleep_level)

        self._install_signal_handlers()

        with open(self.ready_file, "w", encoding="utf-8") as ready_file:
            ready_file.write("ready")
        _LOG.info(
            "Ready for checkpoint. Waiting for watcher signal "
            "(SIGUSR1=checkpoint complete, SIGCONT=restore complete)"
        )

        try:
            event = await self._wait_for_watcher_signal()
            if event == "restore":
                _LOG.info("Restore signal detected (SIGCONT)")
                _LOG.info("Waking up model after restore")
                await engine_client.wake_up()
                return True

            _LOG.info("Checkpoint completion signal detected (SIGUSR1)")
            return False
        finally:
            self._remove_signal_handlers()
            try:
                os.unlink(self.ready_file)
            except OSError:
                pass

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGUSR1, self._checkpoint_done.set)
        loop.add_signal_handler(signal.SIGCONT, self._restore_done.set)

    def _remove_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        loop.remove_signal_handler(signal.SIGUSR1)
        loop.remove_signal_handler(signal.SIGCONT)

    async def _wait_for_watcher_signal(self) -> str:
        waiters = {
            asyncio.create_task(self._checkpoint_done.wait()): "checkpoint",
            asyncio.create_task(self._restore_done.wait()): "restore",
        }
        try:
            done, pending = await asyncio.wait(
                waiters.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            winner = done.pop()
            await winner
            return waiters[winner]
        finally:
            for task in waiters:
                if not task.done():
                    task.cancel()


def _read_required_podinfo_file(podinfo_dir: Path, file_name: str) -> str:
    file_path = podinfo_dir / file_name
    if not file_path.is_file():
        raise RuntimeError(
            "Discovery identity restoration requires "
            f"/etc/podinfo/{file_name} to be populated"
        )

    value = file_path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(
            "Discovery identity restoration requires "
            f"/etc/podinfo/{file_name} to be populated"
        )
    return value


def load_discovery_identity_restoration(
    podinfo_dir: str | Path = _DEFAULT_PODINFO_DIR,
) -> DiscoveryIdentityRestoration:
    """Load the current restore target discovery identity from the Downward API volume."""

    podinfo_path = Path(podinfo_dir)
    return DiscoveryIdentityRestoration(
        namespace=_read_required_podinfo_file(podinfo_path, "dyn_namespace"),
        discovery_backend=_read_required_podinfo_file(
            podinfo_path, "dyn_discovery_backend"
        ),
    )


def apply_discovery_identity_restoration(
    config: Any,
    podinfo_dir: str | Path = _DEFAULT_PODINFO_DIR,
) -> DiscoveryIdentityRestoration:
    """Override stale post-restore discovery identity with the current pod identity."""

    if not hasattr(config, "namespace"):
        raise AttributeError(
            "discovery identity restoration target must have a namespace"
        )
    if not hasattr(config, "discovery_backend"):
        raise AttributeError(
            "discovery identity restoration target must have a discovery_backend"
        )

    identity = load_discovery_identity_restoration(podinfo_dir)
    config.namespace = identity.namespace
    config.discovery_backend = identity.discovery_backend
    return identity


def get_checkpoint_config() -> tuple[bool, CheckpointConfig | None]:
    """Resolve checkpoint mode for checkpoint-job pods."""

    cfg = CheckpointConfig.from_env()
    if cfg is None:
        return False, None

    checkpoint_exists = cfg.checkpoint_exists()
    if checkpoint_exists:
        return True, None

    return False, cfg
