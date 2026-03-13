# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamo.common.utils.snapshot import (
    CheckpointConfig,
    apply_discovery_identity_restoration,
    get_checkpoint_config,
    load_discovery_identity_restoration,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_load_discovery_identity_restoration_reads_current_podinfo(tmp_path):
    (tmp_path / "dyn_namespace").write_text("new-namespace\n", encoding="utf-8")
    (tmp_path / "dyn_discovery_backend").write_text(
        "kubernetes\n", encoding="utf-8"
    )

    identity = load_discovery_identity_restoration(tmp_path)

    assert identity.namespace == "new-namespace"
    assert identity.discovery_backend == "kubernetes"


def test_apply_discovery_identity_restoration_overrides_stale_restore_values(tmp_path):
    (tmp_path / "dyn_namespace").write_text("restored-ns\n", encoding="utf-8")
    (tmp_path / "dyn_discovery_backend").write_text("kubernetes\n", encoding="utf-8")

    config = SimpleNamespace(namespace="old-ns", discovery_backend="etcd")

    identity = apply_discovery_identity_restoration(config, tmp_path)

    assert identity.namespace == "restored-ns"
    assert config.namespace == "restored-ns"
    assert config.discovery_backend == "kubernetes"


def test_apply_discovery_identity_restoration_requires_podinfo_files(tmp_path):
    (tmp_path / "dyn_namespace").write_text("restored-ns\n", encoding="utf-8")

    config = SimpleNamespace(namespace="old-ns", discovery_backend="etcd")

    with pytest.raises(
        RuntimeError,
        match=r"/etc/podinfo/dyn_discovery_backend",
    ):
        apply_discovery_identity_restoration(config, tmp_path)


def test_get_checkpoint_config_returns_none_outside_checkpoint_mode(monkeypatch):
    monkeypatch.delenv("DYN_READY_FOR_CHECKPOINT_FILE", raising=False)

    should_exit, checkpoint_cfg = get_checkpoint_config()

    assert should_exit is False
    assert checkpoint_cfg is None


def test_get_checkpoint_config_exits_when_checkpoint_already_exists(
    monkeypatch, tmp_path
):
    checkpoint_dir = tmp_path / "checkpoints" / "abc123"
    checkpoint_dir.mkdir(parents=True)

    monkeypatch.setenv("DYN_READY_FOR_CHECKPOINT_FILE", str(tmp_path / "ready"))
    monkeypatch.setenv("DYN_CHECKPOINT_PATH", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("DYN_CHECKPOINT_HASH", "abc123")
    monkeypatch.delenv("DYN_CHECKPOINT_LOCATION", raising=False)

    should_exit, checkpoint_cfg = get_checkpoint_config()

    assert should_exit is True
    assert checkpoint_cfg is None


def test_get_checkpoint_config_derives_location_from_path_and_hash(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DYN_READY_FOR_CHECKPOINT_FILE", str(tmp_path / "ready"))
    monkeypatch.setenv("DYN_CHECKPOINT_PATH", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("DYN_CHECKPOINT_HASH", "abc123")
    monkeypatch.delenv("DYN_CHECKPOINT_LOCATION", raising=False)

    should_exit, checkpoint_cfg = get_checkpoint_config()

    assert should_exit is False
    assert checkpoint_cfg is not None
    assert checkpoint_cfg.location == str(tmp_path / "checkpoints" / "abc123")


class FakeCheckpointEngine:
    def __init__(self):
        self.calls: list[tuple[str, int | None]] = []

    async def sleep(self, level: int = 1) -> None:
        self.calls.append(("sleep", level))

    async def wake_up(self) -> None:
        self.calls.append(("wake_up", None))


@pytest.mark.asyncio
async def test_checkpoint_config_run_lifecycle_restores_and_cleans_ready_file(
    monkeypatch, tmp_path
):
    ready_file = tmp_path / "ready"
    cfg = CheckpointConfig(str(ready_file), "pvc", str(tmp_path / "checkpoint"))
    engine = FakeCheckpointEngine()

    monkeypatch.setattr(cfg, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(cfg, "_remove_signal_handlers", lambda: None)

    async def wait_for_restore() -> str:
        assert ready_file.read_text(encoding="utf-8") == "ready"
        return "restore"

    monkeypatch.setattr(cfg, "_wait_for_watcher_signal", wait_for_restore)

    restored = await cfg.run_lifecycle(engine, sleep_level=3)

    assert restored is True
    assert engine.calls == [("sleep", 3), ("wake_up", None)]
    assert not ready_file.exists()


@pytest.mark.asyncio
async def test_checkpoint_config_run_lifecycle_exits_after_checkpoint(
    monkeypatch, tmp_path
):
    ready_file = tmp_path / "ready"
    cfg = CheckpointConfig(str(ready_file), "pvc", str(tmp_path / "checkpoint"))
    engine = FakeCheckpointEngine()

    monkeypatch.setattr(cfg, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(cfg, "_remove_signal_handlers", lambda: None)
    monkeypatch.setattr(
        cfg,
        "_wait_for_watcher_signal",
        lambda: _return_completed_checkpoint(ready_file),
    )

    restored = await cfg.run_lifecycle(engine, sleep_level=1)

    assert restored is False
    assert engine.calls == [("sleep", 1)]
    assert not ready_file.exists()


async def _return_completed_checkpoint(ready_file: Path) -> str:
    assert ready_file.read_text(encoding="utf-8") == "ready"
    return "checkpoint"
