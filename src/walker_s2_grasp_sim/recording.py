from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np


def _quarantine_path(path: Path, dataset_root: Path, stamp: str) -> Path:
    relative = path.relative_to(dataset_root)
    target = dataset_root / "_quarantine" / stamp / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)
    return target


def _repair_incomplete_dataset_tail(root: Path) -> None:
    """Move incomplete tail files away before LeRobot loads all parquet files."""
    info_path = root / "meta" / "info.json"
    episodes_root = root / "meta" / "episodes"
    data_root = root / "data"
    if not info_path.exists() or not episodes_root.exists() or not data_root.exists():
        return

    import pyarrow.parquet as pq

    episode_rows = []
    episode_files = sorted(episodes_root.glob("chunk-*/*.parquet"))
    referenced_data_files: set[Path] = set()
    for episode_path in episode_files:
        table = pq.read_table(episode_path)
        columns = table.to_pydict()
        for row_index in range(table.num_rows):
            row = {key: values[row_index] for key, values in columns.items()}
            episode_rows.append(row)
            referenced_data_files.add(
                root
                / "data"
                / f"chunk-{int(row['data/chunk_index']):03d}"
                / f"file-{int(row['data/file_index']):03d}.parquet"
            )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantined = []
    for data_path in sorted(data_root.glob("chunk-*/*.parquet")):
        should_quarantine = data_path not in referenced_data_files
        if not should_quarantine:
            try:
                pq.read_metadata(data_path)
            except Exception:
                should_quarantine = True
        if should_quarantine:
            quarantined.append((data_path, _quarantine_path(data_path, root, stamp)))

    info = json.loads(info_path.read_text())
    total_episodes = len(episode_rows)
    total_frames = max((int(row["dataset_to_index"]) for row in episode_rows), default=0)
    needs_info_repair = (
        int(info.get("total_episodes", -1)) != total_episodes
        or int(info.get("total_frames", -1)) != total_frames
        or info.get("splits") != {"train": f"0:{total_episodes}"}
    )
    if needs_info_repair:
        info["total_episodes"] = total_episodes
        info["total_frames"] = total_frames
        info["splits"] = {"train": f"0:{total_episodes}"}
        if "total_chunks" in info:
            info["total_chunks"] = len({int(row["data/chunk_index"]) for row in episode_rows})
        if "total_files" in info:
            info["total_files"] = len(referenced_data_files)
        info_path.write_text(json.dumps(info, indent=4) + "\n")

    if quarantined or needs_info_repair:
        print(
            "[WARN] Repaired incomplete LeRobot dataset tail before append: "
            f"episodes={total_episodes}, frames={total_frames}, quarantined={len(quarantined)}"
        )
        for old, new in quarantined:
            print(f"[WARN] Quarantined incomplete parquet: {old} -> {new}")

        try:
            from src.lerobot.datasets.compute_stats import aggregate_stats
            from src.lerobot.datasets.utils import cast_stats_to_numpy, unflatten_dict, write_stats

            stats_list = []
            for row in episode_rows:
                flat_stats = {
                    key.removeprefix("stats/"): value
                    for key, value in row.items()
                    if key.startswith("stats/")
                }
                stats_list.append(cast_stats_to_numpy(unflatten_dict(flat_stats)))
            if stats_list:
                write_stats(aggregate_stats(stats_list), root)
        except Exception as exc:
            print(f"[WARN] Could not recompute repaired dataset stats automatically: {exc}")


class WalkerS2LeRobotRecorder:
    """Small LeRobot v3 episode writer for the standalone Walker grasp sim."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        fps: int,
        dof_names: list[str],
        image_shape: tuple[int, int, int],
        task: str,
        image_keys: tuple[str, ...] = ("head_left", "head_right"),
    ):
        from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.root = Path(root).expanduser()
        self.fps = int(fps)
        self.dof_names = list(dof_names)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.task = task
        self.image_keys = tuple(image_keys)
        self.frame_count = 0
        self._warned_missing_images: set[str] = set()

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.dof_names),),
                "names": self.dof_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(self.dof_names),),
                "names": self.dof_names,
            },
        }
        for key in self.image_keys:
            features[f"observation.images.{key}"] = {
                "dtype": "image",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            }

        if (self.root / "meta" / "info.json").exists():
            _repair_incomplete_dataset_tail(self.root)
            self.dataset = LeRobotDataset(
                repo_id=self.repo_id,
                root=self.root,
            )
            if int(self.dataset.fps) != self.fps:
                raise ValueError(
                    f"Existing dataset FPS is {self.dataset.fps}, requested {self.fps}"
                )
            print(
                "[INFO] Appending to existing LeRobot dataset: "
                f"{self.root} (episodes={self.dataset.meta.total_episodes})"
            )
        else:
            if self.root.exists() and not any(self.root.iterdir()):
                self.root.rmdir()
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.root,
                robot_type="walker_s2_grasp_sim",
                features=features,
                use_videos=False,
                image_writer_threads=max(1, 2 * len(self.image_keys)),
            )
            print(f"[INFO] Created new LeRobot dataset: {self.dataset.root}")
        self.root = self.dataset.root

    def _image_or_blank(self, frames: dict[str, np.ndarray], key: str) -> np.ndarray:
        frame = frames.get(key)
        if frame is None:
            if key not in self._warned_missing_images:
                self._warned_missing_images.add(key)
                print(f"[WARN] Recording blank frames until camera image is available: {key}")
            return np.zeros(self.image_shape, dtype=np.uint8)

        image = np.asarray(frame)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        if image.dtype != np.uint8:
            if image.size and float(np.nanmax(image)) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        if tuple(image.shape) != self.image_shape:
            raise ValueError(
                f"Camera frame {key} has shape {image.shape}, expected {self.image_shape}"
            )
        return np.ascontiguousarray(image)

    def add_frame(self, observation_state, action, camera_frames: dict[str, np.ndarray]) -> None:
        frame = {
            "observation.state": np.asarray(observation_state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "task": self.task,
        }
        for key in self.image_keys:
            frame[f"observation.images.{key}"] = self._image_or_blank(camera_frames, key)
        self.dataset.add_frame(frame)
        self.frame_count += 1

    def save_episode(self) -> None:
        if self.frame_count <= 0:
            print("[WARN] No recorded frames; skipping dataset save.")
            return
        self.dataset.save_episode()
        print(f"[INFO] Saved LeRobot episode with {self.frame_count} frames to: {self.root}")
        self.frame_count = 0

    def discard_episode(self) -> None:
        if self.frame_count <= 0:
            return
        self.dataset.clear_episode_buffer(delete_images=True)
        print(f"[INFO] Discarded unsaved LeRobot episode with {self.frame_count} frames")
        self.frame_count = 0

    def finalize(self) -> None:
        self.dataset.finalize()

    def save(self) -> None:
        self.save_episode()
        self.finalize()


class WalkerS2LeRobotReplay:
    """Read action vectors from a local LeRobot dataset episode."""

    def __init__(self, repo_id: str, root: str | Path):
        from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(repo_id=repo_id, root=Path(root).expanduser())

    def iter_actions(self, episode_index: int = 0):
        episodes = self.dataset.meta.episodes
        if episodes is None or len(episodes) <= int(episode_index):
            raise IndexError(f"Episode {episode_index} does not exist in dataset")
        episode = episodes[int(episode_index)]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        for index in range(start, end):
            item = self.dataset[index]
            action = item["action"]
            if hasattr(action, "detach"):
                action = action.detach().cpu().numpy()
            yield np.asarray(action, dtype=float)
