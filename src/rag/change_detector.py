import json
from pathlib import Path
from typing import Dict, Tuple, List
import logging
import hashlib


logger = logging.getLogger(__name__)


class ChangeDetector:
    def __init__(self, hash_file_path: str) -> None:
        self.hash_file_path = Path(hash_file_path)

    def compute_file_hashes(self, data_path: str) -> Dict[str, str]:
        data_root = Path(data_path).resolve()
        hashes = {}

        for md_file in data_root.rglob("*.md"):
            try:
                relative_path = md_file.resolve().relative_to(data_root).as_posix()
                file_path = self._compute_md5(md_file)
                hashes[relative_path] = file_path
            except Exception as e:
                logger.error(f"Error processing file {md_file}: {e}")

        return hashes

    def detect_changes(self, data_path: str) -> Tuple[List[str], List[str], List[str]]:
        current_hashes = self.compute_file_hashes(data_path)
        saved_hashes = self.load_hashes()

        added = []
        modified = []
        deleted = []

        for path, hash_val in current_hashes.items():
            if path not in saved_hashes:
                added.append(path)
            elif saved_hashes[path] != hash_val:
                modified.append(path)

        for path in saved_hashes:
            if path not in current_hashes:
                deleted.append(path)

        logger.info(
            f"Change detection results - Added: {added}, Modified: {modified}, Deleted: {deleted}"
        )
        return added, deleted, modified

    def load_hashes(self) -> Dict[str, str]:
        if not self.hash_file_path.exists():
            return {}
        try:
            with open(self.hash_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading hashes from {self.hash_file_path}: {e}")
            return {}

    def save_hashes(self, data_path: str) -> None:
        hashes = self.compute_file_hashes(data_path)
        self.hash_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_file_path, "w", encoding="utf-8") as f:
            json.dump(hashes, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved hashes to {self.hash_file_path}")

    def clear_hashes(self) -> None:
        if self.hash_file_path.exists():
            self.hash_file_path.unlink()
            logger.info(f"Cleared hashes from {self.hash_file_path}")

    @staticmethod
    def _compute_md5(file_path: Path) -> str:
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()
