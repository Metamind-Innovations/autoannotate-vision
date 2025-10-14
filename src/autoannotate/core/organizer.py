import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


class DatasetOrganizer:

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def organize_by_clusters(
        self,
        image_paths: List[Path],
        labels: np.ndarray,
        class_names: Dict[int, str],
        copy_files: bool = True,
        create_symlinks: bool = False,
    ) -> Dict:
        organized_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_images": len(image_paths),
                "n_classes": len(class_names),
            },
            "classes": {},
        }

        class_counters = {label: 0 for label in class_names.keys()}

        for img_path, label in zip(image_paths, labels):
            label = int(label)

            if label == -1 or label not in class_names:
                noise_dir = self.output_dir / "unclustered"
                noise_dir.mkdir(exist_ok=True)
                dest_path = noise_dir / img_path.name
                if copy_files:
                    shutil.copy2(img_path, dest_path)
                elif create_symlinks:
                    dest_path.symlink_to(img_path.absolute())
                continue

            class_name = class_names[label]
            class_dir = self.output_dir / class_name
            class_dir.mkdir(exist_ok=True)

            dest_path = class_dir / img_path.name

            if dest_path.exists():
                stem = img_path.stem
                ext = img_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = class_dir / f"{stem}_{counter}{ext}"
                    counter += 1

            if copy_files:
                shutil.copy2(img_path, dest_path)
            elif create_symlinks:
                dest_path.symlink_to(img_path.absolute())

            if class_name not in organized_data["classes"]:
                organized_data["classes"][class_name] = {"count": 0, "images": []}

            organized_data["classes"][class_name]["count"] += 1
            organized_data["classes"][class_name]["images"].append(
                {"original_path": str(img_path), "new_path": str(dest_path), "cluster_id": label}
            )

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(organized_data, f, indent=2)

        return organized_data

    def create_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")

        np.random.seed(seed)

        splits_dir = self.output_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        for split_name in ["train", "val", "test"]:
            (splits_dir / split_name).mkdir(exist_ok=True)

        class_dirs = [
            d
            for d in self.output_dir.iterdir()
            if d.is_dir() and d.name not in ["splits", "unclustered"]
        ]

        split_info = {"train": [], "val": [], "test": []}

        for class_dir in class_dirs:
            class_name = class_dir.name
            images = list(class_dir.glob("*"))

            if len(images) == 0:
                continue

            np.random.shuffle(images)

            n_train = int(len(images) * train_ratio)
            n_val = int(len(images) * val_ratio)

            train_imgs = images[:n_train]
            val_imgs = images[n_train : n_train + n_val]
            test_imgs = images[n_train + n_val :]

            for split_name, split_imgs in [
                ("train", train_imgs),
                ("val", val_imgs),
                ("test", test_imgs),
            ]:
                split_class_dir = splits_dir / split_name / class_name
                split_class_dir.mkdir(exist_ok=True)

                for img in split_imgs:
                    dest = split_class_dir / img.name
                    dest.symlink_to(img.absolute())
                    split_info[split_name].append(str(dest))

        split_metadata_path = splits_dir / "split_info.json"
        with open(split_metadata_path, "w") as f:
            json.dump(
                {
                    "train_count": len(split_info["train"]),
                    "val_count": len(split_info["val"]),
                    "test_count": len(split_info["test"]),
                    "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
                },
                f,
                indent=2,
            )

        return split_info

    def export_labels_file(self, format: str = "csv"):
        metadata_path = self.output_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError("metadata.json not found. Run organize_by_clusters first.")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if format == "csv":
            output_path = self.output_dir / "labels.csv"
            with open(output_path, "w") as f:
                f.write("image_path,class_name,cluster_id\n")
                for class_name, class_data in metadata["classes"].items():
                    for img_info in class_data["images"]:
                        f.write(f"{img_info['new_path']},{class_name},{img_info['cluster_id']}\n")

        elif format == "json":
            output_path = self.output_dir / "labels.json"
            labels_data = []
            for class_name, class_data in metadata["classes"].items():
                for img_info in class_data["images"]:
                    labels_data.append(
                        {
                            "image_path": img_info["new_path"],
                            "class_name": class_name,
                            "cluster_id": img_info["cluster_id"],
                        }
                    )

            with open(output_path, "w") as f:
                json.dump(labels_data, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}")

        return output_path
