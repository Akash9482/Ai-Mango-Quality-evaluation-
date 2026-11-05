from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("mangoooo")
TRAIN_DIR = Path("augmented_images")
VAL_DIR = Path("dataset_split") / "validation"
TEST_DIR = Path("dataset_split") / "test"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RATIOS = (0.8, 0.1, 0.1)
SEED = 42


def gather_image_files(source_root: Path):
    images = []
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        images.append(path)
    return sorted(images)


def clean_and_create(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_images(paths, ratios):
    total = len(paths)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    return paths[:train_end], paths[train_end:val_end], paths[val_end:]


def copy_split(grade_name: str, images, out_root: Path):
    dest = out_root / grade_name
    dest.mkdir(parents=True, exist_ok=True)
    for path in images:
        shutil.copy2(path, dest / path.name)


def main():
    if not SOURCE_DIR.exists():
        raise SystemExit(f"Source directory not found: {SOURCE_DIR}")

    random.seed(SEED)
    clean_and_create(TRAIN_DIR)
    clean_and_create(VAL_DIR)
    clean_and_create(TEST_DIR)

    for grade_dir in sorted(SOURCE_DIR.iterdir()):
        if not grade_dir.is_dir() or grade_dir.name.startswith("."):
            continue

        image_paths = gather_image_files(grade_dir)
        if not image_paths:
            print(f"Skipping empty grade folder: {grade_dir}")
            continue

        random.shuffle(image_paths)
        train_images, val_images, test_images = split_images(image_paths, RATIOS)

        copy_split(grade_dir.name, train_images, TRAIN_DIR)
        copy_split(grade_dir.name, val_images, VAL_DIR)
        copy_split(grade_dir.name, test_images, TEST_DIR)

        print(f"{grade_dir.name}: total={len(image_paths)}, train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

    print("Dataset split rebuilt:")
    print(f"  train: {TRAIN_DIR}")
    print(f"  val:   {VAL_DIR}")
    print(f"  test:  {TEST_DIR}")


if __name__ == "__main__":
    main()
