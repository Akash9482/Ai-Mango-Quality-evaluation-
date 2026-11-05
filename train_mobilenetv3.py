from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.ensemble import IsolationForest
import joblib

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow is not installed for this Python interpreter:\n"
        f"  {sys.executable} -m pip install tensorflow\n\n"
        "If you are on Apple Silicon, also install:\n"
        f"  {sys.executable} -m pip install tensorflow-metal"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "augmented_images"
DEFAULT_VAL_DIR = PROJECT_ROOT / "dataset_split" / "validation"
DEFAULT_TEST_DIR = PROJECT_ROOT / "dataset_split" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mobilenetv3_runs"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AUTOTUNE = tf.data.AUTOTUNE


# This script trains a MobileNetV3 image classifier for two classes: good_images and bad_images.
# The workflow is:
# 1. Read image folders and assign labels.
# 2. Create TensorFlow datasets for training, validation, and testing.
# 3. Train a one-class "is this even a mango" out-of-distribution (OOD) detector,
#    using ONLY the training images, so that non-mango images can later be rejected
#    as "invalid_image" instead of being forced into good_images/bad_images.
# 4. Build a pretrained MobileNetV3 model and replace the final classification layer.
# 5. Train the model in two phases: warmup and fine-tuning.
# 6. Save the best model, final model, predictions, metrics, extracted features,
#    and the OOD detector.
#
# The model is saved in the Keras format using model.save(...). In this script,
# the final trained model is saved as:
#   mobilenetv3_runs/final_model.keras
# and the best checkpoint during training is saved as:
#   mobilenetv3_runs/best_model.keras
# and the OOD detector is saved as:
#   mobilenetv3_runs/ood_detector.joblib

def parse_args() -> argparse.Namespace:
    """Parse command-line options for dataset paths, model settings, and training hyperparameters."""
    parser = argparse.ArgumentParser(description="Train MobileNetV3 on the prepared image dataset.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-size", choices=["small", "large"], default="small")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument(
        "epoch_count",
        nargs="?",
        type=int,
        help="Optional positional warmup epoch count, equivalent to --warmup-epochs.",
    )
    parser.add_argument(
        "--ood-contamination",
        type=float,
        default=0.01,
        help="Expected fraction of noisy/outlier images inside your OWN training set. "
             "Keep this small (0.01-0.05). Higher values make the invalid-image "
             "detector stricter and may start rejecting real mango images.",
    )
    args = parser.parse_args()
    if args.epoch_count is not None:
        args.warmup_epochs = args.epoch_count
    return args


def configure_runtime(seed: int, use_mixed_precision: bool) -> None:
    """Set global seeds and TensorFlow runtime options for deterministic training."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if use_mixed_precision and tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def image_files(root: Path) -> list[Path]:
    """Return a sorted list of images under a root directory with supported extensions."""
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        files.append(path)

    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No images with supported extensions {sorted(SUPPORTED_EXTENSIONS)} were found under: {root}\n"
            "Put images inside class-named subfolders (e.g. good_images/, bad_images/) or pass --train-dir/--val-dir/--test-dir to point to your data."
        )
    return files


def label_from_path(path: Path, root: Path) -> str:
    """Extract the class label from the image folder structure under the root path."""
    relative_parent = path.parent.relative_to(root)
    return str(relative_parent) if str(relative_parent) != "." else "unlabeled"


def collect_split(root: Path, class_names: list[str] | None = None) -> tuple[list[str], list[int], list[str]]:
    """Collect file paths, numeric labels, and class names for a dataset split."""
    paths = image_files(root)
    labels = [label_from_path(path, root) for path in paths]

    if class_names is None:
        class_names = sorted(set(labels))

    class_to_index = {name: index for index, name in enumerate(class_names)}
    unknown_labels = sorted(set(labels) - set(class_names))
    if unknown_labels:
        raise ValueError(f"{root} has labels not present in training data: {unknown_labels}")

    encoded_labels = [class_to_index[label] for label in labels]
    return [str(path) for path in paths], encoded_labels, class_names


def decode_and_resize(path: tf.Tensor, label: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(
    paths: list[str],
    labels: list[int],
    image_size: int,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    """Prepare a tf.data.Dataset from image paths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        # For training, shuffle the dataset to randomize image order each epoch.
        dataset = dataset.shuffle(len(paths), reshuffle_each_iteration=True)

    dataset = dataset.map(lambda path, label: decode_and_resize(path, label, image_size), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def class_weights(labels: list[int]) -> dict[int, float]:
    """Compute sample weights so rare classes count more during training."""
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)
    return {class_id: total / (num_classes * count) for class_id, count in counts.items()}


def build_model(
    image_size: int,
    num_classes: int,
    model_size: str,
    dropout: float,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build the MobileNetV3 feature extractor and classification head."""
    # A neural network needs an input shape. We tell TensorFlow that every image is 224x224 RGB.
    input_layer = tf.keras.Input(shape=(image_size, image_size, 3))

    # Choose which MobileNetV3 version to use. "small" is lighter and faster; "large" is heavier.
    mobilenet_cls = (
        tf.keras.applications.MobileNetV3Small
        if model_size == "small"
        else tf.keras.applications.MobileNetV3Large
    )

    # Load a pretrained model from ImageNet. This is transfer learning:
    # instead of training from random weights, we reuse features learned from millions of images.
    base_model = mobilenet_cls(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
    )

    # Freeze the pretrained base model first. We only train the new classification head at the start.
    base_model.trainable = False

    # Step 1: Extract image features from the pretrained MobileNetV3 base.
    x = base_model(input_layer, training=False)

    # Step 2: Convert the feature maps to a fixed-size vector for each image.
    # GlobalAveragePooling2D reduces the spatial dimensions into one compact feature vector.
    x = tf.keras.layers.GlobalAveragePooling2D(name="cnn_features")(x)

    # Step 3: Apply dropout to reduce overfitting and make the model more robust.
    x = tf.keras.layers.Dropout(dropout)(x)

    # Step 4: Predict class probabilities with the softmax output layer.
    # The network outputs one probability for each class.
    output_layer = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="softmax_layer",
    )(x)
    return tf.keras.Model(input_layer, output_layer), base_model


# ---------------------------------------------------------------------------
# NEW: out-of-distribution ("is this even a mango") detector.
#
# This uses a SEPARATE, frozen, general-purpose ImageNet feature extractor
# (never fine-tuned on your mango data) so it keeps broad visual knowledge
# that is useful for telling "mango" apart from "everything else" — knowledge
# that your fine-tuned good/bad classifier is not required to preserve.
#
# It is fit using ONLY your existing mango images (no negative/invalid
# examples needed), by learning the shape of the mango feature distribution
# and flagging inputs that fall far outside it.
# ---------------------------------------------------------------------------

def build_ood_feature_extractor(image_size: int) -> tf.keras.Model:
    """A frozen, general-purpose ImageNet feature extractor — never fine-tuned,
    used only for out-of-distribution (non-mango) detection."""
    base = tf.keras.applications.MobileNetV3Large(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
        pooling="avg",
    )
    base.trainable = False
    return base


def extract_ood_features(extractor: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    """Run images through the frozen general-purpose extractor."""
    features = []
    for images, _ in dataset:
        features.append(extractor(images, training=False).numpy())
    return np.concatenate(features, axis=0)


def fit_ood_detector(train_features: np.ndarray, contamination: float = 0.01) -> IsolationForest:
    """Fit a one-class anomaly detector using ONLY mango (in-distribution) features."""
    detector = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    detector.fit(train_features)
    return detector


def save_ood_detector(
    detector: IsolationForest,
    output_dir: Path,
    threshold: float,
    centroid: np.ndarray | None = None,
    distance_threshold: float | None = None,
    confidence_threshold: float | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "detector": detector,
        "threshold": float(threshold),
    }
    if centroid is not None:
        payload["centroid"] = centroid
    if distance_threshold is not None:
        payload["distance_threshold"] = float(distance_threshold)
    if confidence_threshold is not None:
        payload["confidence_threshold"] = float(confidence_threshold)

    joblib.dump(payload, output_dir / "ood_detector.joblib")


def load_ood_detector(output_dir: Path) -> tuple[IsolationForest, float, np.ndarray | None, float | None, float | None]:
    data = joblib.load(output_dir / "ood_detector.joblib")
    if isinstance(data, dict) and "detector" in data and "threshold" in data:
        return (
            data["detector"],
            float(data["threshold"]),
            data.get("centroid"),
            data.get("distance_threshold"),
            data.get("confidence_threshold"),
        )
    return data, 0.0, None, None, None


# ---------------------------------------------------------------------------
# end OOD section
# ---------------------------------------------------------------------------


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )


def callbacks(output_dir: Path) -> list[tf.keras.callbacks.Callback]:
    """Create training callbacks that monitor progress and save useful checkpoints."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        # Save the best model during training based on validation accuracy.
        # This creates a checkpoint file named best_model.keras.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        # Stop training early if validation accuracy stops improving.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            mode="max",
            restore_best_weights=True,
        ),
        # If the model stops improving, reduce the learning rate automatically.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
        ),
        # Record training history to a CSV file for analysis.
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]


def save_class_names(class_names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "class_names.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)
        file.write("\n")


def save_test_predictions(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    test_paths: list[str],
    test_labels: list[int],
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Save the model's predicted labels and confidence for each test image."""
    probabilities = model.predict(test_dataset, verbose=0)
    predicted_labels = probabilities.argmax(axis=1)

    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "true_label", "predicted_label", "confidence"])
        for path, true_id, pred_id, probability_row in zip(test_paths, test_labels, predicted_labels, probabilities):
            writer.writerow(
                [
                    path,
                    class_names[true_id],
                    class_names[int(pred_id)],
                    f"{float(probability_row[pred_id]):.6f}",
                ]
            )


def save_softmax_probabilities(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    test_paths: list[str],
    test_labels: list[int],
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Save softmax probability scores for every test image and every class."""
    probabilities = model.predict(test_dataset, verbose=0)

    with (output_dir / "softmax_probabilities.csv").open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Create one CSV column for each class probability.
        probability_columns = [f"probability_{class_name}" for class_name in class_names]
        writer.writerow(["image_path", "true_label", "true_label_number", *probability_columns])

        for path, true_id, probability_row in zip(test_paths, test_labels, probabilities):
            writer.writerow(
                [
                    path,
                    class_names[true_id],
                    true_id,
                    *[f"{float(value):.6f}" for value in probability_row],
                ]
            )


def save_cnn_features(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    image_paths: list[str],
    labels: list[int],
    class_names: list[str],
    output_dir: Path,
    split_name: str,
) -> None:
    """Save the internal CNN features from the named feature extraction layer."""
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("cnn_features").output,
    )
    features = feature_extractor.predict(dataset, verbose=0)

    with (output_dir / f"{split_name}_cnn_features.csv").open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Each feature column stores one number from the CNN feature vector.
        feature_columns = [f"feature_{index}" for index in range(features.shape[1])]
        writer.writerow(["image_path", "label", "label_number", *feature_columns])

        for path, label_id, feature_row in zip(image_paths, labels, features):
            writer.writerow(
                [
                    path,
                    class_names[label_id],
                    label_id,
                    *[f"{float(value):.6f}" for value in feature_row],
                ]
            )


def main() -> None:
    """Execute the full training workflow from dataset loading to evaluation and report saving."""
    # Step 1: Read the command-line arguments chosen for this run.
    args = parse_args()

    # Step 2: Make the training process more reproducible by setting seeds.
    # Also enable GPU memory growth if a GPU is available.
    configure_runtime(args.seed, use_mixed_precision=not args.no_mixed_precision)

    # Step 3: Load the train, validation, and test image folders from disk.
    # The training split defines the class order, which is then reused for validation and test.
    train_paths, train_labels, class_names = collect_split(args.train_dir)
    val_paths, val_labels, _ = collect_split(args.val_dir, class_names)
    test_paths, test_labels, _ = collect_split(args.test_dir, class_names)

    # Step 4: Build TensorFlow datasets that can read images lazily, resize them,
    # batch them, and prepare them efficiently for training.
    train_dataset = make_dataset(train_paths, train_labels, args.img_size, args.batch_size, training=True)
    val_dataset = make_dataset(val_paths, val_labels, args.img_size, args.batch_size, training=False)
    test_dataset = make_dataset(test_paths, test_labels, args.img_size, args.batch_size, training=False)

    # Step 4b (NEW): Train a one-class "is this even a mango" out-of-distribution (OOD)
    # detector using ONLY the mango training images. This uses a separate, frozen,
    # general-purpose ImageNet feature extractor (not fine-tuned), fit with an
    # IsolationForest so it can flag inputs that don't look like mangoes at all,
    # without needing any "invalid image" examples during training.
    print("Training out-of-distribution (invalid image) detector...")
    ood_extractor = build_ood_feature_extractor(args.img_size)
    train_ood_features = extract_ood_features(ood_extractor, train_dataset)
    ood_detector = fit_ood_detector(train_ood_features, contamination=args.ood_contamination)
    ood_threshold = float(getattr(ood_detector, "threshold_", getattr(ood_detector, "offset_", 0.0)))
    ood_centroid = train_ood_features.mean(axis=0)
    ood_distances = np.linalg.norm(train_ood_features - ood_centroid, axis=1)
    ood_distance_threshold = float(np.percentile(ood_distances, 99.0))
    save_ood_detector(
        ood_detector,
        args.output_dir,
        ood_threshold,
        centroid=ood_centroid,
        distance_threshold=ood_distance_threshold,
    )
    print("OOD detector saved to:", args.output_dir / "ood_detector.joblib")
    print(f"OOD threshold set to: {ood_threshold:.6f}")
    print(f"OOD distance threshold set to: {ood_distance_threshold:.6f}")

    # Step 5: Create the MobileNetV3 model and compile it with a loss function and metrics.
    model, base_model = build_model(args.img_size, len(class_names), args.model_size, args.dropout)
    compile_model(model, args.learning_rate)

    # Save the class names so that predictions can be mapped back to human-readable labels later.
    save_class_names(class_names, args.output_dir)

    print(f"Classes: {len(class_names)}")
    print(f"Train images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Test images: {len(test_paths)}")

    # Step 6: Compute class weights to handle imbalance between classes.
    weights = class_weights(train_labels)

    # Step 7: Train the new classification head while the pretrained base model stays frozen.
    # This is the warmup phase, where the model learns to classify using the pretrained features.
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.warmup_epochs,
        class_weight=weights,
        callbacks=callbacks(args.output_dir),
    )

    # Step 8: Unfreeze part of the pretrained base model and fine-tune it.
    # This allows the model to adjust to your specific dataset instead of only using ImageNet features.
    base_model.trainable = True
    fine_tune_from = int(len(base_model.layers) * 0.65)
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Use a smaller learning rate during fine-tuning so the model does not forget what it learned.
    compile_model(model, args.fine_tune_learning_rate)
    total_epochs = args.warmup_epochs + args.fine_tune_epochs

    # Step 9: Continue training for more epochs, but with a lower learning rate and partial fine-tuning.
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=args.warmup_epochs,
        epochs=total_epochs,
        class_weight=weights,
        callbacks=callbacks(args.output_dir),
    )

    # Step 10: Save the fully trained model in Keras format.
    # This creates a single portable file that can be reloaded later with tf.keras.models.load_model.
    model.save(args.output_dir / "final_model.keras")

    train_probabilities = model.predict(train_dataset, verbose=0)
    confidence_threshold = float(np.percentile(np.max(train_probabilities, axis=1), 1.0))
    save_ood_detector(
        ood_detector,
        args.output_dir,
        ood_threshold,
        centroid=ood_centroid,
        distance_threshold=ood_distance_threshold,
        confidence_threshold=confidence_threshold,
    )
    print(f"OOD confidence threshold set to: {confidence_threshold:.6f}")

    # Step 11: Evaluate the trained model on the test split and write the metrics to JSON.
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump({key: float(value) for key, value in test_metrics.items()}, file, indent=2)
        file.write("\n")

    # Step 12: Save predictions and confidence scores so you can inspect individual examples.
    save_test_predictions(model, test_dataset, test_paths, test_labels, class_names, args.output_dir)
    save_softmax_probabilities(model, test_dataset, test_paths, test_labels, class_names, args.output_dir)

    # Step 13: Extract internal CNN feature vectors for later analysis or visualization.
    save_cnn_features(model, train_dataset, train_paths, train_labels, class_names, args.output_dir, "train")
    save_cnn_features(model, val_dataset, val_paths, val_labels, class_names, args.output_dir, "validation")
    save_cnn_features(model, test_dataset, test_paths, test_labels, class_names, args.output_dir, "test")

    print(f"Saved model and reports to: {args.output_dir}")
    print("CNN feature reports: train_cnn_features.csv, validation_cnn_features.csv, test_cnn_features.csv")
    print("Softmax Layer report: softmax_probabilities.csv")
    print("OOD (invalid image) detector: ood_detector.joblib")
    print("Test metrics:", {key: round(float(value), 4) for key, value in test_metrics.items()})


if __name__ == "__main__":
    main()
