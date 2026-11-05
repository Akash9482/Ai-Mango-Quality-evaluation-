from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow is not installed. Install it first with:\n"
        "  python3 -m pip install tensorflow"
    ) from exc


# This script loads the model trained by train_mobilenetv3.py and classifies a
# single new image. It first checks whether the image even looks like a mango
# using the saved out-of-distribution (OOD) detector. If it does not, the
# image is reported as "invalid_image" instead of being forced into
# good_images / bad_images. Only images that pass the OOD check are then
# classified by the fine-tuned MobileNetV3 model.

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "mobilenetv3_runs"
DEFAULT_IMG_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a single image, rejecting non-mango inputs.")
    parser.add_argument("image_path", type=Path, help="Path to the image you want to classify.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    return parser.parse_args()


def build_ood_feature_extractor(image_size: int) -> tf.keras.Model:
    """Must match the extractor used during training in train_mobilenetv3.py."""
    base = tf.keras.applications.MobileNetV3Large(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
        pooling="avg",
    )
    base.trainable = False
    return base


def load_and_prepare_image(image_path: Path, image_size: int) -> tf.Tensor:
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
        raw = tf.io.read_file(str(image_path))
        image = tf.io.decode_image(raw, channels=0, expand_animations=False)
    elif suffix == ".nef":
        image = _load_raw_image(image_path)
    else:
        raise SystemExit(f"Unsupported image format: {image_path.suffix}")

    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, axis=0)  # add batch dimension


def _load_raw_image(image_path: Path) -> tf.Tensor:
    import rawpy

    with rawpy.imread(str(image_path)) as raw:
        rgb = raw.postprocess()
    image = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    return image


def predict_single_image(
    image_path: Path,
    classifier: tf.keras.Model,
    ood_extractor: tf.keras.Model,
    ood_detector,
    class_names: list[str],
    image_size: int,
    ood_threshold: float,
    ood_centroid: np.ndarray | None = None,
    ood_distance_threshold: float | None = None,
    confidence_threshold: float | None = None,
) -> dict:
    """Classify one image. Returns 'invalid_image' if it does not look like
    anything the model was trained on (i.e., not a mango)."""
    batched_image = load_and_prepare_image(image_path, image_size)

    ood_features = ood_extractor(batched_image, training=False).numpy()
    anomaly_score = float(ood_detector.decision_function(ood_features)[0])
    is_mango = anomaly_score >= ood_threshold

    distance = None
    if ood_centroid is not None and ood_distance_threshold is not None:
        distance = float(np.linalg.norm(ood_features[0] - ood_centroid))
        if distance > ood_distance_threshold:
            is_mango = False

    if not is_mango:
        return {
            "image_path": str(image_path),
            "label": "invalid_image",
            "confidence": None,
            "anomaly_score": anomaly_score,
            "distance": distance,
        }

    probabilities = classifier.predict(batched_image, verbose=0)[0]
    top_index = int(np.argmax(probabilities))
    confidence = float(probabilities[top_index])

    if confidence_threshold is not None and confidence < confidence_threshold:
        return {
            "image_path": str(image_path),
            "label": "invalid_image",
            "confidence": confidence,
            "anomaly_score": anomaly_score,
            "distance": distance,
        }

    return {
        "image_path": str(image_path),
        "label": class_names[top_index],
        "confidence": confidence,
        "anomaly_score": anomaly_score,
        "distance": distance,
    }


def main() -> None:
    args = parse_args()

    if not args.image_path.exists():
        raise SystemExit(f"Image not found: {args.image_path}")

    class_names_path = args.model_dir / "class_names.json"
    model_path = args.model_dir / "final_model.keras"
    ood_path = args.model_dir / "ood_detector.joblib"

    for required_path in (class_names_path, model_path, ood_path):
        if not required_path.exists():
            raise SystemExit(
                f"Required file not found: {required_path}\n"
                "Make sure you have run train_mobilenetv3.py first."
            )

    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    classifier = tf.keras.models.load_model(model_path)
    ood_data = joblib.load(ood_path)
    if isinstance(ood_data, dict) and "detector" in ood_data and "threshold" in ood_data:
        ood_detector = ood_data["detector"]
        ood_threshold = float(ood_data["threshold"])
        ood_centroid = ood_data.get("centroid")
        ood_distance_threshold = float(ood_data.get("distance_threshold", 0.0))
        ood_confidence_threshold = float(ood_data.get("confidence_threshold", 0.0)) if "confidence_threshold" in ood_data else None
    else:
        ood_detector = ood_data
        ood_threshold = float(getattr(ood_detector, "threshold_", getattr(ood_detector, "offset_", 0.0)))
        ood_centroid = None
        ood_distance_threshold = None
        ood_confidence_threshold = None
    ood_extractor = build_ood_feature_extractor(args.img_size)

    result = predict_single_image(
        args.image_path,
        classifier,
        ood_extractor,
        ood_detector,
        class_names,
        args.img_size,
        ood_threshold,
        ood_centroid,
        ood_distance_threshold,
        ood_confidence_threshold,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
