"""
inference.py  —  Indian Food Calorie & Nutrient Estimator
==========================================================
Standalone pipeline: Food Classifier + Vessel Classifier
→ Mask R-CNN Area Estimation → Nutrient Lookup → JSON Output

Usage (CLI):
    python inference.py --image path/to/food.jpg

Usage (as module):
    from inference import load_models, estimate_calories
    food_model, vessel_model, nutrient_df = load_models()
    result = estimate_calories("food.jpg", food_model, vessel_model, nutrient_df)
"""

import os
import json
import math
import warnings
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# ── FIX 1: Set env vars BEFORE importing TF/PyTorch ──────────────────
os.environ["TF_GPU_ALLOCATOR"]              = "cuda_malloc_async"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
# ─────────────────────────────────────────────────────────────────────

import tensorflow as tf

# ── FIX 2: Enable memory growth + hard 2 GB VRAM cap for TensorFlow ──
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Hard cap: give TF only 2 GB of your 4 GB VRAM
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
            )
        print("TensorFlow GPU memory growth enabled (cap: 2 GB).")
    except RuntimeError as e:
        print(e)
# ─────────────────────────────────────────────────────────────────────

from tensorflow import keras
import torch
import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION  (edit paths here or pass via env vars)
# ══════════════════════════════════════════════════════════════════════

FOOD_MODEL_PATH   = os.getenv("FOOD_MODEL_PATH",   "models/food_model_resnet.keras")
VESSEL_MODEL_PATH = os.getenv("VESSEL_MODEL_PATH", "models/plate_vs_bowl_model.keras")
NUTRIENT_CSV_PATH = os.getenv("NUTRIENT_CSV_PATH", "data/indian_food_calories_dataset.csv")

FOOD_IMG_SIZE      = (224, 224)
VESSEL_IMG_SIZE    = (224, 224)
FOOD_CONF_THRESH   = 0.40
VESSEL_CONF_THRESH = 0.50

VESSEL_DIMS = {
    "plate": {
        "diameter_cm": 25.0,
        "radius_cm":   12.5,
        "area_cm2":    math.pi * 12.5 ** 2,   # ≈ 490.9 cm²
        "usable_frac": 0.75,
    },
    "bowl": {
        "diameter_cm": 15.0,
        "radius_cm":   7.5,
        "area_cm2":    math.pi * 7.5 ** 2,    # ≈ 176.7 cm²
        "usable_frac": 0.85,
    },
}

FOOD_CLASSES = [
    "adhirasam", "aloo_gobi", "aloo_matar", "aloo_methi", "aloo_shimla_mirch",
    "aloo_tikki", "anarsa", "ariselu", "bandar_laddu", "basundi",
    "bhatura", "bhindi_masala", "biryani", "boondi", "butter_chicken",
    "chak_hao_kheer", "cham_cham", "chana_masala", "chapati", "chicken_razala",
    "chicken_tikka", "chicken_tikka_masala", "chikki", "daal_baati_churma",
    "daal_puri", "dal_makhani", "dal_tadka", "dum_aloo", "gajar_ka_halwa",
    "gulab_jamun", "jalebi", "kadai_paneer", "kadhi_pakoda", "kofta",
    "lassi", "litti_chokha", "makki_di_roti_sarson_da_saag", "malapua",
    "modak", "mysore_pak", "naan", "palak_paneer", "paneer_butter_masala",
    "phirni", "poha", "rabri", "ras_malai", "rasgulla", "shrikhand",
    "sohan_papdi", "unni_appam",
]

VESSEL_CLASSES   = ["food_in_plate", "food_in_bowl"]
VESSEL_CLASS_MAP = {"food_in_plate": "plate", "food_in_bowl": "bowl"}

EXPECTED_FILL = {"plate": 0.65, "bowl": 0.80}

# COCO label list (91 entries)
COCO_LABELS = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]
EXCLUDE_LABELS = {0, 62, 63, 67, 70, 72, 73, 81, 84}
VESSEL_LABELS  = {47, 51}   # cup=47, bowl=51


# ══════════════════════════════════════════════════════════════════════
# 2.  MODEL LOADING  (call once at startup)
# ══════════════════════════════════════════════════════════════════════

_mrcnn_model      = None
_mrcnn_device     = None
_mrcnn_transforms = None


def _load_maskrcnn():
    """Load Mask R-CNN once and cache globally (lazy — only on first inference)."""
    global _mrcnn_model, _mrcnn_device, _mrcnn_transforms
    if _mrcnn_model is not None:
        return
    print("Loading pretrained Mask R-CNN (COCO weights)…")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    device  = torch.device("cpu")  # Forced to CPU to keep VRAM free for Keras
    model   = model.to(device)
    _mrcnn_model      = model
    _mrcnn_device     = device
    _mrcnn_transforms = weights.transforms()
    print(f"  Mask R-CNN ready on: {device}")


def load_models():
    """
    Load food classifier, vessel classifier, and nutrient CSV.
    Mask R-CNN is NOT loaded here — it lazy-loads on the first inference call.

    Returns
    -------
    food_model   : Keras model
    vessel_model : Keras model
    nutrient_df  : pd.DataFrame  (indexed by food_name)
    """
    # ── FIX 3: Force both Keras models onto CPU ───────────────────────
    # Single-image inference at 224×224 is fast enough on CPU (~0.3–0.5 s).
    # This frees your entire 4 GB VRAM budget for Mask R-CNN's CPU tensors
    # and prevents the OOM crash on 8 GB RAM machines.
    print("Loading food classifier (CPU)…")
    with tf.device('/CPU:0'):
        food_model = keras.models.load_model(FOOD_MODEL_PATH)

    print("Loading vessel classifier (CPU)…")
    with tf.device('/CPU:0'):
        vessel_model = keras.models.load_model(VESSEL_MODEL_PATH)
    # ─────────────────────────────────────────────────────────────────

    print("Loading nutrient table…")
    nutrient_df = pd.read_csv(NUTRIENT_CSV_PATH)
    nutrient_df["food_name"] = (
        nutrient_df["food_name"]
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    nutrient_df.set_index("food_name", inplace=True)
    print(f"  Nutrient table: {len(nutrient_df)} foods")

    # ── FIX 4: Mask R-CNN is NOT pre-loaded at startup ────────────────
    # It will lazy-load on the first call to estimate_food_area_fraction().
    # Removing it here saves ~1.5–2 GB RAM during server startup and
    # prevents VS Code from killing the process before the server is ready.
    # _load_maskrcnn()   ← intentionally removed
    # ─────────────────────────────────────────────────────────────────

    return food_model, vessel_model, nutrient_df


# ══════════════════════════════════════════════════════════════════════
# 3.  PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════

def preprocess_for_food(image_path: str) -> np.ndarray:
    img = keras.utils.load_img(image_path, target_size=FOOD_IMG_SIZE)
    arr = keras.utils.img_to_array(img)
    arr = tf.keras.applications.resnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def preprocess_for_vessel(image_path: str) -> np.ndarray:
    img = keras.utils.load_img(image_path, target_size=VESSEL_IMG_SIZE)
    arr = keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ══════════════════════════════════════════════════════════════════════
# 4.  MASK R-CNN HELPERS
# ══════════════════════════════════════════════════════════════════════

def run_maskrcnn(image_path: str, score_thresh: float = 0.50):
    _load_maskrcnn()
    pil_img = Image.open(image_path).convert("RGB")
    img_hw  = (pil_img.height, pil_img.width)
    inp     = _mrcnn_transforms(pil_img).unsqueeze(0).to(_mrcnn_device)

    with torch.no_grad():
        outputs = _mrcnn_model(inp)[0]

    keep   = outputs["scores"] >= score_thresh
    boxes  = outputs["boxes"][keep].cpu().numpy()
    labels = outputs["labels"][keep].cpu().numpy()
    scores = outputs["scores"][keep].cpu().numpy()
    masks  = (outputs["masks"][keep, 0] >= 0.5).cpu().numpy()

    return boxes, labels, scores, masks, img_hw


def get_best_food_mask(boxes, labels, scores, masks, img_hw, vessel_type: str):
    H, W = img_hw

    # Find vessel mask (bowl/cup COCO class)
    vessel_mask = np.zeros((H, W), dtype=bool)
    for idx in range(len(labels)):
        if labels[idx] in VESSEL_LABELS and scores[idx] >= 0.50:
            vessel_mask = vessel_mask | masks[idx]
            break

    if not vessel_mask.any():
        cy, cx = H // 2, W // 2
        ry, rx = int(H * 0.42), int(W * 0.42)
        Y, X   = np.ogrid[:H, :W]
        vessel_mask = ((X - cx) ** 2 / rx ** 2 + (Y - cy) ** 2 / ry ** 2) <= 1.0

    best_mask  = None
    best_score = -1
    best_label = "unknown"

    for idx in range(len(labels)):
        lbl = int(labels[idx])
        if lbl in EXCLUDE_LABELS or lbl in VESSEL_LABELS:
            continue
        m    = masks[idx]
        area = int(m.sum())
        if area < 500:
            continue
        overlap = int((m & vessel_mask).sum())
        if overlap < area * 0.2:
            continue
        combined = float(scores[idx]) * area
        if combined > best_score:
            best_score = combined
            best_mask  = m
            best_label = COCO_LABELS[lbl] if lbl < len(COCO_LABELS) else str(lbl)

    if best_mask is None:
        best_mask  = vessel_mask.copy()
        best_label = "region (fallback)"

    return (
        best_mask.astype(bool),
        vessel_mask.astype(bool),
        best_label,
        float(scores[0]) if len(scores) else 0.0,
    )


def estimate_food_area_fraction(
    image_path: str, vessel_type: str, score_thresh: float = 0.50
):
    """
    Returns
    -------
    area_fraction : float  [0.05, 1.0]
    overlay_bgr   : np.ndarray BGR image with mask, or None
    coco_label    : str
    """
    boxes, labels, scores, masks, img_hw = run_maskrcnn(image_path, score_thresh)
    H, W = img_hw

    food_mask, vessel_mask, coco_label, _ = get_best_food_mask(
        boxes, labels, scores, masks, img_hw, vessel_type
    )

    vessel_pixels = int(vessel_mask.sum())
    food_pixels   = int(food_mask.sum())

    if vessel_pixels == 0:
        return 0.60, None, coco_label

    area_fraction = food_pixels / vessel_pixels
    area_fraction = max(0.05, min(area_fraction, 1.0))

    pil_img = Image.open(image_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (W, H))

    overlay = img_bgr.copy()
    overlay[food_mask] = (
        overlay[food_mask] * 0.45 + np.array([0, 0, 220]) * 0.55
    ).clip(0, 255).astype(np.uint8)
    result_img = cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0)

    return area_fraction, result_img, coco_label


# ══════════════════════════════════════════════════════════════════════
# 5.  VOLUME / WEIGHT / NUTRIENT CALCULATION
# ══════════════════════════════════════════════════════════════════════

def estimate_volume_and_weight(
    vessel_type: str,
    food_name: str,
    area_fraction: float,
    nutrient_df: pd.DataFrame,
) -> dict:
    row          = nutrient_df.loc[food_name]
    avg_serving  = float(row["avg_serving_g"])
    density      = float(row["density_g_per_cm3"])
    food_height  = float(row["avg_height_cm"])

    expected_fill = EXPECTED_FILL.get(vessel_type, 0.65)
    fill_ratio    = area_fraction / expected_fill
    fill_ratio    = max(0.15, min(fill_ratio, 2.0))

    weight_g   = avg_serving * fill_ratio
    volume_cm3 = weight_g / density if density > 0 else 0.0

    return {
        "avg_serving_g":  avg_serving,
        "fill_ratio":     round(fill_ratio, 3),
        "weight_g":       round(weight_g, 1),
        "volume_cm3":     round(volume_cm3, 2),
        "food_height_cm": food_height,
    }


def calculate_nutrients(
    food_name: str, weight_g: float, nutrient_df: pd.DataFrame
) -> dict:
    row   = nutrient_df.loc[food_name]
    scale = weight_g / 100.0
    unc   = float(row["calorie_uncertainty"])

    calories = round(float(row["calories_per_100g"]) * scale, 1)
    protein  = round(float(row["protein_g"])         * scale, 1)
    carbs    = round(float(row["carbs_g"])            * scale, 1)
    fat      = round(float(row["fat_g"])              * scale, 1)
    fiber    = round(float(row["fiber_g"])            * scale, 1)

    return {
        "calories_kcal":    calories,
        "calories_range":   [round(calories * (1 - unc), 1), round(calories * (1 + unc), 1)],
        "protein_g":        protein,
        "carbohydrates_g":  carbs,
        "fat_g":            fat,
        "fiber_g":          fiber,
        "food_type":        str(row["food_type"]),
        "category":         str(row["category"]),
    }


# ══════════════════════════════════════════════════════════════════════
# 6.  FULL INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════

def estimate_calories(
    image_path: str,
    food_model,
    vessel_model,
    nutrient_df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Full pipeline: image → food class + vessel type → area → weight → nutrients → JSON.

    Returns dict with keys: food, vessel, portion, nutrients, area_fraction, status
    """
    result = {"image": os.path.basename(image_path), "status": "ok"}

    # ── Step 1: Food classification ───────────────────────────────────
    food_probs = food_model.predict(preprocess_for_food(image_path), verbose=0)[0]
    food_idx   = int(np.argmax(food_probs))
    food_conf  = float(food_probs[food_idx])
    food_name  = FOOD_CLASSES[food_idx]
    if verbose:
        print(f"[Food]   {food_name:<35}  conf={food_conf:.3f}")

    # ── Step 2: Vessel classification ─────────────────────────────────
    vessel_probs = vessel_model.predict(preprocess_for_vessel(image_path), verbose=0)[0]
    vessel_idx   = int(np.argmax(vessel_probs))
    vessel_conf  = float(vessel_probs[vessel_idx])
    vessel_type  = VESSEL_CLASS_MAP[VESSEL_CLASSES[vessel_idx]]
    if vessel_conf < VESSEL_CONF_THRESH:
        vessel_type = "plate"
    if verbose:
        print(f"[Vessel] {vessel_type:<35}  conf={vessel_conf:.3f}")

    # ── Step 3: Area estimation ───────────────────────────────────────
    try:
        area_fraction, heatmap, coco_label = estimate_food_area_fraction(
            image_path, vessel_type
        )
    except Exception as e:
        if verbose:
            print(f"[Area]   Mask R-CNN failed ({e}) → using fallback 0.60")
        area_fraction, heatmap, coco_label = 0.60, None, "fallback"
    if verbose:
        print(f"[Area]   '{coco_label}'  →  food occupies {area_fraction*100:.1f}% of vessel")

    # ── Step 4: Weight ────────────────────────────────────────────────
    if food_name not in nutrient_df.index:
        result["status"] = "food_not_in_nutrient_table"
        return result

    portion = estimate_volume_and_weight(vessel_type, food_name, area_fraction, nutrient_df)
    if verbose:
        print(f"[Weight] {portion['weight_g']} g  (fill_ratio={portion['fill_ratio']})")

    # ── Step 5: Nutrients ─────────────────────────────────────────────
    nutrients = calculate_nutrients(food_name, portion["weight_g"], nutrient_df)
    if verbose:
        print(f"[Cals]   {nutrients['calories_kcal']} kcal  "
              f"(range {nutrients['calories_range'][0]}–{nutrients['calories_range'][1]})")

    result.update({
        "food": {
            "name":       food_name,
            "category":   nutrients.pop("category"),
            "food_type":  nutrients.pop("food_type"),
            "confidence": round(food_conf, 4),
            "top3": [
                {"name": FOOD_CLASSES[j], "confidence": round(float(food_probs[j]), 4)}
                for j in np.argsort(food_probs)[::-1][:3]
            ],
        },
        "vessel": {
            "type":        vessel_type,
            "confidence":  round(vessel_conf, 4),
            "diameter_cm": VESSEL_DIMS[vessel_type]["diameter_cm"],
        },
        "portion":        portion,
        "nutrients":      nutrients,
        "area_fraction":  round(area_fraction, 4),
        "_heatmap":       heatmap,     # internal; stripped before JSON export
    })
    return result


# ══════════════════════════════════════════════════════════════════════
# 7.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════

def visualise_result(image_path: str, result: dict, save_dir: str = ".") -> str | None:
    """
    Save a 3-panel visualisation PNG to save_dir.
    Returns the output path, or None on failure.
    """
    if result.get("status") != "ok":
        return None
    required = ["food", "vessel", "portion", "nutrients", "area_fraction"]
    if not all(k in result for k in required):
        return None

    img_orig   = np.array(Image.open(image_path).resize(FOOD_IMG_SIZE))
    heatmap    = result.get("_heatmap")
    food_label = result["food"]["name"].replace("_", " ").title()
    vessel     = result["vessel"]["type"].title()
    n          = result["nutrients"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")

    # Panel 1 — original
    axes[0].imshow(img_orig)
    axes[0].set_title("Input image", fontsize=12, color="white", pad=8)
    axes[0].axis("off")

    # Panel 2 — mask overlay
    if heatmap is not None:
        masked_rgb = cv2.cvtColor(
            cv2.resize(heatmap, FOOD_IMG_SIZE), cv2.COLOR_BGR2RGB
        )
        axes[1].imshow(masked_rgb)
    else:
        axes[1].imshow(img_orig)
    axes[1].set_title(
        f"{food_label} on {vessel}  |  "
        f"food area: {result['area_fraction']*100:.1f}%  |  "
        f"est. weight: {result['portion']['weight_g']} g",
        fontsize=10, color="white", pad=8,
    )
    axes[1].axis("off")

    # Panel 3 — nutrient bar chart
    labels = ["Calories\n(kcal)", "Protein\n(g)", "Carbs\n(g)", "Fat\n(g)", "Fiber\n(g)"]
    values = [n["calories_kcal"], n["protein_g"], n["carbohydrates_g"], n["fat_g"], n["fiber_g"]]
    colors = ["#E74C3C", "#2980B9", "#F39C12", "#8E44AD", "#27AE60"]
    bars   = axes[2].bar(labels, values, color=colors, width=0.55, edgecolor="#222")
    for bar, val in zip(bars, values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.015,
            f"{val:.1f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="white",
        )
    cal_low, cal_high = n["calories_range"]
    axes[2].fill_between([-0.5, 0.5], cal_low, cal_high, color="#E74C3C", alpha=0.18,
                         label=f"Cal range\n{cal_low:.0f}–{cal_high:.0f} kcal")
    axes[2].axhline(cal_low,  color="#E74C3C", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[2].axhline(cal_high, color="#E74C3C", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[2].legend(fontsize=8, loc="upper right",
                   facecolor="#222", labelcolor="white", edgecolor="#444")
    axes[2].set_title(
        f"Nutrients  |  portion: {result['portion']['weight_g']} g  "
        f"(×{result['portion']['fill_ratio']:.2f} serving)",
        fontsize=10, color="white", pad=8,
    )
    axes[2].tick_params(colors="white")
    axes[2].spines[["top", "right"]].set_visible(False)
    for s in ["left", "bottom"]:
        axes[2].spines[s].set_color("#444")
    axes[2].set_facecolor("#1a1a1a")

    plt.tight_layout()
    stem     = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(save_dir, f"result_{stem}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def clean_for_json(result: dict) -> dict:
    """Strip internal numpy fields before JSON serialisation."""
    return {k: v for k, v in result.items() if k != "_heatmap"}


# ══════════════════════════════════════════════════════════════════════
# 8.  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Indian Food Calorie Estimator")
    parser.add_argument("--image",   required=True,       help="Path to input image")
    parser.add_argument("--save-dir", default="outputs",  help="Directory for output PNG")
    parser.add_argument("--no-viz",  action="store_true", help="Skip visualisation")
    parser.add_argument("--json-out", default=None,       help="Optional path to save JSON result")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"ERROR: image not found → {args.image}")
        return

    food_model, vessel_model, nutrient_df = load_models()

    print(f"\nProcessing: {args.image}\n" + "─" * 50)
    result = estimate_calories(args.image, food_model, vessel_model, nutrient_df)

    clean = clean_for_json(result)
    print("\n── Result JSON ──────────────────────────────────")
    print(json.dumps(clean, indent=2))

    if not args.no_viz:
        viz_path = visualise_result(args.image, result, save_dir=args.save_dir)
        if viz_path:
            print(f"\nVisualisation saved → {viz_path}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(clean, f, indent=2, default=str)
        print(f"JSON saved → {args.json_out}")


if __name__ == "__main__":
    main()