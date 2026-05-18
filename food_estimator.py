import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda-12.5'
import json
import math
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow import keras
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

warnings.filterwarnings('ignore')

# ── PATH CONFIG ──────────────────────
FOOD_MODEL_PATH    = "/home/shivansh-bharti/Programming/Machine Learning/Food Image Recognition and Calorie Prediction/final_model/food_model_resnet.keras"
VESSEL_MODEL_PATH  = "/home/shivansh-bharti/Programming/Machine Learning/Food Image Recognition and Calorie Prediction/final_model/plate_vs_bowl_model.keras"
NUTRIENT_CSV_PATH  = "/home/shivansh-bharti/Programming/Machine Learning/Food Image Recognition and Calorie Prediction/indian food calories dataset.csv"
OUTPUT_DIR         = "results"

# ── MODEL & VESSEL CONFIG ───────────────────────────────────────────
FOOD_IMG_SIZE      = (224, 224)
VESSEL_IMG_SIZE    = (224, 224)
FOOD_CONF_THRESH   = 0.40
VESSEL_CONF_THRESH = 0.50

VESSEL_DIMS = {
    "plate": {"diameter_cm": 25.0, "radius_cm": 12.5, "area_cm2": math.pi * 12.5**2, "usable_frac": 0.75},
    "bowl":  {"diameter_cm": 15.0, "radius_cm": 7.5,  "area_cm2": math.pi * 7.5**2,  "usable_frac": 0.85}
}

EXPECTED_FILL = {"plate": 0.65, "bowl": 0.80}

FOOD_CLASSES = [
    'adhirasam', 'aloo_gobi', 'aloo_matar', 'aloo_methi', 'aloo_shimla_mirch',
    'aloo_tikki', 'anarsa', 'ariselu', 'bandar_laddu', 'basundi',
    'bhatura', 'bhindi_masala', 'biryani', 'boondi', 'butter_chicken',
    'chak_hao_kheer', 'cham_cham', 'chana_masala', 'chapati', 'chicken_razala',
    'chicken_tikka', 'chicken_tikka_masala', 'chikki', 'daal_baati_churma',
    'daal_puri', 'dal_makhani', 'dal_tadka', 'dum_aloo', 'gajar_ka_halwa',
    'gulab_jamun', 'jalebi', 'kadai_paneer', 'kadhi_pakoda', 'kofta',
    'lassi', 'litti_chokha', 'makki_di_roti_sarson_da_saag', 'malapua',
    'modak', 'mysore_pak', 'naan', 'palak_paneer', 'paneer_butter_masala',
    'phirni', 'poha', 'rabri', 'ras_malai', 'rasgulla', 'shrikhand',
    'sohan_papdi', 'unni_appam'
]

VESSEL_CLASSES   = ['food_in_plate', 'food_in_bowl']
VESSEL_CLASS_MAP = {'food_in_plate': 'plate', 'food_in_bowl': 'bowl'}

# ── SEGMENTATION CONFIG ─────────────────────────────────────────────
VESSEL_LABELS  = {47, 51}                        # COCO cup=47, bowl=51
EXCLUDE_LABELS = {0, 62, 63, 67, 70, 72, 73, 81, 84}  # bg, furniture, etc.


# ── PREPROCESSING ────────────────────────────────────────────────────

def preprocess_for_food(image_path):
    img = keras.utils.load_img(image_path, target_size=FOOD_IMG_SIZE)
    arr = keras.utils.img_to_array(img)
    arr = tf.keras.applications.resnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def preprocess_for_vessel(image_path):
    img = keras.utils.load_img(image_path, target_size=VESSEL_IMG_SIZE)
    arr = keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ── MASK R-CNN ───────────────────────────────────────────────────────

def run_maskrcnn(image_path, model, transforms, device):
    pil_img = Image.open(image_path).convert("RGB")
    img_hw  = (pil_img.height, pil_img.width)
    inp     = transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inp)[0]
    keep = outputs['scores'] >= 0.50
    return (
        outputs['boxes'][keep].cpu().numpy(),
        outputs['labels'][keep].cpu().numpy(),
        outputs['scores'][keep].cpu().numpy(),
        (outputs['masks'][keep, 0] >= 0.5).cpu().numpy(),
        img_hw
    )


def get_best_food_mask(labels, scores, masks, img_hw, vessel_type):
    H, W = img_hw
    vessel_mask = np.zeros((H, W), dtype=bool)
    for idx in range(len(labels)):
        if labels[idx] in VESSEL_LABELS and scores[idx] >= 0.50:
            vessel_mask = vessel_mask | masks[idx]
            break
    if not vessel_mask.any():
        cy, cx = H // 2, W // 2
        ry, rx = int(H * 0.42), int(W * 0.42)
        Y, X   = np.ogrid[:H, :W]
        vessel_mask = ((X - cx)**2 / rx**2 + (Y - cy)**2 / ry**2) <= 1.0

    best_mask, best_score = None, -1
    for idx in range(len(labels)):
        if int(labels[idx]) in EXCLUDE_LABELS or int(labels[idx]) in VESSEL_LABELS:
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

    if best_mask is None:
        best_mask = vessel_mask.copy()
    return best_mask.astype(bool), vessel_mask.astype(bool)


# ── NUTRIENT CALCULATION ─────────────────────────────────────────────

def calculate_nutrients(food_name, weight_g, nutrient_df):
    """
    Scale per-100g values to actual portion weight.
    Returns calories, protein, carbs, fat, fiber + calorie uncertainty range.
    """
    row   = nutrient_df.loc[food_name]
    scale = weight_g / 100.0

    # calorie_uncertainty column may not exist in every CSV version — default to 15%
    unc = float(row["calorie_uncertainty"]) if "calorie_uncertainty" in nutrient_df.columns else 0.15

    calories = round(float(row["calories_per_100g"]) * scale, 1)
    protein  = round(float(row["protein_g"])         * scale, 1)
    carbs    = round(float(row["carbs_g"])            * scale, 1)
    fat      = round(float(row["fat_g"])              * scale, 1)
    fiber    = round(float(row["fiber_g"])            * scale, 1)

    # Calories per gram of the food item (fixed property, not portion-dependent)
    cal_per_g = round(float(row["calories_per_100g"]) / 100.0, 3)

    return {
        "calories_kcal"   : calories,
        "calories_range"  : [round(calories * (1 - unc), 1), round(calories * (1 + unc), 1)],
        "cal_per_g"       : cal_per_g,
        "protein_g"       : protein,
        "carbohydrates_g" : carbs,
        "fat_g"           : fat,
        "fiber_g"         : fiber,
    }


# ── FULL PIPELINE ────────────────────────────────────────────────────

def estimate_calories(image_path, food_model, vessel_model, mrcnn, mrcnn_trans, device, nutrient_df):
    # Step 1 – Food classification
    food_input = preprocess_for_food(image_path)
    food_probs = food_model.predict(food_input, verbose=0)[0]
    food_idx   = np.argmax(food_probs)
    food_name  = FOOD_CLASSES[food_idx]
    food_conf  = float(food_probs[food_idx])

    # Step 2 – Vessel classification 
    vessel_input = preprocess_for_vessel(image_path)
    vessel_probs = vessel_model.predict(vessel_input, verbose=0)[0]
    vessel_type  = VESSEL_CLASS_MAP[VESSEL_CLASSES[np.argmax(vessel_probs)]]

    # Step 3 – Mask R-CNN area estimation
    boxes, labels, scores, masks, img_hw = run_maskrcnn(image_path, mrcnn, mrcnn_trans, device)
    food_mask, vessel_mask = get_best_food_mask(labels, scores, masks, img_hw, vessel_type)

    v_pix, f_pix = int(vessel_mask.sum()), int(food_mask.sum())
    area_frac    = max(0.05, min(f_pix / v_pix, 1.0)) if v_pix > 0 else 0.60

    # Step 4 – Weight estimation
    row        = nutrient_df.loc[food_name]
    fill_ratio = max(0.15, min(area_frac / EXPECTED_FILL.get(vessel_type, 0.65), 2.0))
    weight_g   = float(row["avg_serving_g"]) * fill_ratio

    # Step 5 – Nutrient calculation
    nutrients = calculate_nutrients(food_name, weight_g, nutrient_df)

    return {
        "food"         : food_name,
        "food_conf"    : round(food_conf, 4),
        "weight_g"     : round(weight_g, 1),
        "area_fraction": area_frac,
        "nutrients"    : nutrients,
        "food_mask"    : food_mask,
    }


# ── DISPLAY ───────────────────────────────────────────────────────────

def print_results(image_path, result):
    n         = result["nutrients"]
    food_name = result["food"].replace("_", " ").title()
    weight    = result["weight_g"]
    cal_lo, cal_hi = n["calories_range"]

    print(f"\n{'='*50}")
    print(f"  {os.path.basename(image_path)}")
    print(f"{'='*50}")
    print(f"  Food        : {food_name}  (conf: {result['food_conf']:.1%})")
    print(f"  Est. Weight : {weight} g")
    print(f"{'─'*50}")
    print(f"  {'Nutrient':<22} {'Amount':>10}  {'Per 100g':>10}")
    print(f"  {'─'*44}")

    rows = [
        ("Calories (kcal)",  n["calories_kcal"],   round(n["cal_per_g"] * 100, 1)),
        ("Protein (g)",      n["protein_g"],        round(n["protein_g"]        / weight * 100, 1) if weight else "-"),
        ("Carbohydrates (g)",n["carbohydrates_g"],  round(n["carbohydrates_g"]  / weight * 100, 1) if weight else "-"),
        ("Fat (g)",          n["fat_g"],            round(n["fat_g"]            / weight * 100, 1) if weight else "-"),
        ("Fiber (g)",        n["fiber_g"],          round(n["fiber_g"]          / weight * 100, 1) if weight else "-"),
    ]
    for label, amount, per100 in rows:
        print(f"  {label:<22} {amount:>10}  {per100:>10}")

    print(f"{'─'*50}")
    print(f"  Cal/g       : {n['cal_per_g']} kcal/g")
    print(f"  Calorie range: {cal_lo} – {cal_hi} kcal  (±{round((cal_hi/n['calories_kcal']-1)*100):.0f}%)")
    print(f"{'='*50}\n")


# ── MAIN ─────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading Models...")
    food_model   = keras.models.load_model(FOOD_MODEL_PATH)
    vessel_model = keras.models.load_model(VESSEL_MODEL_PATH)

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights    = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    mrcnn      = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
    mrcnn_trans = weights.transforms()

    nutrient_df = pd.read_csv(NUTRIENT_CSV_PATH)
    nutrient_df['food_name'] = (
        nutrient_df['food_name']
        .str.strip().str.lower()
        .str.replace(' ', '_').str.replace('-', '_')
    )
    nutrient_df.set_index('food_name', inplace=True)

    img_path = input("Enter path to image: ").strip()
    if not os.path.exists(img_path):
        print("File not found.")
        return

    result = estimate_calories(img_path, food_model, vessel_model, mrcnn, mrcnn_trans, device, nutrient_df)
    print_results(img_path, result)

    # Save JSON (excluding the mask array)
    out = {k: v for k, v in result.items() if k != "food_mask"}
    out_path = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(img_path))[0] + "_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Result saved → {out_path}")


if __name__ == "__main__":
    main()