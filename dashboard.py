# ─────────────────────────────────────────────────────────────
# dashboard.py – Dashboard EDA (CCT20)
# ─────────────────────────────────────────────────────────────
import streamlit as st
import json
from pathlib import Path

IMG_DIR = Path(r"C:/Users/sylva/OPC/dashboard/eda")

# ─────────── 1. utilitaire datasets → data.values ───────────
def inline_datasets(spec: dict) -> dict:
    if not isinstance(spec, dict):
        return spec
    datasets = spec.get("datasets", {})
    if "data" in spec and isinstance(spec["data"], dict) and "name" in spec["data"]:
        name = spec["data"]["name"]
        if name in datasets:
            spec["data"] = {"values": datasets[name]}
            spec.pop("datasets", None)
    # récursion
    for key in ("layer", "hconcat", "vconcat", "concat", "spec", "facet", "repeat"):
        if key in spec:
            content = spec[key]
            if isinstance(content, list):
                spec[key] = [inline_datasets(c) for c in content]
            else:
                spec[key] = inline_datasets(content)
    return spec

def show_chart(path: Path, titre: str, desc: str):
    if not path.exists():
        st.error(f"{path.name} manquant"); return
    spec = inline_datasets(json.loads(path.read_text()))
    st.subheader(titre)
    st.markdown(desc)
    st.vega_lite_chart(spec, use_container_width=True)
    st.divider()

# ─────────── 2. mise en page & accessibilité ────────────────
st.set_page_config("Dashboard EDA – CCT20", layout="wide")
st.title("📊 Dashboard – Analyse exploratoire du dataset CCT20")
st.markdown(
    "Graphiques interactifs pour illustrer la distribution spatiale, "
    "temporelle et catégorielle des images."
)

if "font_size" not in st.session_state:
    st.session_state.font_size = 16
if "hi_contrast" not in st.session_state:
    st.session_state.hi_contrast = False

with st.sidebar:
    st.header("Affichage")
    col1, col2 = st.columns(2)
    if col1.button("🔍 +"):  st.session_state.font_size += 2
    if col2.button("🔎 –"):  st.session_state.font_size = max(12, st.session_state.font_size - 2)
    st.session_state.hi_contrast = st.checkbox("Contraste élevé", st.session_state.hi_contrast)

st.markdown(
    f"""
    <style>
      html,body,[class*='css']{{font-size:{st.session_state.font_size}px;}}
      {'body{{filter:invert(1) hue-rotate(180deg);}}' if st.session_state.hi_contrast else ''}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────── 3. affichage des graphiques ────────────────────
show_chart(
    IMG_DIR / "01_images_par_categorie.json",
    "Images par catégorie",
    "Nombre d'images avec au moins une détection, par classe."
)

show_chart(
    IMG_DIR / "02_volume_par_camera.json",
    "Volume d’images par caméra",
    "Nombre de captures enregistrées par chaque caméra."
)

show_chart(
    IMG_DIR / "03_heatmap_camera_categorie.json",
    "Heat-map caméra × catégorie",
    "Nombre d’images pour chaque couple (caméra, classe)."
)

show_chart(
    IMG_DIR / "04_images_temps.json",
    "Répartition temporelle",
    "Variation saisonnière (mois) et rythme journalier (heures)."
)

show_chart(
    IMG_DIR / "05_couverture_classes.json",
    "Couverture cumulative des classes",
    "Nombre de classes nécessaires pour couvrir 80 % des images."
)

show_chart(
    IMG_DIR / "06_bboxes_par_categorie.json",
    "Bounding boxes par catégorie",
    "Nombre total de bboxes annotées pour chaque classe."
)

st.markdown("**Fin du tableau de bord.**")


# ───────────── 4. Résultats YOLO ────────────────

st.header("📸 Résultats des modèles YOLO")

YOLO_DIR = Path("C:/Users/sylva/OPC/dashboard/results_yolo")

images = [
    ("results_yolo_cls.png", "Résultats – YOLO Classify"),
    ("confusion_matrix_normalized_yolo_cls.png", "Matrice de confusion – YOLO Classify"),

    ("results_yolov8m.png", "Résultats – YOLOv8m 224px"),
    ("confusion_matrix_normalized_yolov8.png", "Matrice de confusion – YOLOv8m"),

    ("results_yolov11.png", "Résultats – YOLOv11"),
    ("confusion_matrix_normalized_yolov11.png", "Matrice de confusion – YOLOv11"),
]

for filename, title in images:
    img_path = YOLO_DIR / filename
    if img_path.exists():
        st.subheader(title)
        st.image(str(img_path), use_column_width=True)
        st.divider()
    else:
        st.warning(f"Image non trouvée : {filename}")