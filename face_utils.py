import os
import time
import cv2
import numpy as np

# Base = pasta onde está este arquivo (raiz do projeto)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminhos (sempre apontam para a RAIZ do projeto)
FACES_DIR_ABS = os.path.join(BASE_DIR, "faces")         # absoluto para salvar/acessar
FACES_DIR_REL = "faces"                                  # relativo para gravar no DB
MODEL_PATH = os.path.join(BASE_DIR, "lbph_model.yml")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

# Haar Cascade (vem com OpenCV)
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Parâmetros LBPH
LBPH_RADIUS = 2            # levemente maior (mais textura)
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
LBPH_THRESHOLD = 70.0      # fallback global (se não usar por nível)

# OPCIONAL: thresholds por nível (se quiser endurecer no app.py)
# use: from face_utils import PER_LEVEL_THR
PER_LEVEL_THR = {1: 62.0, 2: 56.0, 3: 52.0}

def ensure_dirs():
    os.makedirs(FACES_DIR_ABS, exist_ok=True)

ensure_dirs()

def _to_abs(path: str) -> str:
    """Se 'path' for relativo, converte para absoluto a partir da raiz (BASE_DIR)."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)

def _to_rel(path_abs: str) -> str:
    """Converte um caminho absoluto dentro do projeto para relativo (para gravar no DB)."""
    try:
        rel = os.path.relpath(path_abs, BASE_DIR)
        return rel.replace("\\", "/")
    except Exception:
        return path_abs

# --------- Pré-processamentos para robustez (iluminação/contraste) ----------
def _clahe(gray: np.ndarray) -> np.ndarray:
    # Equalização adaptativa melhora contraste em baixa luz / alto brilho
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def _norm_0_255(gray: np.ndarray) -> np.ndarray:
    # Normaliza para [0,255] preservando contraste
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return g.astype("uint8")

# --------------------------- Detecção de rosto ------------------------------
def detect_face(img_bgr):
    """
    Retorna (ROI_200x200_gray, bbox) do maior rosto detectado.
    Aplica CLAHE para robustez. bbox no formato (x,y,w,h) em ints.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None, None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe(gray)  # ajuda a detecção com variação de luz

    faces = CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    if len(faces) == 0:
        return None, None

    # pega o maior rosto
    x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (200, 200))

    # normalizações adicionais no patch de rosto
    roi = _clahe(roi)
    roi = _norm_0_255(roi)

    return roi, (int(x), int(y), int(w), int(h))

def get_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y
    )
    return recognizer

# ------------------------------- Treino ------------------------------------
def train_model():
    """
    Treina o LBPH a partir das imagens cadastradas no DB.
    Aceita 'image_path' relativo (faces/…) ou absoluto.
    Salva labels.txt (label -> name).
    """
    from db import get_users  # import tardio para evitar ciclos
    users = get_users()
    if not users:
        return False

    images, labels = [], []
    label_map = {}   # name -> label
    next_label = 0

    for u in users:
        path = u.get("image_path")
        if not path:
            continue

        # aceita relativo (faces/…) e absoluto
        path_abs = _to_abs(path)
        if not os.path.exists(path_abs):
            # fallback para barras invertidas etc.
            alt = _to_abs(path.replace("\\", "/"))
            if os.path.exists(alt):
                path_abs = alt
            else:
                continue

        img = cv2.imread(path_abs, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # se estiver fora do padrão, normaliza
        if img.shape != (200, 200):
            img = cv2.resize(img, (200, 200))
        img = _clahe(img)
        img = _norm_0_255(img)

        # normaliza label por nome
        name = (u.get("name") or "").strip() or "user"
        if name not in label_map:
            label_map[name] = next_label
            next_label += 1

        images.append(img)
        labels.append(label_map[name])

    if not images:
        return False

    recognizer = get_recognizer()
    recognizer.train(images, np.array(labels))
    recognizer.write(MODEL_PATH)

    # grava o mapa label -> name
    inv = {lab: nm for nm, lab in label_map.items()}
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for lab in sorted(inv.keys()):
            f.write(f"{lab}\t{inv[lab]}\n")
    return True

def load_label_map():
    lm = {}
    if not os.path.exists(LABELS_PATH):
        return lm
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lab, name = line.split("\t", 1)
            lm[int(lab)] = name
    return lm

# ------------------------------ Predição ------------------------------------
def predict_face(img_bgr):
    """
    Prediz (label, confidence, bbox) para o maior rosto detectado.
    Se não houver modelo, tenta treinar. Se nada der, retorna (None, None, None).
    """
    roi, bbox = detect_face(img_bgr)
    if roi is None:
        return None, None, None

    recognizer = get_recognizer()
    if not os.path.exists(MODEL_PATH):
        ok = train_model()
        if not ok:
            return None, None, bbox

    recognizer.read(MODEL_PATH)
    label, confidence = recognizer.predict(roi)
    return label, float(confidence), bbox

# ------------------------------ Cadastro ------------------------------------
def save_face_image(name: str, img_bgr, level: int):
    """
    Salva o recorte do rosto em faces/ (200x200) e retorna CAMINHO RELATIVO (faces/…png)
    que é o que vai para o DB. O arquivo fisicamente fica em BASE_DIR/faces/…png
    """
    ensure_dirs()
    roi, _ = detect_face(img_bgr)
    if roi is None:
        return None

    safe = "".join(c for c in name if c.isalnum() or c in ("_", "-")).strip() or "user"
    ts = int(time.time() * 1000)
    filename = f"{safe}_L{int(level)}_{ts}.png"

    abs_path = os.path.join(FACES_DIR_ABS, filename)
    ok = cv2.imwrite(abs_path, roi)
    if not ok:
        return None

    # retornamos relativo (faces/arquivo.png) para guardar no DB
    rel_path = os.path.join(FACES_DIR_REL, filename).replace("\\", "/")
    return rel_path
