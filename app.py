import os
import time
import cv2
import numpy as np
import gradio as gr
from typing import Dict, Tuple, Optional
from skimage.feature import local_binary_pattern, hog
from skimage.transform import resize as sk_resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
    SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dropout)
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras import layers
    from tensorflow.keras.utils import get_file
except Exception:
    tf = None

from joblib import dump, load

EMO_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(CASCADE_PATH)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LBP_KNN_PATH = os.path.join(MODELS_DIR, "lbp_knn.joblib")
HOG_SVM_PATH = os.path.join(MODELS_DIR, "hog_svm_calibrated.joblib")
XCEPTION_W_PATH = os.path.join(MODELS_DIR, "mini_xception_fer2013.h5")


LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
    feature_vector=True,
)

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[..., 0]
    return img

def detect_largest_face(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = _to_gray(bgr)
    faces = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    # Choose largest face (area)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)

def crop_face(img: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    h_img, w_img = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    return img[y0:y1, x0:x1]

def lbp_histogram(gray48: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray48, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    n_bins = int(LBP_P + 2)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)

def hog_features(gray64: np.ndarray) -> np.ndarray:
    feats = hog(gray64, visualize=False, **HOG_PARAMS)
    return feats.astype(np.float32)

def mini_xception(input_shape=(48, 48, 1), num_classes=7, l2_regularization=0.01) -> Model:
    """mini-XCEPTION architecture matching popular FER-2013 weights (oarriaga).
    This version uses residual additions, depthwise separable blocks, and a final
    Conv2D(num_classes) + GAP + softmax head.
    """
    if tf is None:
        raise RuntimeError("TensorFlow not available. Install tensorflow or tensorflow[and-cuda] first.")

    regularization = l2(l2_regularization)

    img_input = Input(shape=input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='predictions')(x)
    return Model(img_input, out, name='mini_XCEPTION')


class MiniXceptionWrapper:
    def __init__(self, weights_path: str):
        self.model = mini_xception()
        if os.path.isfile(weights_path):
            self.model.load_weights(weights_path)
            self.ready = True
        else:
            self.ready = False

    def predict(self, face_gray48: np.ndarray) -> Tuple[int, float, np.ndarray]:
        if not self.ready:
            raise RuntimeError("mini-Xception weights not found. See README to download to ./models/.")
        x = face_gray48.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=(0, -1))  # (1,48,48,1)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return idx, conf, probs

LBP_KNN = load(LBP_KNN_PATH) if os.path.isfile(LBP_KNN_PATH) else None
HOG_SVM = load(HOG_SVM_PATH) if os.path.isfile(HOG_SVM_PATH) else None
XCEPT = MiniXceptionWrapper(XCEPTION_W_PATH) if tf is not None else None

def predict_lbp_knn(face_gray48: np.ndarray) -> Tuple[str, float, float]:
    t0 = time.perf_counter()
    if LBP_KNN is None:
        return ("(not loaded)", 0.0, (time.perf_counter() - t0) * 1000)
    feat = lbp_histogram(face_gray48)
    probs = LBP_KNN.predict_proba([feat])[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    ms = (time.perf_counter() - t0) * 1000
    return EMO_LABELS[idx], conf, ms


def predict_hog_svm(face_gray64: np.ndarray) -> Tuple[str, float, float]:
    t0 = time.perf_counter()
    if HOG_SVM is None:
        return ("(not loaded)", 0.0, (time.perf_counter() - t0) * 1000)
    feat = hog_features(face_gray64)
    probs = HOG_SVM.predict_proba([feat])[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    ms = (time.perf_counter() - t0) * 1000
    return EMO_LABELS[idx], conf, ms


def predict_xception(face_gray48: np.ndarray) -> Tuple[str, float, float]:
    t0 = time.perf_counter()
    if (XCEPT is None) or (not XCEPT.ready):
        return ("(not loaded)", 0.0, (time.perf_counter() - t0) * 1000)
    idx, conf, _ = XCEPT.predict(face_gray48)
    ms = (time.perf_counter() - t0) * 1000
    return EMO_LABELS[idx], conf, ms


def process_frame(bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Returns annotated frame and a dict of metrics per model."""
    print("processando frame")
    if bgr is None:
        return None, {}
    img = bgr.copy()
    roi = detect_largest_face(img)
    if roi is None:
        return img, {}

    x, y, w, h = roi
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    face = crop_face(bgr, roi)
    gray = _to_gray(face)
    face48 = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    face64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

    lbp_label, lbp_conf, lbp_ms = predict_lbp_knn(face48)
    hog_label, hog_conf, hog_ms = predict_hog_svm(face64)
    xcp_label, xcp_conf, xcp_ms = predict_xception(face48)

    results = {
        "LBP + KNN": {"label": lbp_label, "confidence": lbp_conf, "ms": lbp_ms},
        "HOG + Linear SVM": {"label": hog_label, "confidence": hog_conf, "ms": hog_ms},
        "mini-Xception": {"label": xcp_label, "confidence": xcp_conf, "ms": xcp_ms},
    }
    return img[..., ::-1], results


CSS = """
.card {border:1px solid #e5e7eb; border-radius:0.75rem; padding:0.75rem;}
.card h4 {margin: 0 0 0.25rem 0; font-size: 1rem;}
.grid {display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:0.75rem;}
.kpi {font-size: 0.9rem; opacity: 0.9}
"""


def format_cards(results: Dict[str, Dict[str, float]]) -> str:
    if not results:
        return "<em>No face detected.</em>"
    cards = []
    for name, d in results.items():
        label = d["label"]
        conf = d["confidence"]
        ms = d["ms"]
        conf_pct = f"{conf*100:.1f}%" if conf else "–"
        ms_txt = f"{ms:.1f} ms"
        cards.append(
            f"<div class='card'><h4>{name}</h4>"
            f"<div class='kpi'><strong>Label:</strong> {label}</div>"
            f"<div class='kpi'><strong>Confidence:</strong> {conf_pct}</div>"
            f"<div class='kpi'><strong>Latency:</strong> {ms_txt}</div>"
            f"</div>"
        )
    return f"<div class='grid'>{''.join(cards)}</div>"


def run_on_frame(img: np.ndarray):
    if img is None:
        return None, ""
    bgr = img[:, :, ::-1]
    annotated, results = process_frame(bgr)
    return annotated, format_cards(results)


def benchmark(img: np.ndarray, repeats: int = 30):
    if img is None:
        return []

    bgr = img[:, :, ::-1]
    roi = detect_largest_face(bgr)
    if roi is None:
        return []

    face = crop_face(bgr, roi)
    gray = _to_gray(face)
    face48 = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    face64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

    times = {"LBP + KNN": [], "HOG + Linear SVM": [], "mini-Xception": []}
    for _ in range(repeats):
        times["LBP + KNN"].append(predict_lbp_knn(face48)[2])
        times["HOG + Linear SVM"].append(predict_hog_svm(face64)[2])
        times["mini-Xception"].append(predict_xception(face48)[2])

    rows = []
    for k, vals in times.items():
        vals = [v for v in vals if v > 0]
        if not vals:
            continue
        rows.append([k, float(np.mean(vals))])

    return rows

def _capture_frame(im):
    return im

def build_demo():
    with gr.Blocks(css=CSS, title="Expression Predictor Comparator") as demo:
        last_frame = gr.State(value=None)
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
                with gr.Row():
                    bench_btn = gr.Button("Benchmark (30 runs)")
            with gr.Column(scale=1):
                annotated = gr.Image(label="Detected face (box)")
                cards = gr.HTML()
                bench_table = gr.Dataframe(headers=["model", "mean_ms"], label="Mean latency (ms)")

        img.stream(fn=run_on_frame, inputs=img, outputs=[annotated, cards])
        img.stream(fn=_capture_frame, inputs=img, outputs=last_frame)
        bench_btn.click(fn=benchmark, inputs=last_frame, outputs=bench_table)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(max_size=20).launch()
