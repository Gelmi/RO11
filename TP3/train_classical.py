import os
import argparse
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from skimage.feature import local_binary_pattern, hog

EMO_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LBP_P, LBP_R, LBP_METHOD = 8, 1, "uniform"
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                  block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)


def parse_fer2013(csv_path: str):
    # FER2013 CSV columns: emotion,pixels,Usage
    import pandas as pd
    df = pd.read_csv(csv_path)
    X, y = [], []
    for _, row in df.iterrows():
        emo = int(row["emotion"])
        pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
        img = pixels.reshape(48, 48)
        X.append(img)
        y.append(emo)
    return np.array(X), np.array(y)


def lbp_hist(gray48: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray48, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    n_bins = int(LBP_P + 2)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def hog_feat(gray48: np.ndarray) -> np.ndarray:
    from skimage.transform import resize
    r = resize(gray48, (64, 64), anti_aliasing=True)
    f = hog(r, **HOG_PARAMS)
    return f.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fer_csv', required=True)
    ap.add_argument('--out_dir', default='./models')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X, y = parse_fer2013(args.fer_csv)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    Xtr_lbp = np.stack([lbp_hist(im) for im in Xtr])
    Xte_lbp = np.stack([lbp_hist(im) for im in Xte])

    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)
    knn.fit(Xtr_lbp, ytr)
    print("LBP+KNN test report:\n", classification_report(yte, knn.predict(Xte_lbp), target_names=EMO_LABELS))
    dump(knn, os.path.join(args.out_dir, 'lbp_knn.joblib'))

    Xtr_hog = np.stack([hog_feat(im) for im in Xtr])
    Xte_hog = np.stack([hog_feat(im) for im in Xte])

    svc = LinearSVC(dual=False)
    clf = CalibratedClassifierCV(estimator=svc, method='sigmoid', cv=3)
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', clf)
    ])
    print("Training HOG+LinearSVC (Calibrated) with 3-fold CV ...", flush=True)
    pipe.fit(Xtr_hog, ytr)
    print("Calibration + training done.", flush=True)
    print("HOG+SVM test report:\n", classification_report(yte, pipe.predict(Xte_hog), target_names=EMO_LABELS))
    dump(pipe, os.path.join(args.out_dir, 'hog_svm_calibrated.joblib'))


if __name__ == '__main__':
    main()
