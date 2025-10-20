# Expression Predictor Comparator

A simple app that compares **three facial expression predictors** side-by-side with **label**, **confidence**, and **latency** per frame:

1) LBP + KNN
2) **HOG + Linear SVM (Calibrated)**
3) **mini-Xception (CNN) trained on FER-2013**

## Quickstart

```bash
# 1) Clone your repo, then inside it:
python -m venv .venv && source .venv/bin/activate # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt

# 2) Get models
#  Download a pretrained mini-Xception (≈66% FER2013 test acc)
bash mini_xception_weights.sh

# Train classical models quickly on FER-2013
python train_classical.py --fer_csv /path/to/fer2013.csv

# 3) Launch the app
python app.py
```

## Small Demo

https://github.com/Gelmi/RO11-TP4/blob/main/demo.mp4

## Performance

| Model              | Mean Latency   (ms) | Accuracy          |
|--------------------|---------------------|-------------------|
| LBP + KNN          | 16.61               | 0.28              |  
| HOG + Linear SVM   | 3.27                | 0.46              |
| mini-Xception      | 56.19               | 0.66 (pretrained) |

