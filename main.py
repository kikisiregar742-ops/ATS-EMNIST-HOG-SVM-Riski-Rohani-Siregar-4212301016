import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog
from skimage import exposure
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# =========================================================
# 1. INITIALIZATION & CONFIGURATION
# =========================================================
def initialize_project():
    config = {
        "csv_path": r"D:\Kuliah\Semester 5\Machine Vision\emnist-letters-train.csv",
        "n_samples_per_class": 500,
        "n_classes": 26,
        "result_dir": "result",
        "svm_params": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        "seed": 42
    }
    os.makedirs(config["result_dir"], exist_ok=True)
    return config

# =========================================================
# 2. DATA PREPARATION
# =========================================================
def load_and_sample_data(filepath, n_samples_per_class, n_classes, seed=42):
    print("Loading EMNIST dataset...")
    df = pd.read_csv(filepath)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values

    np.random.seed(seed)
    X_sampled, y_sampled = [], []

    for label in range(1, n_classes + 1):
        class_idx = np.where(labels == label)[0]
        if len(class_idx) >= n_samples_per_class:
            selected = np.random.choice(class_idx, n_samples_per_class, replace=False)
        else:
            selected = class_idx
        X_sampled.append(pixels[selected])
        y_sampled.append(labels[selected])
        print(f"Class {label} ({chr(64 + label)}): {len(selected)} samples")

    X = np.vstack(X_sampled).reshape(-1, 28, 28)
    y = np.hstack(y_sampled)
    print(f"\nTotal samples: {len(X)}")
    return X, y

# =========================================================
# 3. FEATURE EXTRACTION (HOG)
# =========================================================
def extract_hog_features(images):
    print("\nExtracting HOG Features...")
    features = []
    for img in tqdm(images, desc="HOG", ncols=80):
        img_norm = img / 255.0
        fd = hog(img_norm, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False, channel_axis=None)
        features.append(fd)
    return np.array(features)

def visualize_hog_sample(image, save_path):
    img_norm = image / 255.0
    fd, hog_img = hog(img_norm, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, channel_axis=None)
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, hog_img.max()))
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(hog_img_rescaled, cmap='gray')
    plt.title("HOG Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# =========================================================
# 4. MODEL TRAINING & LOOCV EVALUATION
# =========================================================
def loocv_evaluation(X, y, svm_params):
    print("\nStarting Leave-One-Out Cross Validation (LOOCV)...")
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    start_time = time.time()

    for train_idx, test_idx in tqdm(loo.split(X), total=len(X), desc="LOOCV Progress", ncols=80):
        model = SVC(**svm_params)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        y_true.extend(y[test_idx])
        y_pred.extend(pred)

    elapsed = (time.time() - start_time) / 60
    print(f"\nLOOCV completed in {elapsed:.2f} minutes.")
    return np.array(y_true), np.array(y_pred)

# =========================================================
# 5. SAVE RESULT 
# =========================================================
def save_results(y_true, y_pred, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    # Save metrics
    metrics_path = os.path.join(out_dir, "Performance_Matrix.csv")
    pd.DataFrame([{"Accuracy": acc, "Precision": prec, "F1_Score": f1}]).to_csv(metrics_path, index=False)

    # Save predictions
    pred_log_path = os.path.join(out_dir, "Prediction_Log.csv")
    pd.DataFrame({"True_Label": y_true, "Predicted_Label": y_pred}).to_csv(pred_log_path, index=False)

    # Save confusion matrix
    labels = [chr(64 + i) for i in range(1, 27)]
    plt.figure(figsize=(10, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - EMNIST (HOG + SVM, 13.000 samples, LOOCV)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    conf_path = os.path.join(out_dir, "Confusion_Matrix.png")
    plt.tight_layout()
    plt.savefig(conf_path, dpi=150)
    plt.close()

    print(f"\nAccuracy : {acc:.4f}\nPrecision: {prec:.4f}\nF1-Score : {f1:.4f}")
    return acc, prec, f1, conf_path

# =========================================================
# 6. GUI DISPLAY
# =========================================================
def show_results_gui(acc, prec, f1, conf_img_path):
    root = tk.Tk()
    root.title("Hasil Evaluasi EMNIST Letters (13.000 Sample - LOOCV)")
    root.geometry("700x800")

    ttk.Label(root, text="HASIL KLASIFIKASI EMNIST LETTERS (13.000 Sample - LOOCV)",
              font=("Arial", 14, "bold")).pack(pady=10)
    ttk.Label(root, text=f"Accuracy : {acc:.4f}", font=("Arial", 12)).pack(pady=5)
    ttk.Label(root, text=f"Precision: {prec:.4f}", font=("Arial", 12)).pack(pady=5)
    ttk.Label(root, text=f"F1-Score : {f1:.4f}", font=("Arial", 12)).pack(pady=5)
    ttk.Separator(root, orient="horizontal").pack(fill="x", pady=10)

    img = Image.open(conf_img_path).resize((650, 650))
    img_tk = ImageTk.PhotoImage(img)
    lbl = ttk.Label(root, image=img_tk)
    lbl.image = img_tk
    lbl.pack(pady=10)

    ttk.Button(root, text="Tutup", command=root.destroy).pack(pady=10)
    root.mainloop()

# =========================================================
# 7. MAIN PROGRAM
# =========================================================
def main():
    cfg = initialize_project()

    print("=" * 70)
    print("EMNIST LETTERS CLASSIFICATION - 13.000 Sample (LOOCV)")
    print("Method: HOG Feature + SVM (RBF) + LOOCV")
    print("=" * 70)

    # Load dataset
    X, y = load_and_sample_data(
        cfg["csv_path"], cfg["n_samples_per_class"], cfg["n_classes"], cfg["seed"]
    )

    # Visualize samples
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i], cmap="gray")
        ax.set_title(chr(64 + y[i]))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["result_dir"], "Sample_Images.png"), dpi=150)
    plt.close()

    # Visualize one HOG example
    visualize_hog_sample(X[0], os.path.join(cfg["result_dir"], "Hog_Visualization.png"))

    # Extract HOG features
    X_feat = extract_hog_features(X)
    X_feat = StandardScaler().fit_transform(X_feat)

    # LOOCV Evaluation
    y_true, y_pred = loocv_evaluation(X_feat, y, cfg["svm_params"])

    # Save and display results
    acc, prec, f1, conf_path = save_results(y_true, y_pred, cfg["result_dir"])
    show_results_gui(acc, prec, f1, conf_path)

if __name__ == "__main__":
    main()
