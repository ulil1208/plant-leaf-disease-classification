# 🌿 Web-Based Plant Leaf Disease Classification

Proyek ini merupakan implementasi Tugas Akhir Pengolahan Citra Digital berupa
aplikasi web untuk klasifikasi penyakit daun tanaman tomat menggunakan
metode **Convolutional Neural Network (CNN) MobileNetV2**.

## 📌 Dataset
Dataset yang digunakan adalah **PlantVillage Dataset** dari Kaggle,
dengan empat kelas:
- Tomato_Early_blight
- Tomato_Late_blight
- Tomato_Leaf_Mold
- Tomato_healthy

Dataset tidak disertakan dalam repository ini karena ukuran yang besar.

## ⚙️ Metodologi
1. Pre-processing:
   - Gaussian Filter
   - Median Filter
   - Histogram Equalization
2. Feature Extraction:
   - GLCM
   - Local Binary Pattern (LBP)
3. Classification:
   - CNN MobileNetV2 (Transfer Learning)
4. Evaluation:
   - Accuracy
   - Precision
   - Recall
   - F1-score

## 🚀 Cara Menjalankan Aplikasi
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

# plant-leaf-disease-classification
