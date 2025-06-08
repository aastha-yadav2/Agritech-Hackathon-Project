# AgriLeaf 🌱

**AI-Powered Plant Disease Detection Using Deep Learning**

AgriLeaf is a user-friendly AI project to help farmers and agriculture enthusiasts identify plant diseases from leaf images. Using a Convolutional Neural Network (CNN) and the PlantVillage dataset, this project classifies plant diseases and provides actionable remedies, all via a simple web interface powered by Streamlit.

---

## Features

- **Multi-class Plant Disease Detection**  
  Classifies leaf images into various disease categories (supports 38+ crops and diseases as per PlantVillage dataset).
- **Remedy Suggestions**  
  Offers basic treatment and prevention advice for each detected disease.
- **Web Application**  
  Easy-to-use Streamlit app for uploading and diagnosing leaf images.
- **Modular Codebase**  
  Clean separation for model training, inference, and web UI.

---

## Demo

![AgriLeaf Demo Screenshot](demo_screenshot.png)  
*Upload a leaf image and instantly get disease prediction and recommendations!*

---

## Project Structure

```
AgriLeaf/
├── src/
│   ├── model/
│   │   ├── train_model.py        # Train the CNN model
│   │   └── inference.py          # Predict disease from image
│   └── app/
│       └── streamlit_app.py      # Streamlit web application
├── data/
│   ├── disease_remedy.json       # Disease-to-remedy mapping
│   └── PlantVillage/             # (Download dataset from Kaggle)
├── models/
│   ├── agri_leaf_cnn.h5          # Trained model file (created after training)
│   └── class_indices.npy         # Class index mapping (created after training)
├── requirements.txt
└── README.md
```

---

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/AgriLeaf.git
cd AgriLeaf
```

### 2. Download the Dataset

- Download the [PlantVillage dataset from Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease).
- Extract and place the dataset folder as:  
  `data/PlantVillage/`

> ⚠️ **Do NOT upload the dataset to GitHub.**  
> Keep it local due to size and licensing.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional: Use pre-trained model if available)

```bash
python src/model/train_model.py
```
- This will save `agri_leaf_cnn.h5` and `class_indices.npy` in the `models/` directory.

### 5. Run Inference on a Single Image

```bash
python src/model/inference.py path_to_leaf_image.jpg
```

### 6. Run the Web Application

```bash
streamlit run src/app/streamlit_app.py
```
- Open the provided local URL in your browser.
- Upload a leaf image to get predictions and remedies.

---

## Notes

- **Remedies:** Edit or extend `data/disease_remedy.json` to add or update disease advice.
- **Model Tuning:** Adjust hyperparameters or model architecture in `train_model.py` as needed.
- **Contributions:** Feel free to fork and open pull requests for improvements.

---

## How to Add This Project to Your GitHub Repository

1. **Prepare your project directory**  
   Organize your files as shown in the structure above.

2. **Initialize a Git repo**  
   In your project folder, open a terminal and run:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Add AgriLeaf plant disease detection project"
   ```

3. **Create a new repo on GitHub**  
   - Go to [GitHub New Repository](https://github.com/new).
   - Name it `AgriLeaf` (or any name you like).
   - Do NOT initialize with a README (you already have one).

4. **Add your remote and push**  
   Replace `<your-username>` with your GitHub username:
   ```bash
   git remote add origin https://github.com/<your-username>/AgriLeaf.git
   git branch -M main
   git push -u origin main
   ```

5. **Done!**  
   Share your repository link for submissions or collaboration.

---

## License

This project is for educational and research purposes.  
Please cite the PlantVillage dataset and respect its license.

---

## Acknowledgements

- [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- TensorFlow/Keras, Streamlit, and open-source contributors

---
