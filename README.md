# 🧠 McDonald's Market Segmentation (Fast Food Case Study)

A machine learning project to replicate a market segmentation case study on McDonald's customers using clustering techniques, powered by Python and scikit-learn.

---

## ✨ Features

- 💬 Customer segmentation using unsupervised learning (KMeans)
- 📊 Feature preprocessing (scaling, encoding)
- 🔍 Elbow method to determine optimal number of clusters
- 🧬 PCA-based cluster visualization
- 📄 Modular and scalable codebase structured like an ML pipeline
- 🚀 Ready for deployment and GitHub publication

---

## 🚀 Tech Stack

| Technology          | Purpose                           |
|---------------------|-----------------------------------|
| Python              | Core programming language         |
| pandas              | Data manipulation and I/O         |
| scikit-learn        | Clustering and preprocessing      |
| matplotlib          | Visualizations                    |
| Flask (optional)    | Web deployment interface          |
| Docker (optional)   | Containerization                  |
| GitHub              | Version control and collaboration |

---

## 🏗️ Architecture

```plaintext
┌────────────────────────┐     ┌────────────────────────┐
│  Dataset (CSV File)    │     │   main.py              │
└────────────┬───────────┘     └────────┬───────────────┘
             │                          │
             ▼                          ▼
   ┌──────────────────┐        ┌──────────────────────┐
   │ Load & Preprocess│◀──────▶│   src/ modules      │
   └────────┬─────────┘        └────────┬─────────────┘
            ▼                           ▼
   ┌──────────────────┐        ┌──────────────────────┐
   │ Clustering Logic │        │   Visualization      │
   └────────┬─────────┘        └────────┬─────────────┘
            ▼                           ▼
     Outputs: Cluster labels, PCA plots, Elbow chart
```

---

## 📁 Project Structure

```
fastfood_segmentation/
├── data/                         # Dataset CSV
├── models/                       # (Optional) Saved models
├── outputs/                      # Plots and analysis results
├── src/
│   ├── data/                     # Data loading
│   ├── preprocessing/            # Data cleaning, encoding, scaling
│   ├── model/                    # Clustering (KMeans)
│   └── visualization/            # Elbow & cluster plots
├── main.py                       # Pipeline entry point
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```

---

## 🔧 Installation

### Prerequisites
- Python 3.10+

### 🛠 Manual Setup

```bash
git clone https://github.com/your-username/mcdonalds-segmentation.git
cd mcdonalds-segmentation
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the full pipeline:

```bash
python main.py
```

Expected outputs:
- `outputs/elbow_plot.png`
- `outputs/cluster_plot.png`

---

## 📄 Dataset Description

| Column           | Type      | Description                                   |
|------------------|------------|----------------------------------------------|
| yummy, cheap...  | Categorical| Yes/No features about food perception        |
| Like             | Integer    | Rating from -3 to +3                         |
| Age              | Integer    | Customer age                                 |
| VisitFrequency   | Categorical| Visit frequency (encoded)                    |
| Gender           | Categorical| Gender (encoded)                             |

---

## 📈 Output Samples
- **Elbow plot** for optimal clusters
- **PCA cluster plot** for customer visualization

---

## 🧪 Testing (Optional)
If tests are added:
```bash
python -m pytest tests/
```

---

## 📝 License
MIT License

---

## 🙏 Acknowledgements
- Book: *Market Segmentation Analysis*
- scikit-learn open source community
- Inspiration from McDonald's case study

---

## 📞 Contact
For questions or suggestions, open an issue or contact:
**Vishal Gorule** – [gorulevishal984@gmail.com] – [Vision Expo]
