# 🎬 Multimodal Movie Genre Classification

A Machine Learning and Deep Learning project developed as part of a Data Science internship associated with IIT Guwahati. The project explores **multimodal learning** by combining movie poster images and plot descriptions to predict movie genres using text and visual features.

## 📌 Project Overview

Movie genre classification is a multi-label classification problem because a single movie can belong to multiple genres, such as Action, Drama, Comedy, or Thriller.

This project investigates how combining textual information from movie plots with visual information from movie posters can improve genre prediction compared with using either modality independently.

### Objectives

- Build a text-based baseline for movie genre classification.
- Build an image-based baseline using movie posters.
- Explore multimodal fusion techniques to combine text and image features.
- Evaluate models using multi-label classification metrics.
- Compare the performance of individual modalities with fused representations.

## 🗂️ Dataset

The project uses the **MM-IMDb / MM-IMDB multimodal movie dataset**, which contains movie plot descriptions, poster images, and genre labels.

The dataset used in the project contains the following feature dimensions:

| Component | Shape |
|---|---|
| Text features | `(25959, 300)` |
| Genre labels | `(25959, 23)` |
| Poster images | `(25959, 3, 256, 160)` |
| Pre-extracted VGG features | `(25959, 4096)` |

The dataset contains 23 genre labels, making this a multi-label classification task.

*Note: These dimensions describe the project data representation. Verify them against the final processed dataset before publication.*

## 🧠 Methodology

### 1. Text-Based Classification

Textual representations of movie plots are used to predict genre labels.

Techniques explored include:

- DistilBERT-based text representation and classification.
- Preprocessed numerical text features.
- Binary Cross-Entropy Loss for multi-label prediction.

### 2. Image-Based Classification

Movie poster images provide visual information that may indicate genre.

The image-processing workflow includes:

- Poster image preprocessing and resizing.
- Transfer learning using a pretrained MobileNetV2 model.
- Extraction of visual representations from posters.
- Classification across multiple genre labels.

### 3. Multimodal Fusion

The project explores combining text and visual representations to capture complementary information.

Fusion strategies investigated include:

- **Early Fusion:** Concatenating text and image features before classification.
- **Gated Fusion:** Learning how much information to use from each modality.
- **Attention Fusion:** Exploring attention-based interactions between modalities.

The fused representation is used to predict the movie's genre labels.

## 🏗️ Workflow

1. Load and inspect the multimodal dataset.
2. Preprocess text, poster images, and genre labels.
3. Extract or load text and visual features.
4. Train text-only and image-only baseline models.
5. Implement multimodal fusion architectures.
6. Evaluate models on the appropriate data split.
7. Compare classification performance using multi-label metrics.

## 📊 Evaluation Metrics

The following metrics are used to evaluate the models:

| Metric | Purpose |
|---|---|
| Micro F1-score | Measures overall multi-label performance by aggregating decisions across labels. |
| Macro F1-score | Gives equal importance to each genre label. |
| Hamming Loss | Measures the fraction of incorrectly predicted labels. |
| ROC Curve | Evaluates discrimination performance across classification thresholds. |

## 🛠️ Technologies Used

- **Programming:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Natural Language Processing:** Hugging Face Transformers
- **Machine Learning:** Scikit-learn
- **Data Processing:** NumPy, Pandas, h5py
- **Visualization:** Matplotlib
- **Development Environment:** Jupyter Notebook / Google Colab
- **Version Control:** Git and GitHub

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Soumik-011204/Internship-project-at-IIT-Guwahati.git
cd Internship-project-at-IIT-Guwahati
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install torch torchvision transformers scikit-learn numpy pandas h5py matplotlib jupyter
```

*For reproducibility, use the dependency versions specified by the project, if a `requirements.txt` file is available.*

## ▶️ Running the Project

1. Open the repository in Jupyter Notebook or Google Colab.
2. Download or configure access to the required dataset.
3. Update dataset paths in the notebooks or configuration files.
4. Execute the preprocessing and feature-extraction steps.
5. Train the baseline models.
6. Run the multimodal fusion experiments.
7. Review the evaluation metrics and generated plots.

The exact notebook execution order depends on the files included in the repository.

## 📁 Project Structure

The repository's exact file structure should be documented after verifying its contents. A suggested organization is:

```text
Internship-project-at-IIT-Guwahati/
├── notebooks/          # Experiment notebooks
├── data/               # Dataset or data references
├── models/             # Model definitions and checkpoints
├── results/            # Evaluation outputs and plots
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

*This is a suggested structure, not a verified listing of the current repository.*

## 🔬 Key Learning Outcomes

- Understanding multi-label classification.
- Working with text and image modalities.
- Applying transfer learning to image classification.
- Building deep learning pipelines with PyTorch.
- Exploring early, gated, and attention-based feature fusion.
- Evaluating models using classification metrics.
- Managing experiments and documenting machine learning workflows.

## 🚀 Future Improvements

- Experiment with stronger pretrained vision and language models.
- Tune classification thresholds independently for each genre.
- Improve performance on underrepresented genres.
- Optimize multimodal fusion architectures.
- Add reproducible training configurations and model comparison visualizations.
- Develop an interface for predicting genres from a movie poster and plot description.

## 👨‍💻 Author

**Soumik Sarkar**

B.Tech in Computer Science and Engineering  
Jorhat Engineering College

- GitHub: [Soumik-011204](https://github.com/Soumik-011204)

