# GEOL0069_Week4
Week 4 coursework for GEOL0069.

# 🧊 Sea Ice & Lead Classification via Unsupervised Learning
### GEOL0069 – Artificial Intelligence for Earth Observation | Week 4 Assignment

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Background & Scientific Context](#background--scientific-context)
3. [Methods](#methods)
   - [K-Means Clustering](#1-k-means-clustering)
   - [Gaussian Mixture Models (GMM)](#2-gaussian-mixture-models-gmm)
4. [Results](#results)
   - [Echo Waveform Analysis](#echo-waveform-analysis)
   - [Confusion Matrix & Accuracy](#confusion-matrix--accuracy)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Data](#data)
6. [Repository Structure](#repository-structure)
7. [Contact](#contact)
8. [Acknowledgements](#acknowledgements)

---

## Project Overview

This project applies **unsupervised machine learning** to classify Sentinel-3 radar altimetry echoes as either **sea ice** or **leads** (gaps of open water within sea ice). The GMM-based classifier achieves **~99.6% overall accuracy** against the ESA official surface-type flags, and the resulting mean echo shapes with standard deviation envelopes clearly capture the physical difference between the two surface types.

The notebook builds directly on `Chapter1_Unsupervised_Learning_Methods_Michel.ipynb` and extends it with:

- Full altimetry preprocessing pipeline (peakiness, stack standard deviation, NaN removal)
- Both K-Means and GMM classification, with cluster count inspection
- Mean ± standard deviation waveform plots for each class
- **Sub-bin FFT waveform alignment** to remove tracker-range jitter before computing aligned means
- Quantitative evaluation via a **confusion matrix and classification report** against ESA ground truth

---

## Background & Scientific Context

The **Sentinel-3 satellite mission** (ESA/Copernicus) carries a SAR radar altimeter (SRAL) whose primary goal is to measure sea-surface topography. Over the polar oceans it transmits Ku-band radar pulses toward the surface; the backscattered power as a function of time — called an **echo** or **waveform** — encodes information about the physical nature of the surface.

- **Sea ice** returns a broad, gently-sloped echo reflecting a rough, heterogeneous surface with lower peak power.  
- **Leads** (narrow channels of open water between ice floes) act as specular reflectors, returning a narrow, high-power spike — similar to a calm ocean surface.

Distinguishing these two classes automatically is critical for accurate freeboard and sea-ice thickness retrievals. Here we demonstrate that their waveform shapes are separable in a two-feature space (waveform **peakiness** and **stack standard deviation**) without any labelled training data.

> For further reading see: Zhong et al. (2023), *Remote Sensing*, and the [Sentinel-3 Altimetry User Guide](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-3-altimetry).

---

## Methods

### 1. K-Means Clustering

K-Means partitions the feature space into *k* clusters by iteratively assigning each point to the nearest centroid and recomputing centroids until convergence.

**Why K-Means?**
- No assumptions about cluster shape or probability distribution.
- Computationally efficient and easy to interpret.
- A natural baseline before applying more expressive models.

**Key parameters used:** `k = 2`, `init='k-means++'`, `n_init=10`

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data_cleaned)
clusters_kmeans = kmeans.predict(data_cleaned)
```

**Limitations:** K-Means assumes spherical, equal-variance clusters — a poor fit when the two surface classes differ substantially in spread, as is the case here for sea ice versus leads.

---

### 2. Gaussian Mixture Models (GMM)

A GMM models the data as a weighted sum of *K* multivariate Gaussian distributions, each with its own mean **μ** and covariance **Σ**. Parameters are estimated via the **Expectation-Maximisation (EM)** algorithm:

- **E-step**: compute the posterior probability that each point belongs to each component.  
- **M-step**: update **μ**, **Σ**, and mixing weights to maximise the data log-likelihood.

This yields a **soft clustering** — each echo is assigned a probability of being sea ice or lead — and naturally accommodates the different spreads of the two classes.

**Key parameters used:** `n_components = 2`, `random_state = 0`

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_cleaned)
clusters_gmm = gmm.predict(data_cleaned)
```

**Cluster counts from GMM prediction:**
| Cluster | Count | Assigned Class |
|---------|-------|----------------|
| 0 | 8 880 | Sea Ice |
| 1 | 3 315 | Lead |

---

## Results

### Echo Waveform Analysis

After classification, the mean echo waveform and ±1 standard deviation envelope were computed for each class. Waveforms were also **aligned at the sub-bin level** using FFT oversampling (×24, corresponding to ~1 cm range resolution) to remove tracker-range jitter before stacking, producing cleaner composite shapes.

| Observation | Sea Ice | Lead |
|-------------|---------|------|
| Peak power | Lower | Higher |
| Waveform shape | Broad, gradual decay | Sharp spike, rapid decay |
| Standard deviation envelope | Narrower (more consistent) | Wider (more variable) |

The higher variability in lead waveforms reflects differences in lead width, surface roughness, and freeze/refreeze state. The lower power of sea ice echoes is consistent with its rougher surface reducing specular backscatter (Jiang & Wu, 2004).

*Figures (mean ± std and aligned waveforms) are generated inline in the notebook.*

---

### Confusion Matrix & Accuracy

The GMM predictions were compared against the **ESA official surface-type classification** flags from the Sentinel-3 L2 product (flag value 1 = sea ice, 2 = lead). The ESA labels were offset by −1 (0 = sea ice, 1 = lead) to match the cluster indices before computing the matrix.

```
Confusion Matrix (rows = ESA truth, columns = GMM predicted):

              Pred: Sea Ice  Pred: Lead
True: Sea Ice      8856          22
True: Lead           24        3293
```

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Sea Ice (0) | 1.00 | 1.00 | 1.00 | 8 878 |
| Lead (1) | 0.99 | 0.99 | 0.99 | 3 317 |
| **Overall accuracy** | | | **1.00** | **12 195** |

The GMM correctly classifies 99.6% of all echoes, with only 46 misclassifications out of 12 195. This demonstrates that the two waveform-derived features (peakiness and stack standard deviation) are highly discriminative for this binary classification task.

---

## Getting Started

### Prerequisites

- A Google account with access to [Google Colab](https://colab.research.google.com/)
- Access to [Google Drive](https://drive.google.com/) (for storing data and the notebook)
- Sentinel-3 SRAL L2 NetCDF file (see [Data](#data) section below)

### Installation

The notebook runs entirely in Google Colab. The only non-standard packages that need to be installed are:

```python
!pip install netCDF4
!pip install rasterio
!pip install basemap
!pip install cartopy
```

Mount your Google Drive at the start of the session:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Data

The Sentinel-3 SAR altimetry data used in this project is a Level-2 `.SEN3` product. It is available via the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).

The specific file used is:

```
S3B_SR_2_LAN_SI_20190301T231304_20190301T233006_20230405T162425_1021_022_301______LN3_R_NT_005.SEN3
```

> ⚠️ The data file is not included in this repository due to its size. Please download it from the Copernicus Data Space and place it in your Google Drive before running the notebook.

The base notebook (`Chapter1_Unsupervised_Learning_Methods_Michel.ipynb`) is available at:  
🔗 https://drive.google.com/file/d/1HDSLjsWhLIDF-qbRj6sbGVd9t1LB7890/view?usp=drive_link

---

## Repository Structure

```
📦 GEOL0069-Week4
 ┣ 📓 Unit_2_Unsupervised_Learning_Methods.ipynb   # Main assignment notebook
 ┣ 📓 Chapter1_Unsupervised_Learning_Methods_Michel.ipynb  # Base reference notebook
 ┗ 📄 README.md                                    # This file
```

---

## Contact

**Your Name** – your.email@ucl.ac.uk

Project Link: `https://github.com/YOUR_USERNAME/GEOL0069-Week4`

---

## Acknowledgements

- This project is part of the module **GEOL0069 – Artificial Intelligence for Earth Observation**, taught in the UCL Earth Sciences Department.
- Base notebook provided by **Dr Michel Tsamados**, UCL.
- Sentinel-3 data courtesy of the **European Space Agency (ESA)** via the Copernicus programme.
- Waveform classification methodology informed by **Zhong et al. (2023)** and **Jiang & Wu (2004)**.

---

<p align="right"><a href="#-sea-ice--lead-classification-via-unsupervised-learning">↑ Back to top</a></p>
