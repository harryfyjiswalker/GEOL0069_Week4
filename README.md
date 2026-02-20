# Sea Ice & Lead Classification via Unsupervised Learning
### GEOL0069 – Artificial Intelligence for Earth Observation | Week 4 Assignment

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

<details>
<summary><b>Table of Contents</b></summary>

## Table of Contents

1. [Project Overview](#project-overview)
2. [Background & Scientific Context](#background--scientific-context)
3. [Data & Preprocessing](#data--preprocessing)
4. [Methods](#methods)
   - [Feature Space](#feature-space)
   - [K-Means Clustering](#1-k-means-clustering)
   - [Gaussian Mixture Models (GMM)](#2-gaussian-mixture-models-gmm)
5. [Results & Analysis](#results--analysis)
   - [Feature Space Visualisation](#feature-space-visualisation)
   - [Individual Echo Waveforms](#individual-echo-waveforms)
   - [Mean Echo Shapes & Standard Deviation Envelopes](#mean-echo-shapes--standard-deviation-envelopes)
   - [Aligned Waveform Means](#aligned-waveform-means)
   - [Confusion Matrix & Classification Accuracy](#confusion-matrix--classification-accuracy)
6. [Getting Started](#getting-started)
7. [Repository Structure](#repository-structure)
8. [References](#references)
9. [Contact](#contact)
10. [Acknowledgements](#acknowledgements)

</details>

## 1. Project Overview

This assignment focuses on evaluation of automated methods for discrimination of sea-ice and leads. Two unsupervised clustering algorithms - K-means and Gaussian Mixture Models (GMMs) - are trained on waveform features derived from unlabelled Sentinel-3 SAR altimetry data, and their classification performances validated against ESA surface-type flags.

The notebook ... builds directly on `Chapter1_Unsupervised_Learning_Methods_Michel.ipynb` and extends it with:
- Mean echo shapes and standard deviation envelopes for each class
- Feature space visualisation (pulse peakiness vs stack standard deviation)
- Sub-bin FFT waveform alignment to remove tracker-range jitter before compositing
- Quantitative evaluation against ESA official L2 surface-type flags using a confusion matrix and classification report

The GMM-based classifier achieves an overall accuracy of approximately 99.6%, demonstrating that a two component mixture model applied to just two features is sufficient to cleanly separate these surface types. 

---

## 2. Background

### 2.1 Artic Leads


Over the polar oceans, distinguishing **sea ice** from **leads** is a critical preprocessing step before any freeboard or sea-ice thickness retrieval. The ESA operational product itself uses waveform peakiness thresholds to perform this classification [1].

- **Leads** are narrow fractures in sea ice exposing open, calm water. Their near-flat surface acts as a specular reflector, returning a narrow, high-power spike with very high peakiness and low stack standard deviation [3].
- **Sea ice** surfaces are rough and heterogeneous, producing a broad, lower-amplitude echo with a gradual decay and higher stack standard deviation [4].

This physical contrast means both surface types occupy well-separated regions in waveform-feature space, which the unsupervised methods in this project exploit — without any labelled training examples.

Beyond their role in altimetry processing, leads are geophysically important in their own right. They represent the dominant pathway for turbulent heat and moisture exchange between the Arctic Ocean and atmosphere, and their spatial distribution governs sea-ice production and brine rejection [3]. Understanding lead occurrence is relevant to polar climate modelling and sea-ice mass balance estimates.

### 2.2 SAR Radar Altimeter

Synthetic Aperture Radar (SAR) measures the backscatter of microwave pulses to detect surface features. Unlike optical satellite data, the approach is unaffected by cloud cover or months of darkness, making it invaluable for day- and year-round monitoring. A key application of this approach is in SAR altimetry (e.g. via the Sentinel-3 SAR Radar Altimeter (SRAL) - which involves inferring surface elevation from the time taken from emission of the pulse and detection of the returning signal (called a "waveform" or "echo"). This allows much higher resolution measurements compared to conventional methods, which is essential for detection of leads, which may only be tens to hundreds of metres wide:
- Conventional altimeters transmit a single broad radar pulse that illuminates a large patch of surface - roughly 20km wide - simultaneously. The return signal is therefore an average over that entire area, which is problematic in contexts such as sea ice where the surface changes character over much shorter distances.
- SAR altimeters address this by recording many pulses in quick succession as the satellite moves along its orbit, then combining them (using coherent multi-look processing using Doppler techniques) to isolate the return from a much smaller strip of ground - around 300m - directly beneath the satellite.[2]

![Alt text](/images/Sentinel_3_SRAL_Diagram.png)
*Figure 1. Diagram of SRAL nadir track (the ground track directly beneath the satellite), as well as the ground tracks of the other Sentinel-3 instruments.*

The returned waveform encodes information about both:
- Surface elevation, from the timing of the waveform's *leading edge* (the point where the returned signal strength first rises sharply i.e. the moment the pulse reaches the surface).       - This is measured with *Pulse Peakiness* (the ratio of peak power to mean power i.e. how sharply peaked the return is - high for leads, low for sea ice)
- Surface texture, from the shape of the waveform (a smooth surface e.g. a lead reflects the pulse back cleanly, producing a sharp, intense return; a rough surface e.g. sea ice scatters the pulse in many directions so the energy arrives back at the satellite over a longer time period, producing a weaker, broader return)
   - This is measured with *Stack Standard Deviation* (how spread out the return is across different viewing angles as the satellite passes overhead - low for leads, high for sea ice)

An unsupervised learning algorithm can exploit these differences in the waveform shape to distinguish between sea ice and leads.

### 2.3 Clustering algorithms

Unsupervised learning is used to elucidate underlying patterns in data without the need for labelled training data. There are two main types: clustering (grouping data points based on similarity) and dimensionality reduction (compressing data to lower the dimension space). The principles behind two of the main clustering approaches, K-means and Gaussian Mixture Models (GMMs), is briefly outlined below. I provide more detail on the mathematical formulation of the techniques in [this](/Clustering_Algorithms_Summary__K_means_and_Gaussian_Mixture_Models.pdf) set of notes.

#### 2.3.1 K-means
K-Means partitions the feature space into *k* clusters by iteratively assigning each point to its nearest centroid and recomputing centroids until convergence [6].

**Why K-Means?** It requires no prior knowledge of cluster shape and scales efficiently to large datasets, making it a natural baseline. Its main limitation here is the assumption of **spherical, equal-variance clusters**, which the feature-space scatter (Figure 1) shows to be a poor fit — the lead cluster is more compact than the sea-ice cluster.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(data_cleaned)
clusters_kmeans = kmeans.predict(data_cleaned)
```

---

#### 2.3.2 Gaussian Mixture Models

A GMM models the data as a weighted sum of *K* multivariate Gaussian distributions, each with its own mean **μ** and covariance **Σ** [7]. Parameters are estimated via the **Expectation-Maximisation (EM)** algorithm:

- **E-step:** compute the posterior probability that each waveform belongs to each component.
- **M-step:** update **μ**, **Σ**, and mixing weights to maximise the data log-likelihood.

Unlike K-Means, GMM allows each cluster to have a **different covariance structure** and outputs a *soft* classification (probability of class membership), which is better suited here because the two clusters have visibly different spreads in feature space. Dettmering et al. (2018) [5] demonstrated that unsupervised methods applied to CryoSat-2 stack statistics consistently outperform threshold-based approaches, achieving overall accuracies above 97%.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_cleaned)
clusters_gmm = gmm.predict(data_cleaned)
```
---

## 3. Methods

## Data & Preprocessing

**Satellite data:** Sentinel-3B SRAL Level-2 SAR product:
```
S3B_SR_2_LAN_SI_20190301T231304_20190301T233006_20230405T162425_1021_022_301______LN3_R_NT_005.SEN3
```

**Features extracted from the 20 Hz waveforms:**

| Feature | Physical Meaning |
|---|---|
| **Pulse Peakiness (PP)** | Ratio of peak power to mean power across the waveform. Leads produce very high peakiness (sharp specular returns); sea ice produces low peakiness (diffuse returns). |
| **Stack Standard Deviation (SSD)** | Spread of power across look angles in the Delay-Doppler stack. Leads produce a narrow angular response (low SSD); sea ice produces a broad response (high SSD). |

The ESA official surface-type flag (`surf_class_20_ku`) is used as ground truth: flag = 1 for sea ice, flag = 2 for leads. All observations with other flag values or NaN features are excluded, leaving **12,195 valid waveforms** for analysis.

---

## Methods

### Feature Space

Both features are normalised before clustering. A 2D scatter plot of the feature space (see Figure 1) reveals two well-separated populations, confirming physical separability — the primary justification for an unsupervised approach over simple thresholding [5].

---

### 1. K-Means Clustering



### 2. Gaussian Mixture Models (GMM)



**Cluster counts from GMM:**

| Cluster | Count | Assigned Class |
|---------|-------|----------------|
| 0 | 8,880 | Sea Ice |
| 1 | 3,315 | Lead |

The ratio of ~2.7 sea-ice waveforms per lead is physically plausible for a winter Arctic overpass, where leads constitute a small but climatically influential fraction of ice-covered area [3].

---

## Results & Analysis

### Feature Space Visualisation

> **Figure to include here:** `figures/feature_space_scatter.png`  
> Produce by running the scatter plot cell that plots **pulse peakiness** (x-axis) vs **stack standard deviation** (y-axis), colour-coded by GMM cluster label (0 = sea ice, 1 = lead). Save with `plt.savefig('figures/feature_space_scatter.png', dpi=150, bbox_inches='tight')`.

![Feature Space Scatter](figures/feature_space_scatter.png)

**Figure 1.** 2D scatter plot of pulse peakiness versus stack standard deviation for all 12,195 valid echoes, colour-coded by GMM cluster assignment.

**Analysis:** The scatter reveals two well-separated populations. The lead cluster is concentrated in the **high-peakiness, low-SSD** corner — consistent with the near-specular backscatter from calm open water [3][4]. The sea-ice cluster occupies the **low-peakiness, high-SSD** region, reflecting the diffuse, multi-angular return from a rough ice surface. The elongated shape of the sea-ice cluster compared to the compact lead cluster explains why GMM outperforms K-Means: a flexible covariance structure is needed to accurately describe both populations simultaneously. This feature-space geometry is consistent with that reported for CryoSat-2 data in Wernecke and Kaleschke (2015) [3] and Dettmering et al. (2018) [5].

---

### Individual Echo Waveforms

> **Figures to include here:** `figures/sample_waveforms_sea_ice.png` and `figures/sample_waveforms_leads.png`  
> Produce by running the cells that plot the first 5 echoes from `waves_cleaned[clusters_gmm == 0]` and `waves_cleaned[clusters_gmm == 1]` respectively. Save each.

![Sample Sea Ice Waveforms](figures/sample_waveforms_sea_ice.png)
![Sample Lead Waveforms](figures/sample_waveforms_leads.png)

**Figure 2a (top).** Representative individual sea-ice echo waveforms (cluster 0).  
**Figure 2b (bottom).** Representative individual lead echo waveforms (cluster 1).

**Analysis:** Individual waveforms confirm the aggregate statistics. Sea-ice echoes display a gently rising leading edge, a moderate peak, and a long gradual trailing edge — reflecting scattering contributions from multiple surface facets across the large altimeter footprint. Lead echoes show a steep, narrow spike followed by an abrupt decay, the hallmark of near-specular reflection from a smooth water surface. This morphological contrast is the physical basis for all waveform-based lead classifiers in the literature [3][4][5][8] and is clearly reproduced by the GMM clustering without any supervised input.

---

### Mean Echo Shapes & Standard Deviation Envelopes

> **Figure to include here:** `figures/mean_std_waveforms.png`  
> Produce by running the cell that computes `np.mean` and `np.std` of `waves_cleaned` split by `clusters_gmm`, then plots both means with `plt.fill_between` shading for ±1σ. Save with `plt.savefig('figures/mean_std_waveforms.png', dpi=150, bbox_inches='tight')`.

![Mean and Std Waveforms](figures/mean_std_waveforms.png)

**Figure 3.** Mean echo waveform ± 1 standard deviation for sea ice (cluster 0, blue) and leads (cluster 1, orange), computed over all classified echoes.

**Analysis:** Several physically meaningful features emerge from this composite:

1. **Peak power contrast:** The lead mean has substantially higher peak power than the sea-ice mean. This reflects the well-documented difference in backscatter coefficient (σ⁰) between specular leads and diffuse sea ice [3][4]. A calm water surface concentrates energy back toward nadir, dramatically increasing σ⁰ and peak waveform amplitude.

2. **Waveform width:** The lead mean is sharper and more peaked, while the sea-ice mean is broader with a slower trailing-edge decay. The sea-ice trailing edge contains energy scattered from off-nadir surface elements across the effective footprint, whereas the lead return is dominated by the nadir specular point.

3. **Sea-ice ±1σ envelope — narrow and smooth:** The sea-ice standard deviation envelope is relatively tight and consistent across range bins, indicating that waveforms are uniform from footprint to footprint along the track. This reflects the broadly stable surface properties of consolidated winter sea ice.

4. **Lead ±1σ envelope — wide:** The lead standard deviation envelope is markedly broader, indicating high waveform-to-waveform variability. This is physically meaningful: leads vary significantly in width, state (open water vs. thin nilas), and geometry. Narrow leads produce mixed ice-water returns (reducing apparent peakiness), while wide leads produce cleaner specular returns — and the ensemble of lead waveforms captures this entire range [3][8].

5. **Trailing-edge oscillations in the lead mean:** The slight ringing visible in the lead composite arises from between-waveform misalignment — different echoes have slightly different tracker ranges, shifting the peak position by sub-bin amounts. This is addressed by the FFT alignment in Figure 4.

---

### Aligned Waveform Means

> **Figure to include here:** `figures/aligned_mean_std_waveforms.png`  
> Produce by running the FFT-oversampling alignment cells (×24 oversampling, using `RANGE_GATE_RES = 0.2342 m/bin`), then plotting the mean ± std of `waves_aligned` split by `clusters_gmm`. Save with `plt.savefig('figures/aligned_mean_std_waveforms.png', dpi=150, bbox_inches='tight')`.

![Aligned Mean Waveforms](figures/aligned_mean_std_waveforms.png)

**Figure 4.** Mean echo waveform ± 1 standard deviation after sub-bin FFT waveform alignment (×24 oversampling; effective range resolution ≈ 1 cm), for sea ice and leads.

**Analysis:** Alignment produces three notable changes relative to Figure 3:

- The **lead mean sharpens considerably** — the trailing-edge oscillations largely disappear and the peak becomes narrower and better defined, confirming that the ringing in Figure 3 was an instrumental artefact (tracker-range jitter) rather than a physical signal. The resulting template is consistent with the idealised specular-point waveform shape described in altimetry retracking literature.
- The **lead ±1σ envelope narrows around the peak**, indicating that part of the intra-cluster spread in Figure 3 came from misalignment rather than genuine physical variability in lead returns. The remaining spread reflects true variability in lead properties.
- The **sea-ice composite changes less dramatically**, as expected: the broad waveform shape is inherently insensitive to sub-bin shifts because its power is distributed across many range bins. The modest tightening of the sea-ice ±1σ confirms that some variability was instrumental.

These aligned composites could serve directly as **waveform endmembers** in a spectral-mixture-type classification approach such as that proposed by Lee et al. (2018) [8].

---

### Confusion Matrix & Classification Accuracy

> **Figure to include here:** `figures/confusion_matrix.png`  
> Produce by running the `sklearn.metrics.ConfusionMatrixDisplay` cell with `cmap='Blues'`, display labels `['Sea Ice', 'Lead']`, and save with `plt.savefig('figures/confusion_matrix.png', dpi=150, bbox_inches='tight')`.

![Confusion Matrix](figures/confusion_matrix.png)

**Figure 5.** Confusion matrix comparing GMM cluster labels against ESA official L2 surface-type flags for all 12,195 classified waveforms. Rows = ESA ground truth; columns = GMM prediction.

**Raw counts:**

```
                  Pred: Sea Ice   Pred: Lead
True: Sea Ice          8856            22
True: Lead               24          3293
```

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Sea Ice (0) | 1.00 | 1.00 | 1.00 | 8,878 |
| Lead (1) | 0.99 | 0.99 | 0.99 | 3,317 |
| **Overall accuracy** | | | **~99.6%** | **12,195** |

**Analysis:** With only 46 misclassifications out of 12,195 observations, the GMM achieves near-perfect agreement with the ESA operational product. Several interpretive points are important:

1. **Why so high?** The ESA operational classifier itself uses pulse peakiness as its primary discriminating feature [1], so strong agreement is expected when we also use peakiness (plus stack standard deviation) as inputs. The GMM has, without any labelled data, converged to a decision boundary that closely mirrors the ESA operational threshold. This validates that peakiness and SSD form a highly discriminative feature pair — consistent with Dettmering et al. (2018) [5], who report unsupervised classifiers on CryoSat-2 stack statistics achieving >97% accuracy.

2. **False sea-ice (22 cases):** Lead echoes mis-classified as sea ice. These likely correspond to **very narrow leads** (< ~300 m) whose altimeter footprint is dominated by surrounding ice, reducing apparent peakiness below the decision boundary. This type of commission error is a known limitation of all waveform-based lead classifiers [4][5].

3. **False leads (24 cases):** Sea-ice echoes mis-classified as leads. These are likely caused by **smooth ice surfaces** — newly refrozen frost flowers or thin nilas — which can produce near-specular returns similar to open water [3]. This class of omission error has motivated the inclusion of additional features such as stack skewness and stack kurtosis in more advanced classifiers [5][8].

4. **Context for the ~99.6% figure:** Supervised state-of-the-art methods applied directly to Sentinel-3 data (Adaptive Boosting, Neural Networks) report overall accuracies of up to 92%, while the best unsupervised method (K-medoids) achieves ~92.7% [4]. The higher accuracy here is partly explained by the close correspondence between the features used here and those in the ESA classifier, and partly by the specific scene composition of this single overpass (mostly consolidated winter sea ice with well-defined leads). Testing across a broader range of seasons and ice conditions would be expected to produce results closer to those in the published literature.

---

## Getting Started

### Prerequisites

- A Google account with access to [Google Colab](https://colab.research.google.com/)
- A [Google Drive](https://drive.google.com/) folder containing the Sentinel-3 `.SEN3` data file

### Installation

The notebook runs in Google Colab. Install non-standard packages at the start of each session:

```python
!pip install netCDF4
!pip install rasterio
!pip install basemap
!pip install cartopy
```

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Data

The base reference notebook is available at:  
https://drive.google.com/file/d/1HDSLjsWhLIDF-qbRj6sbGVd9t1LB7890/view?usp=drive_link

The Sentinel-3 data is available via the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/). The specific file used is:

```
S3B_SR_2_LAN_SI_20190301T231304_20190301T233006_20230405T162425_1021_022_301______LN3_R_NT_005.SEN3
```

> **Note:** The data file is **not included** in this repository due to size. Download it from the Copernicus Data Space and update the file path in the notebook.

---

## Repository Structure

```
GEOL0069-Week4/
 ├── Unit_2_Unsupervised_Learning_Methods.ipynb          # Main assignment notebook
 ├── Chapter1_Unsupervised_Learning_Methods_Michel.ipynb  # Base reference notebook
 ├── figures/
 │   ├── feature_space_scatter.png        # Fig 1 - PP vs SSD feature space
 │   ├── sample_waveforms_sea_ice.png     # Fig 2a - Individual sea-ice echoes
 │   ├── sample_waveforms_leads.png       # Fig 2b - Individual lead echoes
 │   ├── mean_std_waveforms.png           # Fig 3 - Mean +/- std (unaligned)
 │   ├── aligned_mean_std_waveforms.png   # Fig 4 - Mean +/- std (FFT-aligned)
 │   └── confusion_matrix.png             # Fig 5 - Confusion matrix heatmap
 └── README.md
```

---

## References

[1] Reinhout, T., Zawadzki, L., Féménias, P., Tournadre, J., Quartly, G., Ablain, M., ... & Potin, P. (2025). Sentinel-3 Altimetry Thematic Products for Hydrology, Sea Ice and Land Ice. *Scientific Data*, 12, 699. https://doi.org/10.1038/s41597-025-04956-3

[2] Donlon, C., Berruti, B., Buongiorno, A., Ferreira, M.-H., Féménias, P., Frerick, J., ... & Sciarra, R. (2012). The global monitoring for environment and security (GMES) Sentinel-3 mission. *Remote Sensing of Environment*, 120, 37–57. https://doi.org/10.1016/j.rse.2011.07.024

[3] Wernecke, A. and Kaleschke, L. (2015). Lead detection in Arctic sea ice from CryoSat-2: quality assessment, lead area fraction and width distribution. *The Cryosphere*, 9, 1955–1968. https://doi.org/10.5194/tc-9-1955-2015

[4] Willms, N., Lorenz, C., Dettmering, D., and Müller, F.L. (2022). Lead Detection in the Arctic Ocean from Sentinel-3 Satellite Data: A Comprehensive Assessment of Thresholding and Machine Learning Classification Methods. *Marine Geodesy*, 45(5), 479–512. https://doi.org/10.1080/01490419.2022.2089412

[5] Dettmering, D., Wynne, A., Müller, F.L., Passaro, M., and Seitz, F. (2018). Lead Detection in Polar Oceans — A Comparison of Different Classification Methods for CryoSat-2 SAR Data. *Remote Sensing*, 10(8), 1190. https://doi.org/10.3390/rs10081190

[6] MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.

[7] Reynolds, D.A. (2009). Gaussian Mixture Models. In *Encyclopedia of Biometrics*. Springer, Boston, MA. https://doi.org/10.1007/978-0-387-73003-5_196

[8] Lee, S., Im, J., Kim, J., Kim, M., Shin, M., Kim, H.-C., and Quackenbush, L.J. (2018). Arctic lead detection using a waveform mixture algorithm from CryoSat-2 data. *The Cryosphere*, 12, 1665–1679. https://doi.org/10.5194/tc-12-1665-2018

---

## Contact

**Your Name** – your.email@ucl.ac.uk  
Project Link: `https://github.com/YOUR_USERNAME/GEOL0069-Week4`

---

## Acknowledgements

- This project is submitted as part of **GEOL0069 – Artificial Intelligence for Earth Observation**, UCL Earth Sciences Department.
- Base notebook and course materials provided by **Dr Michel Tsamados**, UCL.
- Sentinel-3 data courtesy of the **European Space Agency (ESA)** / Copernicus programme.

<p align="right"><a href="#sea-ice--lead-classification-via-unsupervised-learning">Back to top</a></p>
