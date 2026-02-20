# Sea Ice & Lead Classification via Unsupervised Learning
### GEOL0069 – Artificial Intelligence for Earth Observation | Week 4 Assignment

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zC4AWfp0Af7_LH2F_ZHQNeYrFT6Oe19A)

---

<details>
<summary><b>Table of Contents</b></summary>

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background & Scientific Context](#2-background)
   - [Arctic Leads](#21-arctic-leads)
   - [SAR Radar Altimeter](#22-sar-radar-altimeter)
   - [Clustering Algorithms](#23-clustering-algorithms)
3. [Methods](#3-methods)
   - [Data & Preprocessing](#31-data--preprocessing)
   - [Gaussian Mixture Models (GMM)](#32-gaussian-mixture-models-gmm)
4. [Discussion and Results](#4-discussion-and-results)
   - [Feature Space Analysis](#41-feature-space-analysis)
   - [Echo Waveform Analysis](#42-echo-waveform-analysis)
   - [Results](#43-results)
5. [Getting Started](#getting-started)
6. [Repository Structure](#repository-structure)
7. [References](#references)
8. [Contact](#contact)
9. [Acknowledgements](#acknowledgements)

</details>

## 1. Project Overview

This assignment focuses on evaluation of automated methods for discrimination of sea-ice and leads. A Gaussian Mixture Model (GMM) is trained on waveform features derived from unlabelled Sentinel-3 SAR altimetry data, and their classification performances validated against ESA surface-type flags.

The notebook `GEOL0069_Week4_Assignment.ipynb` builds directly on `Chapter1_Unsupervised_Learning_Methods_Michel.ipynb` and involves:
- Mean waveform shapes and standard deviation envelopes for each class
- Feature space visualisation (pulse peakiness vs stack standard deviation)
- Comparison of cross-correlation and orbit geometry-based waveform alignment, followed by revisualisation of mean waveform shapes and standard deviations 
- Quantitative evaluation against ESA official L2 surface-type flags using a confusion matrix and classification report

The GMM-based classifier achieves per-class F1-scores of 1.00 and 0.99 for sea ice and lead respectively, demonstrating that a two component mixture model applied to just two features is sufficient to cleanly separate these surface types. 

---

## 2. Background

### 2.1 Arctic Leads

Sea ice leads form when wind and ocean currents exert mechanical stress on the ice, causing it to pull apart or slide against itself until it fractures into long, narrow channels of open water. These leads play a significant role in climate regulation by facilitating heat and gas exchange between the ocean and atmosphere - accounting for approximately 70% of total upward heat transfer from the ocean in winter and accelerating melting by absorbing solar radiation in the summer - while also serving both as points of access to the surface (and air) for marine animals like seals and natural corridors for human navigation.[1] As such, identifying and mapping these leads is of importance. For this, Synthetic Aperture Radar (SAR) data, in combination with unsupervised learning techniques, can be effective.[3][4]

### 2.1 SAR Radar Altimeter

SAR (SAR) measures the backscatter of microwave pulses to detect surface features. Unlike optical satellite data, the approach is unaffected by cloud cover or months of darkness, making it invaluable for day- and year-round monitoring.[2] A key application of this approach is in SAR altimetry (e.g. via the Sentinel-3 SAR Radar Altimeter (SRAL) - which involves inferring surface elevation from the time taken from emission of the pulse and detection of the returning signal (called a "waveform" or "echo"). This allows much higher resolution measurements compared to conventional methods, which is essential for detection of leads, which may only be tens to hundreds of metres wide:
- Conventional altimeters transmit a single broad radar pulse that illuminates a large patch of surface - roughly 20km wide - simultaneously. The return signal is therefore an average over that entire area, which is problematic in contexts such as sea ice where the surface changes character over much shorter distances.
- SAR altimeters address this by recording many pulses in quick succession as the satellite moves along its orbit, then combining them (using coherent multi-look processing using Doppler techniques) to isolate the return from a much smaller strip of ground - around 300m - directly beneath the satellite.[3][4]

<p align="center">
  <img src="/images/Sentinel_3_SRAL_Diagram.png" width="50%" alt="Sentinel 3 SRAL Diagram>
  <br>
  <em>Figure 1: Diagram of SRAL nadir track (the ground track directly beneath the satellite), as well as the ground tracks of the other Sentinel-3 instruments.[3].</em>
</p>

The returned waveform encodes information about both:
- Surface elevation, from the timing of the waveform's *leading edge* (the point where the returned signal strength first rises sharply i.e. the moment the pulse reaches the surface). 
- Surface texture, from the shape of the waveform (a smooth surface e.g. a lead reflects the pulse back cleanly, producing a sharp, intense return; a rough surface e.g. sea ice scatters the pulse in many directions so the energy arrives back at the satellite over a longer time period, producing a weaker, broader return) [5]

Such information can be extracted via two key features:[5][6]

| Feature | Physical Meaning |
|---|---|
| **Pulse Peakiness (PP)** | Ratio of peak power to mean power across the waveform. Leads produce very high peakiness (sharp specular returns); sea ice produces low peakiness (diffuse returns). |
| **Stack Standard Deviation (SSD)** | Spread of power across different viewing angles as the satellite passes overhead. Leads produce a narrow angular response (low SSD); sea ice produces a broad response (high SSD). | 

An unsupervised learning algorithm can exploit differences in these features to distinguish between sea ice and leads.[7]

### 2.3 Clustering algorithms

Unsupervised learning is used to elucidate underlying patterns in data without the need for labelled training data. There are two main types: clustering (grouping data points based on similarity) and dimensionality reduction (compressing data to lower the dimension space). The principles behind two of the main clustering approaches, K-means and Gaussian Mixture Models (GMMs), is briefly outlined below. I provide more detail on the mathematical formulation of these techniques in [this](/Clustering_Algorithms_Summary__K_means_and_Gaussian_Mixture_Models.pdf) set of notes.

#### 2.3.1 K-means
K-Means partitions the feature space into *k* clusters by iteratively assigning each point to its nearest centroid and recomputing the position of the centroids until there is minimal or no further movement (convergence). It is a centroid-based clustering algorithm (i.e. that groups data around central points) that works by minimising the Within-Cluster Sum of Squares (WCCS), also known as distortion:

$$J(c, \mu) = \sum_{i=1}^{n} \sum_{j=1}^{k} \mathbb{1}\{c_i = j\} \|x_i - \mu_j\|^2$$

**Where:**
* $\mathbb{1}\{c_i = j\}$ is an **indicator function** (1 if point $i$ is assigned to cluster $j$, 0 otherwise).
* $\|x_i - \mu_j\|^2$ is the **squared $L_2$ norm** (Euclidean distance).

The objective function $J$ depends on two sets of variables: the discrete assignments $c$ and the continuous centroids $\mu$. Since we cannot optimize both simultaneously, we use an **alternating optimization** approach:

1.  *Assignment Step:* Fix $\mu$ and minimize $J$ with respect to $c$ (assign each point to its nearest centroid).
2.  *Update Step:* Fix $c$ and minimize $J$ with respect to $\mu$ (move each centroid to the mean of its assigned points).

This process repeats until the assignments no longer change or a maximum number of iterations is reached.[8]

Advantages:
- Requires no prior knowledge of cluster shape
- Scales efficiently to large datasets

Drawbacks:
- *Spherical Assumption:* Because K-means uses the L2 norm to assign points to the nearest centroid, it implicitly assumes that clusters are spherical and have similar diameters. As such, it can be ineffective on datasets featuring non-spherical geometries—such as elongated, elliptical, or manifold shapes—as well as clusters with significantly varying densities or very different sizes.
- *Hard-clustering:* In K-means, a point belongs entirely to one cluster, even if it is exactly on the border between the two; as such, it forces a binary decision on ambiguous data ponts.
- *Non-convexity:* The loss surface in K-means in non-convex and therefore prone to lcoal optima, so the final result depends strongly on the initial random mean positions (K-means++ can be used to circumvent this issue by choosing initial centroids that are far apart from each other)
- k (number of clusters) must be chosen manually [9]

An example implementation is shown below.

```python
# Import KMeans algorithm from scikit-learn library, alongside matplotlib and numpy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Generate 2D array of 100 rows and 2 columns of random data
X = np.random.rand(100, 2)

kmeans = KMeans(n_clusters=4) #initialise K-means
kmeans.fit(X) #iterative movement of cluster centres
y_kmeans = kmeans.predict(X) #assign labels to each data point based on which centroid it is closer to

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
```
<p align="center">
  <img src="/images/KMeansIllustration.png" width="50%" alt="Sentinel 3 SRAL Diagram">
</p>
---

#### 2.3.2 Gaussian Mixture Models

A GMM models the data as a weighted sum of *K* multivariate Gaussian distributions, each with its own mean **μ** and covariance **Σ** [10]. Parameters are estimated via the **Expectation-Maximisation (EM)** algorithm:

- **E-step:** compute the posterior probability that each waveform belongs to each component.
- **M-step:** update **μ**, **Σ**, and mixing weights to maximise the data log-likelihood.[11]

These have advantages over k-means due to:
- Flexibility in cluster shapes: GMM allows each cluster to have a different covariance structure
- Probabilistic assignment: GMMs outputs a *soft* classification (probability of class membership), which represents ambiguous data points more completely[12]

Again, below is an example implementation.

```python
#Import GMM algorithm from scikit-learn
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Same generation of random sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3) #initialise GMM (where N_components means number of Gaussian distributions)
gmm.fit(X) #begin Expectation-Maximisation
y_gmm = gmm.predict(X) #assign each point to Gaussian Distribution it most likely belongs to

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.show()

```
<p align="center">
  <img src="/images/GMMIllustration.png" width="50%" alt="Sentinel 3 SRAL Diagram">
</p>
---

## 3. Methods

## 3.1 Data & Preprocessing

To transform the data into meaningful information for the classification model, Pulse Peakiness and Stack Standard Deviation are extracted from raw Sentinel-3 data:
```
S3B_SR_2_LAN_SI_20190301T231304_20190301T233006_20230405T162425_1021_022_301______LN3_R_NT_005.SEN3
```
The ESA official surface-type flag (`surf_class_20_ku`) is used as ground truth: flag = 1 for sea ice, flag = 2 for leads. All observations with other flag values or NaN features are excluded, leaving 12,195 valid waveforms for analysis.

---

### 3.2 Gaussian Mixture Models (GMM)

Two-component GMM is selected over K-means for clustering here, to provide robustness to potentially non-spherical clusters (Liu et al., 2025), with random state set to zero to allow for reproducibility.[7] The model is fitted to the cleaned waveforms and the predicted ratio of leads to sea-ice checked for physical plausibility, with significantly more sea ice expected.[5]

**Cluster counts from GMM:**

| Cluster | Count | Assigned Class |
|---------|-------|----------------|
| 0 | 8,880 | Sea Ice |
| 1 | 3,315 | Lead |

---

## 4 Discussion and Results

### 4.1 Feature Space Analysis

<p align="center">
  <img src="/images/ClusteringFeatureSpace1.png" width="70%" alt="Sentinel 3 SRAL Diagram">
  <br>
  <em>Figure 2: Clustering feature spaces for the Gaussian Mixture Model.</em>
</p>

We first analyse the feature space (Figure 2) to evaluate the model's success in separating the two classes, plotting $\sigma_0$ (dB), the backscatter coefficient (a measure of how strongly the surface reflects the radar pulse back towards the satellite) against both PP and SSD, as well as PP against SSD to assess how well the two clusters separate in the classification feature space itself.[2] We observe strong separation following intuitive patterns: the sea-ice cluster cluster occupies the low-peakiness, weak-backscatter, high-SSD regions, reflecting the more diffuse, multi-angular return expected from a rough ice surface compared to the smooth leads. The elongated, non-spherical cluster shapes validate the choice of GMM over K-means.

### 4.2 Echo Waveform Analysis

#### 4.2.1 Waveform Alignment

We compute the mean waveform across all echoes in that cluster and the standard deviation at each range bin (a fixed slice of time after the radar pulse was transmitted), visualised in Figure 3.

<p align="center">
  <img src="/images/UnalignedMeanSd.png" width="60%" alt="Sentinel 3 SRAL Diagram">
  <br>
  <em>Figure 3: Mean and standard deviation of unaligned sea-ice and lead waveforms.</em>
</p>

It is noted, however, that when many waveforms are recorded along a satellite track, each one is centred slightly differently within the 256-bin window, leading to the lack of sharpness in the peaks observed in Figure 3. As such, for fairer comparison, initial alignment of the waveforms is required. For this, two methods are investigated:

- *Cross-correlation:* Each waveform is aligned to the first waveform in the cluster using cross-correlation (which finds the shift that maximises the similarity between two signals)[13]
- *Physics-based alignment:* Alignment using the known orbit geometry, developed at the Alfred Wegner Institute [14]

The results of alignment using cross-correlation and orbit geometry are displayed in Figure 4 and 5, respectively.

<p align="center">
  <img src="/images/CrossCorrelationAlignment.png" width="70%" alt="Sentinel 3 SRAL Diagram">
  <br>
  <em>Figure 4: Effect of waveform alignment using cross-correlation.</em>
</p>

We observe that cross-correlation performs poorly in aligning the waveforms on this data: the waveform shape is pulled further from the expected sharp appearance. As cross-correlation works by finding the shift that maximises the similarity between two signals, it works well when waveforms are of similar shape. It is possible that the significant noise present in sea-ice and lead waveforms lead to the lack of a dominant feature for cross-correlation to use, instead basing shifts on the noise.

<p align="center">
  <img src="/images/PhysicsBasedAlignment.png" width="70%" alt="Sentinel 3 SRAL Diagram">
  <br>
  <em>Figure 5: Effect of physics-based alignment using orbit geometry.</em>
</p>

By contrast, the physics-based alignment (of the normalised waveforms in this case) produces a meaningful improvement for both surface classes.

*   Peak height: The lead mean peak height increases from 0.31 to 0.77, demonstrating that averaging the peaks without alignment had resulted in flattening of the peak due to differing bin positions of the peak along the x-axis. Alignment appears to allow recovery of a peak closer to the true shape of an individual echo (as observed in the first plot below). The sea ice peak also increases, from 0.47 to 0.74. As sea-ice waveforms are natively broader, the peak is less sensitive to small bin shifts; however, it is evident that the lack of alignment still caused suppression of the peak
*   Standard deviation: The lead mean standard deviation decreases from 0.026 to 0.018, demonstrating that a significant proportion of the spread in the unaligned calculation was due to instrumental rather than physical variability. The sea-ice mean standard deviation also decreases, but less sharply, suggesting that the instrumental variation contributed less to the standard deviation than in the case of leads.

This allows us to obtain a view of the distribution of leads and sea-ice caused solely by physical - rather than instrumental - variation.

*Table 2. Waveform Comparison Metrics*

| Metric | Category | Unaligned | Aligned |
| :--- | :--- | :---: | :---: |
| **Peak Height** (0-1 scale) | Sea Ice | 0.4653 | **0.7393** |
| | Lead | 0.3064 | **0.7732** |
| **Mean Std. Dev.** | Sea Ice | 0.0811 | **0.0749** |
| | Lead | 0.0264 | **0.0180** |

#### 4.2.2 Waveform Shape

The shapes of the two waveform types follow expected patterns. As seen in the unnormalised Figure 3, the lead mean has substantially higher peak power than the sea-ice mean, reflecting the documented difference in backscatter coefficient (σ⁰) between specular leads and diffuse sea ice.[5][6] From Figure 6, we observe that the sea-ice return rises gradually from early bins: the greater roughness of the sea ice is expected to lead to greater scattering and therefore less concentrated pulse return per unit time. The fact that the "leading edge" occurs earlier for sea-ice than leads also indicates that the sea-ice elevation is higher (the pulse hits it earlier), which is also expected. By contrast, the lead return is much sharper, in line with the smoother surface of the leads (which causes almost all energy to be reflected back at a specific, well-defined time from a single point directly below the satellite).

The tighter standard deviation envelope for sea-ice suggests that, along the satellite's track, sea-ice echoes are fairly homogeneous in terms of roughness properties. By contrast, the lead standard deviation envelope is markedly broader, which may correspond to the significant variation in lead width (e.g. for thin leads, the return signal may be diluted by surrounding ice), state (e.g. whether the lead is partially frozen), and shape. In general, the classification aligns with physical expectations and past analyses in the literature.[5][6]

### 4.3 Results

Finally, we validate our predictions against ESA official flag data. As the data is imbalanced (sea ice is more prevalent than leads), accuracy is an unreliable metric. In place, a confusion matrix is calculated (Figure 7), from which per-class precision, recall and F1-scores (the harmonic mean of precision and recall) are derived (Table 3). Out of the 12,195 observations, the confusion matrix shows only 24 misclassifications of lead as sea ice, and 22 of sea ice as lead, resulting in an F1-score of 1.00 for sea ice and 0.99 for lead. 

<p align="center">
  <img src="/images/ConfusionMatrix.png" width="50%" alt="Sentinel 3 SRAL Diagram">
  <br>
  <em>Figure 7: onfusion matrix comparing GMM cluster labels against ESA official L2 surface-type flags for all 12,195 classified waveforms. Rows = ESA ground truth; columns = GMM prediction..</em>
</p>

*Table 3. Classification Report*

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Sea Ice (0) | 1.00 | 1.00 | 1.00 | 8,878 |
| Lead (1) | 0.99 | 0.99 | 0.99 | 3,317 |

The excellent performance supports the use of both (i) PP and SSD as effective discriminatory features for sea ice and leads and (ii) clustering algorithms as an effective means of harnessing these features, in line with previous findings.[7][15] Misclassification of lead echoes as sea ice may correspond to very narrow leads, whose altimeter footprint is dominated by surrounding ice, thus reducing apparent peakiness below the clustering decision boundary; further error analysis confirming this would be a useful next step. Similarly, sea-ice echoes misclassified as leads may correspond to smooth ice surfaces - perhaps those that have been newly refrozen - which may produce near-specular returns similar to open water.[5] Testing across a range of scenes, both varying in location and season, would also be valuable to analyse and address model robustness and transferability.

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

The base reference notebook, developed by Dr Michel Tsamados is available at:  
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

[1] Marcq, S. and Weiss, J. (2012) ‘Influence of sea ice lead-width distribution on turbulent heat transfer between the ocean and the atmosphere’, The Cryosphere, 6(1), pp. 143–156. doi: 10.5194/tc-6-143-2012.

[2] Colin, E. (2024) ‘What are the physical quantities in a SAR image? When and Why to Calibrate in a Training Database?’, Medium, 24 August. Available at: https://elisecolin.medium.com/what-are-the-physical-quantities-in-a-sar-image-c788a8265abd (Accessed: 16 February 2026).

[3] European Space Agency (n.d.) S3 Altimetry Instruments. Available at: https://sentiwiki.copernicus.eu/web/s3-altimetry-instruments (Accessed: 20 February 2026).

[4] Donlon, C., Berruti, B., Buongiorno, A., Ferreira, M.-H., Féménias, P., Frerick, J., Sciarra, R. (2012). The global monitoring for environment and security (GMES) Sentinel-3 mission. *Remote Sensing of Environment*, 120, 37–57. https://doi.org/10.1016/j.rse.2011.07.024

[5] Wernecke, A. and Kaleschke, L. (2015). Lead detection in Arctic sea ice from CryoSat-2: quality assessment, lead area fraction and width distribution. *The Cryosphere*, 9, 1955–1968. https://doi.org/10.5194/tc-9-1955-2015

[6] Bij de Vaate, I., Martin, E., Slobbe, D. C., Naeije, M., & Verlaan, M. (2022). Lead Detection in the Arctic Ocean from
Sentinel-3 Satellite Data: A Comprehensive Assessment of Thresholding and Machine Learning Classification Methods.
Marine Geodesy, 45(5), 462-495. https://doi.org/10.1080/01490419.2022.2089412

[7] Liu, W., Tsamados, M., Petty, A., Jin, T., Chen, W. and Stroeve, J. (2025) ‘Enhanced sea ice classification for ICESat-2 using combined unsupervised and supervised machine learning’, Remote Sensing of Environment, 318, 114607. doi: 10.1016/j.rse.2025.114607.

[8] Nazarpour, K. (2022) Machine Learning - Lecture 19: K-means Clustering. [Lecture slides] INFR10086: Machine Learning, University of Edinburgh. Available at: https://homepages.inf.ed.ac.uk/htang2/mlg2022/mlg20-kmeans-slides.pdf (Accessed: 20 February 2026).

[9] Google (2025) ‘Advantages and disadvantages of k-means’, Google for Developers. Available at: https://developers.google.com/machine-learning/clustering/kmeans/advantages-disadvantages (Accessed: 17 February 2026).

[10] Mackey, L. (2014) ‘Lecture 2’. [Lecture notes] STATS 306B: Unsupervised Learning. Stanford University. Scribed by Qian, J. and Wang, M. Available at: https://web.stanford.edu/~lmackey/stats306b/doc/stats306b-spring14-lecture2_scribed.pdf (Accessed: 20 February 2026).

[11] GeeksforGeeks (2025) ‘Gaussian Mixture Model’, GeeksforGeeks. Available at: https://www.geeksforgeeks.org/machine-learning/gaussian-mixture-model/ (Accessed: 16 February 2026).

[12] Yadav, A. (2024) ‘K Means Clustering vs Gaussian Mixture’, Medium, 13 July. Available at: https://medium.com/@amit25173/k-means-clustering-vs-gaussian-mixture-bec129fbe844 (Accessed: 17 February 2026).

[13] MathWorks (2026) Align Signals Using Cross-Correlation. Available at: https://www.mathworks.com/help/signal/ug/align-signals-using-cross-correlation.html (Accessed: 16 February 2026).

[14] Alfred Wegener Institute (2025) aligned-waveform-generator. Available at: https://gitlab.awi.de/siteo/aligned-waveform-generator (Accessed: 27 February 2026).

[15] Dettmering D, Wynne A, Müller FL, Passaro M, Seitz F. Lead Detection in Polar Oceans—A Comparison of Different Classification Methods for Cryosat-2 SAR Data. Remote Sensing. 2018; 10(8):1190. https://doi.org/10.3390/rs10081190

---

## Contact

**Harry Fyjis-Walker** – harryfyjiswalker@gmail.com 
Project Link: `https://github.com/YOUR_USERNAME/GEOL0069-Week4`

---

## Acknowledgements

- This project is submitted as part of an assignment for **GEOL0069 – Artificial Intelligence for Earth Observation**, UCL Earth Sciences Department.
- Base notebook and course materials provided by **Dr Michel Tsamados**, UCL.
- Sentinel-3 data courtesy of the **European Space Agency (ESA)** / Copernicus programme.

<p align="right"><a href="#sea-ice--lead-classification-via-unsupervised-learning">Back to top</a></p>


