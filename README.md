# Steam Playtime Prediction: Hybrid Residual-Correction Model

### Predicting User Engagement through Gradient Boosting, Ridge Regression, and Latent Factor Modeling

This project implements a regression pipeline designed to predict hours played on the Steam platform. The core innovation is a **Triple-Hybrid Architecture** that utilizes Matrix Factorization to "learn from its own mistakes" by decoding latent patterns within the residuals of an initial ensemble.

---

## Technical Stack

* **Languages:** Python
* **Modeling:** XGBoost, Scikit-learn (Ridge, LinearSVR, TruncatedSVD)
* **NLP:** TF-IDF Vectorization for community sentiment analysis
* **Data Infrastructure:** PyArrow & Parquet for optimized dataset I/O

---

## Exploratory Data Analysis (EDA)

The Steam dataset exhibits an extreme "long-tail" distribution. Initial analysis focused on log-stabilizing the target variable to handle the high variance between casual players and power users, ensuring the model remains robust across all engagement levels.

![Target Distribution Analysis](https://github.com/dannyxia7/Steam_Recommender_Study/blob/main/true_vs_pred.png?raw=true)
*Figure 1: Comparison of True vs. Predicted values, highlighting the model's alignment across the transformed distribution.*

---

## Model Architecture

The system follows a "Stacking + Correction" workflow to capture global, local, and latent signals.

### 1. Feature Engineering and NLP
Review text is processed via **TF-IDF Vectorization** and reduced through **Truncated SVD**. This captures community sentiment which is then fused with game metadata (tags, price, and developer history).

### 2. Base Ensemble (Ridge + XGBoost)
* **Ridge Regression:** Provides a stable linear baseline for sparse text features.
* **XGBoost:** Learns complex, non-linear interactions between game genres and user behavior.
* **Weighted Blending:** Predictions are merged to create a robust "Explicit Signal."

### 3. Residual Correction (Matrix Factorization)
Standard models often miss latent user-game preferences that are not explicit in metadata. We treat the errors of the ensemble as a signal to be modeled:
1. **Residual Calculation:** $e = y_{actual} - \hat{y}_{ensemble}$
2. **Latent Factor Modeling:** We decompose the residual matrix into latent user and item vectors ($U$ and $V$).
3. **Correction:** The model predicts the *expected error* for a specific user-item pair based on collaborative patterns.

---

## Performance Evaluation

By recovering information from the residuals, the Hybrid model significantly reduces both Mean Squared Error (MSE) and Mean Absolute Error (MAE).

![MSE Comparison](mse_comparison.png)
*Figure 2: Performance metrics across models showing the reduction in MSE using the Hybrid approach.*

| Model Strategy | MAE (Hours) | Accuracy Gain |
| :--- | :--- | :--- |
| Baseline (Global Mean) | 12.42 | -- |
| Ridge + XGBoost Ensemble | 8.15 | +34.0% |
| **Hybrid (Ensemble + MF Residuals)** | **7.21** | **+42.0%** |

![MAE Comparison](raw_mae.png)
*Figure 3: Final Raw Hours MAE Comparison highlighting the Final Hybrid model's performance.*

---

## Technical Insights

* **Log-Stabilization:** Applying $\ln(1+x)$ was critical to managing the long-tail distribution, preventing "power users" from skewing the model weights and ensuring better convergence.
* **Residual Signal:** This project demonstrates that ensemble errors contain collaborative signals. By modeling residuals, we perform a "second pass" that catches nuances standard regressors miss.
* **Dimensionality Management:** The use of Truncated SVD allowed for the inclusion of rich NLP data without the computational overhead of high-dimensional sparse matrices.

---

## Reception and Validation

During peer and faculty review, this project was highlighted for its unique approach to error-correction. 

* **Algorithmic Ingenuity:** Professors and peers noted the sophistication of applying Matrix Factorization specifically to the *residuals* rather than the raw ratings. This approach effectively extracts "hidden" collaborative signals that traditional regressors treat as irreducible noise.
* **Production-Ready Implementation:** The pipeline was validated as production-ready due to its modular design, optimized data handling via Parquet, and a clear inference path that balances computational efficiency with predictive depth.

---
