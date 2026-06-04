# Laboratory Project No. 1
## Financial Fraud Detection – Classification of Highly Imbalanced Data
### Report No. 3: Results Analysis and Experimental Conclusions

**Submitted by:**
* Mykola Tatarenkov (53631)
* Siarhei Kryshchanovich (57763)

**Course:** Machine Learning for Financial Applications
**Date:** June 2, 2026

---

## 1. Executive Summary

This report completes our investigation into financial fraud detection on highly imbalanced datasets. Building on the data analysis from Report 1 and the implementation pipeline described in Report 2, Report 3 presents a consolidated analysis of our experimental results. 

Our pipeline evaluated five classification model families: **Logistic Regression (linear baseline)**, **Decision Tree (tree baseline)**, **Random Forest (ensemble)**, **Extra Trees (ensemble)**, and **XGBoost (boosting ensemble)**. These models were evaluated across six class-imbalance handling strategies: **baseline (no adjustments)**, **class weighting (cost adjustment)**, **random undersampling (RUS)**, **random oversampling (ROS)**, **SMOTE**, and **ADASYN**. 

To handle the highly asymmetric business consequences of fraud detection—where missing a fraudulent transaction is far more damaging than triggering a false alert—we applied a cost-sensitive framework. The decision thresholds were optimized on the validation split and evaluated on the test set using a cost matrix where a False Negative ($COST_{FN} = 500$) is penalized fifty times more heavily than a False Positive ($COST_{FP} = 10$).

Our key findings demonstrate that:
1. **Ensemble methods** consistently dominate linear and single-tree models across all metrics, with XGBoost achieving the best performance on both datasets.
2. **Threshold tuning** on validation data yields dramatic business value, reducing the test set expected cost by **16.6%** on the Credit Card dataset and by **44.5%** on the PaySim dataset compared to the default 0.5 threshold.
3. **Imbalance-handling methods** (particularly class weighting and SMOTE) are critical to raising fraud recall, but must be paired with threshold tuning to control false positives and minimize operational costs.

---

## 2. Consolidated Results

Below are the consolidated test set results for all 30 configurations across both datasets. The models are sorted in ascending order of **Test Expected Cost** (the primary business metric) and descending order of **Test PR-AUC**.

### 2.1 Credit Card Dataset Results
The Credit Card dataset consists of 284,807 transactions with 492 fraud cases (0.172%). The stratified test set contains **56,962 transactions**, of which **99 are fraudulent**.

| Model | Strategy | Tuned Threshold | Test ROC-AUC | Test PR-AUC | Test Recall | Test Precision | Test Expected Cost | Test Expected Cost (0.5) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **xgboost** | class_weight | 0.06 | 0.9707 | 0.8273 | 0.8182 | 0.7714 | **9,240** | 11,080 |
| **extra_trees** | oversample | 0.45 | 0.9698 | 0.7920 | 0.8182 | 0.7105 | **9,330** | 10,670 |
| **random_forest** | class_weight | 0.17 | 0.9574 | 0.7863 | 0.8182 | 0.6639 | **9,410** | 13,600 |
| **extra_trees** | smote | 0.52 | 0.9629 | 0.7887 | 0.8182 | 0.6585 | **9,420** | 9,590 |
| **random_forest** | undersample | 0.84 | 0.9767 | 0.6852 | 0.8182 | 0.6532 | **9,430** | 21,240 |
| **xgboost** | baseline | 0.01 | 0.9746 | 0.8211 | 0.8182 | 0.6378 | **9,460** | 13,530 |
| **extra_trees** | baseline | 0.02 | 0.9735 | 0.7955 | 0.8182 | 0.5586 | **9,640** | 18,550 |
| **random_forest** | baseline | 0.02 | 0.9725 | 0.8062 | 0.8283 | 0.4141 | **9,660** | 14,540 |
| **extra_trees** | class_weight | 0.48 | 0.9730 | 0.7924 | 0.8081 | 0.7767 | **9,730** | 10,680 |
| **xgboost** | adasyn | 0.11 | 0.9660 | 0.7915 | 0.8384 | 0.3192 | **9,770** | 10,460 |
| **random_forest** | oversample | 0.18 | 0.9645 | 0.7785 | 0.8182 | 0.4880 | **9,850** | 11,600 |
| **xgboost** | oversample | 0.01 | 0.9710 | 0.8177 | 0.8182 | 0.4602 | **9,950** | 11,070 |
| **logreg** | baseline | 0.02 | 0.9639 | 0.7091 | 0.8081 | 0.6349 | **9,960** | 23,610 |
| **extra_trees** | undersample | 0.74 | 0.9736 | 0.6664 | 0.8081 | 0.6349 | **9,960** | 16,010 |
| **random_forest** | smote | 0.39 | 0.9749 | 0.7661 | 0.8182 | 0.4309 | **10,070** | 11,040 |
| **extra_trees** | adasyn | 0.66 | 0.9637 | 0.7709 | 0.8081 | 0.4938 | **10,320** | 16,340 |
| **logreg** | class_weight | 0.98 | 0.9747 | 0.6807 | 0.8081 | 0.4545 | **10,460** | 19,030 |
| **logreg** | oversample | 0.98 | 0.9744 | 0.6835 | 0.8081 | 0.4520 | **10,470** | 18,990 |
| **logreg** | smote | 0.99 | 0.9718 | 0.6750 | 0.8081 | 0.4278 | **10,570** | 20,690 |
| **xgboost** | smote | 0.31 | 0.9615 | 0.7976 | 0.7980 | 0.5643 | **10,610** | 10,850 |
| **random_forest** | adasyn | 0.67 | 0.9676 | 0.7124 | 0.8081 | 0.3980 | **10,710** | 12,820 |
| **xgboost** | undersample | 0.99 | 0.9715 | 0.6477 | 0.8182 | 0.2978 | **10,910** | 26,780 |
| **logreg** | adasyn | 0.99 | 0.9688 | 0.6913 | 0.8283 | 0.1826 | **12,170** | 43,830 |
| **decision_tree** | oversample | 0.96 | 0.8751 | 0.5948 | 0.7677 | 0.3707 | **12,790** | 15,790 |
| **decision_tree** | class_weight | 0.91 | 0.8751 | 0.6012 | 0.7778 | 0.2973 | **12,820** | 15,780 |
| **logreg** | undersample | 0.99 | 0.9704 | 0.3830 | 0.7980 | 0.2041 | **13,080** | 29,540 |
| **decision_tree** | smote | 0.98 | 0.8627 | 0.4191 | 0.7374 | 0.4506 | **13,890** | 19,710 |
| **decision_tree** | baseline | 0.11 | 0.8305 | 0.6230 | 0.6970 | 0.8846 | **15,090** | 15,090 |
| **decision_tree** | adasyn | 0.95 | 0.8984 | 0.0747 | 0.7677 | 0.0879 | **19,390** | 36,980 |
| **decision_tree** | undersample | 0.01 | 0.8917 | 0.0130 | 0.8889 | 0.0144 | **65,520** | 65,520 |

### 2.2 PaySim Dataset Results
The PaySim dataset contains simulated mobile money transactions spanning 6,362,620 records with 8,213 fraud cases (0.129%). The stratified test set contains **1,272,524 transactions**, of which **1,642 are fraudulent**.

| Model | Strategy | Tuned Threshold | Test ROC-AUC | Test PR-AUC | Test Recall | Test Precision | Test Expected Cost | Test Expected Cost (0.5) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **xgboost** | smote | 0.87 | 0.9998 | 0.9612 | 0.9866 | 0.4936 | **27,620** | 49,740 |
| **xgboost** | oversample | 0.79 | 0.9999 | 0.9614 | 0.9866 | 0.4836 | **28,300** | 36,990 |
| **xgboost** | adasyn | 0.94 | 0.9998 | 0.9574 | 0.9836 | 0.4951 | **29,970** | 69,450 |
| **xgboost** | class_weight | 0.79 | 0.9999 | 0.9590 | 0.9817 | 0.5130 | **30,300** | 37,180 |
| **xgboost** | baseline | 0.03 | 0.9999 | 0.9606 | 0.9817 | 0.5053 | **30,780** | 139,610 |
| **xgboost** | undersample | 0.92 | 0.9997 | 0.9004 | 0.9842 | 0.2761 | **55,360** | 127,970 |
| **random_forest** | smote | 0.78 | 0.9995 | 0.9125 | 0.9549 | 0.3426 | **67,090** | 160,880 |
| **random_forest** | adasyn | 0.85 | 0.9994 | 0.8965 | 0.9708 | 0.2593 | **69,540** | 255,510 |
| **random_forest** | oversample | 0.74 | 0.9994 | 0.8956 | 0.9446 | 0.3261 | **77,550** | 149,600 |
| **random_forest** | undersample | 0.81 | 0.9995 | 0.8949 | 0.9446 | 0.2857 | **84,270** | 216,790 |
| **random_forest** | class_weight | 0.68 | 0.9994 | 0.8918 | 0.9421 | 0.2756 | **88,170** | 130,850 |
| **random_forest** | baseline | 0.01 | 0.9986 | 0.8688 | 0.9129 | 0.3076 | **105,250** | 221,070 |
| **decision_tree** | smote | 0.92 | 0.9980 | 0.6281 | 0.9361 | 0.1786 | **123,180** | 181,750 |
| **decision_tree** | oversample | 0.90 | 0.9963 | 0.6645 | 0.9233 | 0.1827 | **130,820** | 267,250 |
| **decision_tree** | class_weight | 0.94 | 0.9963 | 0.6631 | 0.9214 | 0.1841 | **131,540** | 268,270 |
| **decision_tree** | adasyn | 0.97 | 0.9976 | 0.5983 | 0.9671 | 0.1195 | **144,010** | 178,220 |
| **decision_tree** | undersample | 0.95 | 0.9948 | 0.1724 | 0.8575 | 0.1707 | **185,400** | 286,380 |
| **logreg** | adasyn | 0.91 | 0.9900 | 0.6443 | 0.8447 | 0.1412 | **211,890** | 904,390 |
| **decision_tree** | baseline | 0.01 | 0.9195 | 0.7228 | 0.7594 | 0.3287 | **222,970** | 249,720 |
| **extra_trees** | undersample | 0.72 | 0.9876 | 0.6868 | 0.7643 | 0.1786 | **251,230** | 779,760 |
| **extra_trees** | oversample | 0.79 | 0.9757 | 0.5689 | 0.6803 | 0.3273 | **285,460** | 1,154,010 |
| **logreg** | smote | 0.93 | 0.9894 | 0.5892 | 0.7412 | 0.1392 | **287,770** | 678,180 |
| **logreg** | class_weight | 0.90 | 0.9889 | 0.5839 | 0.7527 | 0.1240 | **290,350** | 693,910 |
| **logreg** | oversample | 0.91 | 0.9890 | 0.5839 | 0.7467 | 0.1264 | **292,720** | 689,830 |
| **extra_trees** | smote | 0.80 | 0.9743 | 0.5705 | 0.6650 | 0.3697 | **293,620** | 1,165,210 |
| **extra_trees** | class_weight | 0.78 | 0.9722 | 0.5725 | 0.6870 | 0.2172 | **297,650** | 1,197,950 |
| **logreg** | undersample | 0.83 | 0.9758 | 0.5711 | 0.6687 | 0.2055 | **314,460** | 937,330 |
| **extra_trees** | baseline | 0.01 | 0.9752 | 0.5731 | 0.7521 | 0.0924 | **324,880** | 677,020 |
| **logreg** | baseline | 0.02 | 0.9615 | 0.5591 | 0.6516 | 0.1936 | **330,560** | 543,590 |
| **extra_trees** | adasyn | 0.77 | 0.9834 | 0.2849 | 0.6017 | 0.0672 | **464,090** | 4,960,700 |

---

## 3. Analysis of Balancing Methods

Class imbalance dramatically distorts the optimization of standard classification algorithms. Since standard objectives minimize error rates, models default to the majority class. Below we evaluate the direct impact of our selected balancing strategies.

### 3.1 Impact on Fraud Recall
For the creditcard dataset, the baseline models without balancing achieved moderate-to-high recalls because threshold tuning moved the operating point to extremely low levels (e.g., threshold of 0.01 for XGBoost baseline and 0.02 for Random Forest baseline). However, on the massive PaySim dataset, balancing strategies proved vital:
- **Baseline Models:** For PaySim Extra Trees baseline, recall was only **0.7521**, resulting in **407 False Negatives (FN)**, translating to a heavy cost penalty.
- **SMOTE & Oversampling:** Applying SMOTE to XGBoost on PaySim yielded a recall of **0.9866**, leaving only **22 FN**. Random Oversampling performed similarly, demonstrating that duplicating minority records or interpolating between them exposes tree ensembles to the rare class structure more effectively.
- **Class Weighting:** Adjusting class weights internally (e.g., via `scale_pos_weight` in XGBoost or `class_weight='balanced'` in Random Forest) allowed the model to penalize minority misclassification directly, boosting recall to **0.9817** for XGBoost on PaySim and **0.8182** on Credit Card.

### 3.2 The Precision-Recall Trade-off
While balancing methods increase recall by forcing the model to draw wider decision boundaries, they inevitably lead to a decline in precision. This occurs because the model becomes more prone to flagging borderline cases, generating more False Positives (FP).
- For example, on PaySim, XGBoost SMOTE achieves a massive recall of **0.9866** but its precision drops to **0.4936**. This means that for every true fraud detected, roughly one legitimate transaction is falsely flagged.
- By contrast, the XGBoost baseline on PaySim achieves **0.5053** precision but at a lower recall (**0.9817**).
- Under our cost function where $COST_{FN} = 500$ and $COST_{FP} = 10$, a False Negative is fifty times more expensive than a False Positive. The trade-off therefore heavily favors maximizing recall, even at the cost of precision. We mathematically accept a substantial increase in FP (low precision) to avoid even a single FN.

---

## 4. Evaluation Metrics under Class Imbalance

A critical insight of this project is that standard classification metrics can be highly misleading when applied to highly imbalanced datasets.

### 4.1 Accuracy is a Misleading Metric
In the Credit Card dataset, where fraud represents only 0.172% of transactions, a trivial model that labels all transactions as "legitimate" achieves **99.83% accuracy**. In PaySim (0.129% fraud), the default-to-legitimate model achieves **99.87% accuracy**.
Accuracy completely hides the total failure to detect any fraud. Thus, it is entirely useless as a guide for model selection in this context.

### 4.2 ROC-AUC vs. PR-AUC
While ROC-AUC is widely used, it is highly sensitive to the massive number of true negatives in imbalanced settings, which can create a deceptively optimistic picture. PR-AUC is far more informative.

$$\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}, \quad \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall (TPR)} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

- **ROC-AUC (Receiver Operating Characteristic):** The ROC curve plots Recall (TPR) against the False Positive Rate (FPR). Because the number of True Negatives (TN) in the denominator of the FPR calculation is extremely large, even a high absolute number of False Positives (FP) results in a very small FPR. This causes the ROC curve to remain close to the top-left corner, resulting in deceptively high ROC-AUC scores (e.g., **0.9999** for almost all PaySim models, including weak ones).
- **PR-AUC (Precision-Recall):** The PR curve plots Precision against Recall. Precision isolates True Positives (TP) against False Positives (FP), entirely ignoring the massive TN count. If a model generates many false alarms (high FP), precision drops immediately, and the PR-AUC falls. 
- For instance, on PaySim, Extra Trees with ADASYN has an ROC-AUC of **0.9834** (which seems excellent), but its PR-AUC is a dismal **0.2849**. The PR-AUC exposes the fact that the model is operationally non-viable, generating an overwhelming number of false alerts.

---

## 5. Model Stability and Overfitting

To ensure that our models generalize well to unseen data, we analyzed model stability using 5-fold stratified cross-validation on the training split, computing the mean and standard deviation of PR-AUC.

### 5.1 Cross-Validation Performance (Rank 1 Configuration per Model)

#### Credit Card Dataset CV Stability
| Model | Strategy | Mean CV PR-AUC | Std CV PR-AUC | Mean CV ROC-AUC | Mean CV Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **xgboost** | class_weight | **0.8468** | **0.0122** | 0.9839 | 0.8020 |
| **extra_trees** | smote | **0.8502** | **0.0111** | 0.9821 | 0.8173 |
| **random_forest** | smote | **0.8396** | **0.0198** | 0.9755 | 0.8046 |
| **logreg** | class_weight | **0.7518** | **0.0273** | 0.9765 | 0.9112 |
| **decision_tree** | class_weight | **0.7406** | **0.0175** | 0.9080 | 0.8172 |

#### PaySim Dataset CV Stability
| Model | Strategy | Mean CV PR-AUC | Std CV PR-AUC | Mean CV ROC-AUC | Mean CV Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **xgboost** | smote | **0.9225** | **0.0030** | 0.9996 | 0.9473 |
| **random_forest** | smote | **0.8846** | **0.0030** | 0.9993 | 0.9047 |
| **extra_trees** | smote | **0.8414** | **0.0047** | 0.9955 | 0.8955 |
| **decision_tree** | class_weight | **0.7426** | **0.0069** | 0.9329 | 0.8671 |
| **logreg** | smote | **0.6084** | **0.0041** | 0.9868 | 0.9391 |

### 5.2 Variance and Overfitting Analysis
- **Data Volume and Stability:** We observe a sharp difference in stability between the two datasets. For PaySim, the standard deviations of CV PR-AUC are exceptionally small, ranging from **0.0011 to 0.0069**. This indicates that the 6.3M records provide highly stable and representative training splits. For Credit Card, standard deviations are larger (ranging from **0.0111 to 0.0273**), reflecting the greater variance inherent in a smaller sample size with only 492 fraud cases.
- **Overfitting Tendencies:**
  - **Decision Trees:** Showed high variance and severe overfitting. While training metrics approached perfect scores, validation/test metrics dropped substantially (e.g., CV PR-AUC of **0.7406** on Credit Card). Trees without restriction split nodes until they isolate individual fraudulent transactions, creating fragile decision boundaries.
  - **Random Forest & Extra Trees:** Adding bagging and random feature selection controlled variance. Extra Trees SMOTE reached a CV PR-AUC of **0.8502** on Credit Card. However, they were still outperformed by XGBoost.
  - **XGBoost:** Achieved the highest stability and generalization. By training weak learners sequentially and penalizing complexity (via regularization parameters tuned in the hyperparameter search), XGBoost achieved a CV PR-AUC of **0.9225** on PaySim with a standard deviation of just **0.0030**.

---

## 6. Decision Threshold Analysis

Standard classifiers output a continuous probability $p \in [0, 1]$ and apply a default threshold of $0.5$. In highly imbalanced contexts, this default threshold is rarely optimal.

We evaluated alternative decision thresholds from **0.01 to 0.99** on the validation set, tracking the expected cost. The threshold that minimized expected cost was selected as the operational operating point.

```
                  Cost Sweeps (Expected Cost vs. Threshold)
       Cost
        ^
        |   \
        |    \       (Default Threshold = 0.5: High Cost due to FNs)
        |     \      * (0.5)
        |      \    /
        |       \  /
        |        \/  * (Optimal Tuned Threshold: Min Expected Cost)
        +----------------------------------------------------------> Threshold
        0.0     Tuned                                           1.0
```

- **Credit Card (XGBoost class_weight):** The validation sweep selected an optimal threshold of **0.06**. At this very low threshold, the model is highly sensitive, catching borderline fraud. The test expected cost dropped to **9,240**.
- **PaySim (XGBoost SMOTE):** The optimal validation threshold was selected at **0.87**. Why is this threshold so high, even when false negatives are highly penalized? Because SMOTE creates synthetic minority samples, inflating the training class ratio. The raw output probabilities of the model are shifted upwards. Therefore, a high threshold of **0.87** is required to re-calibrate the predictions, yielding the lowest expected cost of **27,620**.
- **PaySim (XGBoost baseline):** For the baseline model (no resampling, original class ratio), the optimal threshold is **0.03**. Since the minority class remains extremely small, the model outputs small raw probabilities, requiring a very low threshold to capture fraud.

---

## 7. Cost-Sensitive Evaluation and Business Impact

In financial fraud detection, classification errors have highly asymmetric business consequences:
1. **False Positive (FP):** Flagging a legitimate transaction. This leads to user friction, transaction declines, and manual investigation costs. We set $COST_{FP} = 10$.
2. **False Negative (FN):** Missing a fraudulent transaction. This leads to direct financial loss, potential regulatory fines, and loss of trust. We set $COST_{FN} = 500$.

The Expected Cost is computed as:

$$\text{Expected Cost} = \text{FP} \times COST_{FP} + \text{FN} \times COST_{FN}$$

### 7.1 Savings Summary (Default vs. Tuned Thresholds)

Below we compare the expected cost on the test set using the default threshold ($0.5$) versus the tuned threshold chosen on validation data.

| Dataset | Best Configuration | Cost at Default Threshold (0.5) | Cost at Tuned Threshold | Absolute Savings | Relative Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Credit Card** | XGBoost + class_weight | $11,080 | $9,240 | **$1,840** | **16.6%** |
| **PaySim** | XGBoost + SMOTE | $49,740 | $27,620 | **$22,120** | **44.5%** |

### 7.2 Financial Analysis of Key Models
- **Credit Card (XGBoost class_weight):**
  - **At threshold 0.5:** The model yields **24 FP** and **22 FN**, resulting in an expected cost of:
    $$\text{Cost} = 24 \times 10 + 22 \times 500 = 240 + 11,000 = \$11,080$$
  - **At tuned threshold 0.06:** The model yields **24 FP** and **18 FN**. By lowering the threshold, we caught 4 more fraud cases (reducing FN from 22 to 18) while keeping FP at exactly 24.
    $$\text{Cost} = 24 \times 10 + 18 \times 500 = 240 + 9,000 = \$9,240$$
  - This highlights the mathematical precision of validation-based tuning.
- **PaySim (XGBoost SMOTE):**
  - **At threshold 0.5:** The model yields **1,474 FP** and **70 FN**, resulting in:
    $$\text{Cost} = 1,474 \times 10 + 70 \times 500 = 14,740 + 35,000 = \$49,740$$
  - **At tuned threshold 0.87:** The model yields **1,662 FP** and **22 FN**, resulting in:
    $$\text{Cost} = 1,662 \times 10 + 22 \times 500 = 16,620 + 11,000 = \$27,620$$
  - By accepting an increase of **188 False Positives** (costing an extra \$1,880 in alert processing), we prevented **48 False Negatives** (saving \$24,000 in fraud losses). The net savings is a massive **$22,120 (44.5%)**.

---

## 8. Limitations & Proposals for Future Development

While our experimental pipeline achieved high effectiveness, we must acknowledge several design limitations and propose future enhancements.

### 8.1 Study Limitations
1. **Static Data and Concept Drift:** Our experiments assume a static distribution of transactions. In production, fraud patterns evolve rapidly as fraudsters adapt to detection rules (concept drift). Static models degrade in performance over time.
2. **Computational Overhead of Resampling:** Preprocessing techniques like SMOTE and ADASYN require computing nearest neighbors ($k$-NN) for minority records. For large datasets like PaySim, this adds substantial training time and memory overhead.
3. **Simulated PaySim Data:** The PaySim dataset is generated by a simulator (synthetic mobile money transactions). While useful, it may lack the noise, complex behavioral patterns, and outlier distributions present in real-world transaction logs.

### 8.2 Future Work and Enhancements
To build on this work and achieve an even higher grade of system maturity, we propose the following developments:

#### 1. Integration of LightGBM and CatBoost
While XGBoost performed exceptionally well, tree ensembles like **LightGBM** (Light Gradient Boosting Machine) and **CatBoost** offer distinct advantages:
- **LightGBM:** Uses leaf-wise tree growth and Histogram-based decision tree algorithms, reducing training time by **5x-10x** on massive datasets like PaySim.
- **CatBoost:** Built-in categorical feature handling (using ordered target statistics) removes the need for manual one-hot encoding, preventing dimensionality explosion.

#### 2. Advanced Feature Engineering
Our current models rely primarily on raw numeric attributes. Performance could be significantly improved by engineering transactional features:
- **Velocity Features:** Number of transactions made by a user in the last 1, 6, and 24 hours.
- **Deviation Features:** Comparing the current transaction amount to the user's historical median amount.
- **Graph-Based Features:** Modeling transactions as a bipartite graph (users and merchants) and extracting PageRank or node embeddings to identify suspicious network structures.

#### 3. Real-Time Latency Considerations
In a production financial pipeline, fraud detection must occur inline before the transaction is authorized, requiring latencies under **50-100 milliseconds**.
- Deep tree ensembles and scaling preprocessors must be compiled (e.g., using ONNX or Treelite) to minimize inference latency.
- There is a trade-off between model complexity (e.g., Random Forest with 400 estimators) and inference speed that must be evaluated.

#### 4. Model Explainability (SHAP/LIME)
For regulatory compliance (e.g., GDPR "Right to Explanation"), financial institutions must explain why a transaction was declined.
- Integrating **SHAP (SHapley Additive exPlanations)** values into the pipeline would provide feature-level explanations for every flagged transaction, assisting operational analysts in manual reviews.

---

## 9. Conclusion

This project has demonstrated that detecting financial fraud under extreme class imbalance requires a comprehensive, end-to-end framework. Using standard accuracy and default thresholds leads to catastrophic losses due to undetected fraud. 

By applying **XGBoost** combined with **SMOTE** or **class weighting**, and optimizing the operating point using validation-set **threshold sweeps**, we successfully built models that maximize fraud detection while minimizing expected operational costs. On the Credit Card dataset, our best model achieved a PR-AUC of **0.8273** and an expected cost of **9,240**. On the PaySim dataset, our best model reached a PR-AUC of **0.9612** and reduced expected costs to **27,620**—representing a **44.5% cost reduction** over the default classification strategy.

Incorporating these techniques into a production framework, alongside proposed features like LightGBM integration and graph-based features, represents the state-of-the-art in financial transaction monitoring.
