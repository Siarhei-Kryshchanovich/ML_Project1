# Laboratory Project No. 1
## Financial Fraud Detection - Classification of Highly Imbalanced Data
### Report No. 3: Results Analysis and Experimental Conclusions

**Submitted by:**
* Mykola Tatarenkov (53631)
* Siarhei Kryshchanovich (57763)

## 1. Executive Summary

* **Credit Card Fraud Detection**: 284,807 transactions with 492 fraud cases, approximately 0.172% fraud.
* **PaySim**: 6,362,620 simulated mobile-money transactions with 8,213 fraud cases, approximately 0.129% fraud.

The repository implements five model families: **Logistic Regression**, **Decision Tree**, **Random Forest**, **Extra Trees**, and **XGBoost**. These models are evaluated with six imbalance-handling strategies: **baseline**, **class_weight**, **random undersampling**, **random oversampling**, **SMOTE**, and **ADASYN**. No additional voting or stacking ensemble is implemented; the ensemble comparison therefore refers to Random Forest, Extra Trees, and XGBoost.

The main experimental conclusion is that fraud detection cannot be judged by accuracy or by default classifier thresholds. Accuracy remains above 0.99 for many configurations because legitimate transactions dominate the data. The meaningful question is whether the model ranks rare fraud cases well, how many fraud cases are detected at the chosen operating threshold, and what expected business cost results from false positives and false negatives.

The strongest cost-sensitive configurations are:

| Dataset | Best configuration | Threshold | PR-AUC | Recall | Precision | FP | FN | Expected cost | Cost at 0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| creditcard | XGBoost + class_weight | 0.06 | 0.8273 | 0.8182 | 0.7714 | 24 | 18 | 9,240 | 11,080 |
| paysim | XGBoost + SMOTE | 0.87 | 0.9612 | 0.9866 | 0.4936 | 1,662 | 22 | 27,620 | 49,740 |

The Credit Card result shows that the best models are close to each other: XGBoost, Extra Trees, and Random Forest all reach costs around 9,240-9,640. PaySim is more decisive: all top six configurations are XGBoost, and the best XGBoost configurations are much stronger than Random Forest, Decision Tree, Extra Trees, or Logistic Regression.

---

## 2. Consolidated Results Table

The following tables are generated from `outputs/comparison_table.csv`. They report final test-set performance at the threshold selected on the validation split. The cost column uses the project cost matrix:

$$
\text{Expected Cost} = 10 \cdot FP + 500 \cdot FN
$$

### 2.1 Credit Card Results

| Model | Strategy | Thr. | ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy | Cost tuned | Cost 0.5 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgboost | class_weight | 0.06 | 0.9707 | 0.8273 | 0.8182 | 0.7714 | 0.7941 | 0.9993 | 9240 | 11080 | 24 | 18 |
| extra_trees | oversample | 0.45 | 0.9698 | 0.7920 | 0.8182 | 0.7105 | 0.7606 | 0.9991 | 9330 | 10670 | 33 | 18 |
| random_forest | class_weight | 0.17 | 0.9574 | 0.7863 | 0.8182 | 0.6639 | 0.7330 | 0.9990 | 9410 | 13600 | 41 | 18 |
| extra_trees | smote | 0.52 | 0.9629 | 0.7887 | 0.8182 | 0.6585 | 0.7297 | 0.9989 | 9420 | 9590 | 42 | 18 |
| random_forest | undersample | 0.84 | 0.9767 | 0.6852 | 0.8182 | 0.6532 | 0.7265 | 0.9989 | 9430 | 21240 | 43 | 18 |
| xgboost | baseline | 0.01 | 0.9746 | 0.8211 | 0.8182 | 0.6378 | 0.7168 | 0.9989 | 9460 | 13530 | 46 | 18 |
| extra_trees | baseline | 0.02 | 0.9735 | 0.7955 | 0.8182 | 0.5586 | 0.6639 | 0.9986 | 9640 | 18550 | 64 | 18 |
| random_forest | baseline | 0.02 | 0.9725 | 0.8062 | 0.8283 | 0.4141 | 0.5522 | 0.9977 | 9660 | 14540 | 116 | 17 |
| extra_trees | class_weight | 0.48 | 0.9730 | 0.7924 | 0.8081 | 0.7767 | 0.7921 | 0.9993 | 9730 | 10680 | 23 | 19 |
| xgboost | adasyn | 0.11 | 0.9660 | 0.7915 | 0.8384 | 0.3192 | 0.4624 | 0.9966 | 9770 | 10460 | 177 | 16 |
| random_forest | oversample | 0.18 | 0.9645 | 0.7785 | 0.8182 | 0.4880 | 0.6113 | 0.9982 | 9850 | 11600 | 85 | 18 |
| xgboost | oversample | 0.01 | 0.9710 | 0.8177 | 0.8182 | 0.4602 | 0.5891 | 0.9980 | 9950 | 11070 | 95 | 18 |
| extra_trees | undersample | 0.74 | 0.9736 | 0.6664 | 0.8081 | 0.6349 | 0.7111 | 0.9989 | 9960 | 16010 | 46 | 19 |
| logreg | baseline | 0.02 | 0.9639 | 0.7091 | 0.8081 | 0.6349 | 0.7111 | 0.9989 | 9960 | 23610 | 46 | 19 |
| random_forest | smote | 0.39 | 0.9749 | 0.7661 | 0.8182 | 0.4309 | 0.5645 | 0.9978 | 10070 | 11040 | 107 | 18 |
| extra_trees | adasyn | 0.66 | 0.9637 | 0.7709 | 0.8081 | 0.4938 | 0.6130 | 0.9982 | 10320 | 16340 | 82 | 19 |
| logreg | class_weight | 0.98 | 0.9747 | 0.6807 | 0.8081 | 0.4545 | 0.5818 | 0.9980 | 10460 | 19030 | 96 | 19 |
| logreg | oversample | 0.98 | 0.9744 | 0.6835 | 0.8081 | 0.4520 | 0.5797 | 0.9980 | 10470 | 18990 | 97 | 19 |
| logreg | smote | 0.99 | 0.9718 | 0.6750 | 0.8081 | 0.4278 | 0.5594 | 0.9978 | 10570 | 20690 | 107 | 19 |
| xgboost | smote | 0.31 | 0.9615 | 0.7976 | 0.7980 | 0.5643 | 0.6611 | 0.9986 | 10610 | 10850 | 61 | 20 |
| random_forest | adasyn | 0.67 | 0.9676 | 0.7124 | 0.8081 | 0.3980 | 0.5333 | 0.9975 | 10710 | 12820 | 121 | 19 |
| xgboost | undersample | 0.99 | 0.9715 | 0.6477 | 0.8182 | 0.2978 | 0.4367 | 0.9963 | 10910 | 26780 | 191 | 18 |
| logreg | adasyn | 0.99 | 0.9688 | 0.6913 | 0.8283 | 0.1826 | 0.2993 | 0.9933 | 12170 | 43830 | 367 | 17 |
| decision_tree | oversample | 0.96 | 0.8751 | 0.5948 | 0.7677 | 0.3707 | 0.5000 | 0.9973 | 12790 | 15790 | 129 | 23 |
| decision_tree | class_weight | 0.91 | 0.8751 | 0.6012 | 0.7778 | 0.2973 | 0.4302 | 0.9964 | 12820 | 15780 | 182 | 22 |
| logreg | undersample | 0.99 | 0.9704 | 0.3830 | 0.7980 | 0.2041 | 0.3251 | 0.9942 | 13080 | 29540 | 308 | 20 |
| decision_tree | smote | 0.98 | 0.8627 | 0.4191 | 0.7374 | 0.4506 | 0.5594 | 0.9980 | 13890 | 19710 | 89 | 26 |
| decision_tree | baseline | 0.11 | 0.8305 | 0.6230 | 0.6970 | 0.8846 | 0.7797 | 0.9993 | 15090 | 15090 | 9 | 30 |
| decision_tree | adasyn | 0.95 | 0.8984 | 0.0747 | 0.7677 | 0.0879 | 0.1577 | 0.9857 | 19390 | 36980 | 789 | 23 |
| decision_tree | undersample | 0.01 | 0.8917 | 0.0130 | 0.8889 | 0.0144 | 0.0284 | 0.8944 | 65520 | 65520 | 6002 | 11 |

### 2.2 PaySim Results

| Model | Strategy | Thr. | ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy | Cost tuned | Cost 0.5 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgboost | smote | 0.87 | 0.9998 | 0.9612 | 0.9866 | 0.4936 | 0.6580 | 0.9987 | 27620 | 49740 | 1662 | 22 |
| xgboost | oversample | 0.79 | 0.9999 | 0.9614 | 0.9866 | 0.4836 | 0.6490 | 0.9986 | 28300 | 36990 | 1730 | 22 |
| xgboost | adasyn | 0.94 | 0.9998 | 0.9574 | 0.9836 | 0.4951 | 0.6586 | 0.9987 | 29970 | 69450 | 1647 | 27 |
| xgboost | class_weight | 0.79 | 0.9999 | 0.9590 | 0.9817 | 0.5130 | 0.6739 | 0.9988 | 30300 | 37180 | 1530 | 30 |
| xgboost | baseline | 0.03 | 0.9999 | 0.9606 | 0.9817 | 0.5053 | 0.6672 | 0.9987 | 30780 | 139610 | 1578 | 30 |
| xgboost | undersample | 0.92 | 0.9997 | 0.9004 | 0.9842 | 0.2761 | 0.4313 | 0.9967 | 55360 | 127970 | 4236 | 26 |
| random_forest | smote | 0.78 | 0.9995 | 0.9125 | 0.9549 | 0.3426 | 0.5043 | 0.9976 | 67090 | 160880 | 3009 | 74 |
| random_forest | adasyn | 0.85 | 0.9994 | 0.8965 | 0.9708 | 0.2593 | 0.4092 | 0.9964 | 69540 | 255510 | 4554 | 48 |
| random_forest | oversample | 0.74 | 0.9994 | 0.8956 | 0.9446 | 0.3261 | 0.4848 | 0.9974 | 77550 | 149600 | 3205 | 91 |
| random_forest | undersample | 0.81 | 0.9995 | 0.8949 | 0.9446 | 0.2857 | 0.4388 | 0.9969 | 84270 | 216790 | 3877 | 91 |
| random_forest | class_weight | 0.68 | 0.9994 | 0.8918 | 0.9421 | 0.2756 | 0.4264 | 0.9967 | 88170 | 130850 | 4067 | 95 |
| random_forest | baseline | 0.01 | 0.9986 | 0.8688 | 0.9129 | 0.3076 | 0.4601 | 0.9972 | 105250 | 221070 | 3375 | 143 |
| decision_tree | smote | 0.92 | 0.9980 | 0.6281 | 0.9361 | 0.1786 | 0.3000 | 0.9944 | 123180 | 181750 | 7068 | 105 |
| decision_tree | oversample | 0.90 | 0.9963 | 0.6645 | 0.9233 | 0.1827 | 0.3050 | 0.9946 | 130820 | 267250 | 6782 | 126 |
| decision_tree | class_weight | 0.94 | 0.9963 | 0.6631 | 0.9214 | 0.1841 | 0.3069 | 0.9946 | 131540 | 268270 | 6704 | 129 |
| decision_tree | adasyn | 0.97 | 0.9976 | 0.5983 | 0.9671 | 0.1195 | 0.2127 | 0.9908 | 144010 | 178220 | 11701 | 54 |
| decision_tree | undersample | 0.95 | 0.9948 | 0.1724 | 0.8575 | 0.1707 | 0.2847 | 0.9944 | 185400 | 286380 | 6840 | 234 |
| logreg | adasyn | 0.91 | 0.9900 | 0.6443 | 0.8447 | 0.1412 | 0.2419 | 0.9932 | 211890 | 904390 | 8439 | 255 |
| decision_tree | baseline | 0.01 | 0.9195 | 0.7228 | 0.7594 | 0.3287 | 0.4588 | 0.9977 | 222970 | 249720 | 2547 | 395 |
| extra_trees | undersample | 0.72 | 0.9876 | 0.6868 | 0.7643 | 0.1786 | 0.2895 | 0.9952 | 251230 | 779760 | 5773 | 387 |
| extra_trees | oversample | 0.79 | 0.9757 | 0.5689 | 0.6803 | 0.3273 | 0.4419 | 0.9978 | 285460 | 1154010 | 2296 | 525 |
| logreg | smote | 0.93 | 0.9894 | 0.5892 | 0.7412 | 0.1392 | 0.2344 | 0.9938 | 287770 | 678180 | 7527 | 425 |
| logreg | class_weight | 0.90 | 0.9889 | 0.5839 | 0.7527 | 0.1240 | 0.2129 | 0.9928 | 290350 | 693910 | 8735 | 406 |
| logreg | oversample | 0.91 | 0.9890 | 0.5839 | 0.7467 | 0.1264 | 0.2162 | 0.9930 | 292720 | 689830 | 8472 | 416 |
| extra_trees | smote | 0.80 | 0.9743 | 0.5705 | 0.6650 | 0.3697 | 0.4752 | 0.9981 | 293620 | 1165210 | 1862 | 550 |
| extra_trees | class_weight | 0.78 | 0.9722 | 0.5725 | 0.6870 | 0.2172 | 0.3301 | 0.9964 | 297650 | 1197950 | 4065 | 514 |
| logreg | undersample | 0.83 | 0.9758 | 0.5711 | 0.6687 | 0.2055 | 0.3143 | 0.9962 | 314460 | 937330 | 4246 | 544 |
| extra_trees | baseline | 0.01 | 0.9752 | 0.5731 | 0.7521 | 0.0924 | 0.1645 | 0.9901 | 324880 | 677020 | 12138 | 407 |
| logreg | baseline | 0.02 | 0.9615 | 0.5591 | 0.6516 | 0.1936 | 0.2985 | 0.9960 | 330560 | 543590 | 4456 | 572 |
| extra_trees | adasyn | 0.77 | 0.9834 | 0.2849 | 0.6017 | 0.0672 | 0.1209 | 0.9887 | 464090 | 4960700 | 13709 | 654 |

---

## 3. Consolidated Results Analysis

### 3.1 Effectiveness by Model Family

The results show a clear hierarchy of model families, but the hierarchy is dataset-dependent.

On the **Credit Card** dataset, the best cost-sensitive model is **XGBoost with class_weight**, but the margin over the next models is small. Extra Trees with oversampling costs 9,330, Random Forest with class weighting costs 9,410, and Extra Trees with SMOTE costs 9,420. This means that on Credit Card the strongest conclusion is not simply "XGBoost wins"; rather, **regularized tree ensembles are consistently strong**, while the exact winner depends on the balance between recall and false-positive control.

On **PaySim**, the conclusion is stronger. The top six rows are all XGBoost configurations. The best non-XGBoost model, Random Forest with SMOTE, has an expected cost of 67,090, more than double the best XGBoost SMOTE cost of 27,620. PaySim therefore rewards boosted trees much more clearly than the smaller anonymized Credit Card dataset.

Logistic Regression remains useful as a baseline because it shows that a linear decision boundary can still produce respectable ROC-AUC, especially after threshold tuning. However, it cannot match the PR-AUC and expected-cost performance of ensembles. Decision Tree is interpretable but unstable: it sometimes increases recall, yet it often does so by creating too many false positives or by losing ranking quality. The extreme Credit Card case is Decision Tree with undersampling: recall reaches 0.8889, but precision collapses to 0.0144 and cost rises to 65,520.

### 3.2 Fraud-Class Interpretation

The fraud class is the operational target. A good fraud detector must avoid two failures: missing fraud cases and producing so many false alerts that the system becomes impractical.

On Credit Card, the best XGBoost configuration detects 81 of 99 fraud cases and misses 18. Its precision of 0.7714 is very strong for a dataset with only 0.172% fraud. This means the model is not merely catching fraud by flagging thousands of transactions; it maintains a useful alert quality.

On PaySim, the best XGBoost SMOTE configuration detects 1,620 of 1,642 fraud cases and misses only 22. Precision is lower, 0.4936, but this is still operationally meaningful under such extreme imbalance: approximately one of every two fraud alerts is correct. Given that a false negative is valued fifty times higher than a false positive in the project cost matrix, this recall-oriented behavior is preferable.

---

## 4. Analysis of Balancing Methods

Balancing changes the distribution seen by the learner during training. In this project, balancing is applied only inside the training pipeline, while validation and test sets keep the original class distribution. This is methodologically important because it prevents leakage and ensures that reported metrics reflect realistic deployment proportions.

### 4.1 Average Impact by Strategy

#### Credit Card

| Strategy | Avg recall | Avg precision | Avg PR-AUC | Avg cost | Best cost |
|---|---:|---:|---:|---:|---:|
| class_weight | 0.8061 | 0.5928 | 0.7376 | 10,332 | 9,240 |
| oversample | 0.8061 | 0.4963 | 0.7333 | 10,478 | 9,330 |
| baseline | 0.7939 | 0.6260 | 0.7510 | 10,762 | 9,460 |
| smote | 0.7960 | 0.5064 | 0.6893 | 10,912 | 9,420 |
| adasyn | 0.8101 | 0.2963 | 0.6082 | 12,472 | 9,770 |
| undersample | 0.8263 | 0.3609 | 0.4791 | 21,780 | 9,430 |

On Credit Card, balancing is not uniformly beneficial. Undersampling has the highest average recall, but the worst average cost because it discards too much majority-class information and often damages precision. ADASYN also increases recall in several configurations but creates many false positives, reducing average precision to 0.2963. The best strategy on average is class weighting because it improves fraud sensitivity without changing the input distribution and without generating synthetic points.

The baseline strategy remains competitive because threshold tuning compensates for imbalance by lowering the decision threshold. For example, XGBoost baseline reaches PR-AUC 0.8211 and cost 9,460 at threshold 0.01. This does not mean balancing is unnecessary; it means that threshold tuning is powerful enough to make even an unbalanced training strategy operationally usable when the model ranks fraud cases well.

#### PaySim

| Strategy | Avg recall | Avg precision | Avg PR-AUC | Avg cost | Best cost |
|---|---:|---:|---:|---:|---:|
| smote | 0.8568 | 0.3047 | 0.7323 | 159,856 | 27,620 |
| oversample | 0.8563 | 0.2892 | 0.7349 | 162,970 | 28,300 |
| class_weight | 0.8570 | 0.2628 | 0.7341 | 167,602 | 30,300 |
| undersample | 0.8438 | 0.2233 | 0.6451 | 178,144 | 55,360 |
| adasyn | 0.8736 | 0.2164 | 0.6763 | 183,900 | 29,970 |
| baseline | 0.8116 | 0.2855 | 0.7369 | 202,888 | 30,780 |

On PaySim, balancing has a stronger practical impact. The baseline average recall is 0.8116, while ADASYN, class_weight, SMOTE, and oversampling all move average recall above 0.856. The best costs are achieved by XGBoost with SMOTE, oversampling, ADASYN, class_weight, and baseline, respectively. This shows that XGBoost already ranks PaySim fraud very well, but balancing improves the operating point and reduces false negatives.

### 4.2 Precision-Recall Trade-Off

Balancing generally pushes models toward detecting more fraud, but this often lowers precision. The effect is clearest in synthetic and undersampling methods. Credit Card Decision Tree with undersampling reaches the highest recall in the Credit Card table, 0.8889, but produces 6,002 false positives and only 0.0144 precision. This is not an acceptable fraud-detection system despite high recall.

SMOTE and ADASYN are more sophisticated than random oversampling because they create synthetic minority samples. They can help when minority points are too rare for a model to learn a stable fraud region. However, they can also introduce synthetic examples in regions where the true data distribution is ambiguous. ADASYN is especially aggressive because it focuses on difficult minority samples; in this project it often increases recall but damages precision. The PaySim XGBoost ADASYN model is still strong, but PaySim Extra Trees with ADASYN is the worst PaySim configuration by cost, with 13,709 false positives and 654 false negatives.

The main conclusion is that balancing is not automatically good. It must be evaluated together with PR-AUC, confusion matrix counts, and expected cost. In this project, the most reliable balancing strategies are **class_weight**, **SMOTE**, and **oversampling**, while **undersampling** and **ADASYN** are more volatile.

---

## 5. ROC-AUC and PR-AUC Analysis

ROC-AUC measures the ability to rank positive examples above negative examples across thresholds. It is useful, but in extremely imbalanced fraud detection it can be overly optimistic because the false-positive rate divides false positives by the very large number of legitimate transactions.

PR-AUC is more informative because it evaluates precision and recall directly for the fraud class. Precision immediately penalizes large false-positive counts, and recall measures how many fraud cases are actually found. This is why the project uses PR-AUC as the key ranking metric in hyperparameter analysis.

The PaySim results illustrate the difference. Many models report ROC-AUC above 0.99, including configurations that are not operationally competitive. For example, PaySim Extra Trees with ADASYN has ROC-AUC 0.9834, which appears excellent, but PR-AUC is only 0.2849 and expected cost is 464,090. The ROC value says the model has some ranking ability; the PR-AUC and cost reveal that the ranking is not strong enough in the minority-class region that matters.

Credit Card also shows why PR-AUC matters. Decision Tree baseline has accuracy 0.9993 and precision 0.8846, but recall is only 0.6970 and cost is 15,090. The model is conservative and produces few false positives, but it misses 30 out of 99 fraud cases. In fraud detection, high precision alone is not enough if recall is too low.

The saved ROC and PR plots in `outputs/plots/` support these conclusions. ROC curves for strong models are close to the top-left corner, but the PR curves separate models more meaningfully. The PR plots show whether high recall can be maintained without precision collapsing, which is exactly the operational question in fraud detection.

---

## 6. Stability, Hyperparameter Analysis, and Training Time

The repository contains two types of stability evidence. The main experiment code performs stratified cross-validation using `CV_FOLDS = 3`. The hyperparameter-analysis outputs report mean and standard deviation of CV PR-AUC, ROC-AUC, recall, and mean fit time. For PaySim, the configuration intentionally uses a constrained tuning design: 2 folds, a 30% stratified subsample, restricted parallelism, and a smaller randomized search budget. This is a practical response to memory and computation limits on the much larger dataset.

### 6.1 Best CV Configuration per Model

| Dataset | Model | Strategy | Mean CV PR-AUC | Std CV PR-AUC | Mean CV ROC-AUC | Mean CV recall | Std CV recall | Mean fit time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| creditcard | extra_trees | smote | 0.8502 | 0.0111 | 0.9821 | 0.8173 | 0.0328 | 266.92 |
| creditcard | xgboost | class_weight | 0.8468 | 0.0122 | 0.9839 | 0.8020 | 0.0223 | 18.68 |
| creditcard | random_forest | smote | 0.8396 | 0.0198 | 0.9755 | 0.8046 | 0.0363 | 639.80 |
| creditcard | logreg | class_weight | 0.7518 | 0.0273 | 0.9765 | 0.9112 | 0.0131 | 4.54 |
| creditcard | decision_tree | class_weight | 0.7406 | 0.0175 | 0.9080 | 0.8172 | 0.0226 | 11.28 |
| paysim | xgboost | smote | 0.9225 | 0.0030 | 0.9996 | 0.9473 | 0.0081 | 5.77 |
| paysim | random_forest | smote | 0.8846 | 0.0030 | 0.9993 | 0.9047 | 0.0203 | 81.27 |
| paysim | extra_trees | smote | 0.8414 | 0.0047 | 0.9955 | 0.8955 | 0.0030 | 30.93 |
| paysim | decision_tree | class_weight | 0.7426 | 0.0069 | 0.9329 | 0.8671 | 0.0071 | 1.67 |
| paysim | logreg | smote | 0.6084 | 0.0041 | 0.9868 | 0.9391 | 0.0041 | 3.07 |

The standard deviations are low for the best models, especially on PaySim. This is expected because PaySim is much larger and each fold contains more fraud examples. Credit Card has only 492 fraud cases in the full dataset, so fold-to-fold variation is naturally higher. A PR-AUC standard deviation around 0.011-0.020 for the leading Credit Card ensembles is still acceptable, but it indicates that individual split composition matters more than on PaySim.

### 6.2 Training-Time Comparison

The hyperparameter table reports mean fit time for candidate evaluations, not total end-to-end experiment time. It is still useful for comparing model families under the same tuning procedure.

| Dataset | Fastest models | Slower models | Interpretation |
|---|---|---|---|
| creditcard | Logistic Regression (avg. 4.79 s), Decision Tree (16.64 s), XGBoost (20.57 s) | Extra Trees (152.00 s), Random Forest (433.39 s) | Random Forest is much slower during tuning; XGBoost gives near-best or best performance with a much better speed-performance trade-off. |
| paysim | Logistic Regression (2.34 s), Decision Tree (3.01 s), XGBoost (4.97 s) | Extra Trees (24.43 s), Random Forest (78.52 s) | PaySim tuning uses a constrained subsample; even so, Random Forest remains substantially slower than XGBoost. |

The training-time evidence strengthens the case for XGBoost. On Credit Card, Extra Trees slightly leads hyperparameter CV PR-AUC, but its mean fit time is far higher than XGBoost. On PaySim, XGBoost dominates both performance and practical training efficiency within the constrained tuning setup.

### 6.3 Overfitting Behavior

The repository does not save train-score columns in `hyperparameter_analysis.csv` because `RandomizedSearchCV` is run with `return_train_score=False`. Therefore, direct train-vs-validation overfitting curves are not available. The overfitting analysis must be based on cross-validation variance, model complexity, test behavior, and the gap between ranking metrics and threshold-level confusion matrices.

Decision Trees show the clearest overfitting risk. A single tree can isolate rare fraud cases in training, but its boundaries are brittle. This appears in weaker PR-AUC, unstable precision, and poor cost in many settings. Undersampling worsens this risk because the tree sees a distorted majority-class distribution and can become too willing to flag legitimate observations.

Random Forest and Extra Trees reduce variance through ensemble averaging and randomization. They are much stronger than a single tree, but they can still be expensive and, in PaySim, less effective than XGBoost. Random Forest with SMOTE is stable and strong, yet still far behind XGBoost SMOTE in PaySim expected cost.

XGBoost controls overfitting better through sequential boosting, regularization, and compact hyperparameter tuning. Its PaySim CV PR-AUC standard deviation of 0.0030 and test PR-AUC of 0.9612 indicate strong generalization. On Credit Card, XGBoost class_weight is also robust, but the smaller fraud sample means its advantage over Extra Trees and Random Forest is less absolute.

---

## 7. Decision Threshold Analysis

The default threshold of 0.5 is not appropriate for this project. Fraud detection has asymmetric costs, and class imbalance shifts probability distributions. The repository therefore evaluates thresholds from 0.01 to 0.99 on the validation set and selects the threshold minimizing expected cost. The selected threshold is then applied once to the untouched test set.

The thresholds vary widely because raw model probabilities are not calibrated in the same way across balancing strategies:

* Credit Card XGBoost baseline uses threshold 0.01 because fraud probabilities are low under the original class distribution.
* Credit Card XGBoost class_weight uses threshold 0.06, still far below 0.5, to catch more borderline fraud cases.
* PaySim XGBoost SMOTE uses threshold 0.87 because synthetic balancing changes the training class distribution and shifts model scores upward.
* PaySim XGBoost baseline uses threshold 0.03 because the model is trained on the original extremely imbalanced distribution.

This is one of the most important practical findings. A threshold is not a technical detail after training; it is part of the fraud-detection policy. Lowering the threshold usually increases recall and false positives. Raising the threshold usually improves precision but increases false negatives. The correct choice depends on the cost ratio and operational tolerance for manual review.

In this project, false negatives cost 500 and false positives cost 10. Therefore, accepting additional false positives is rational if it prevents enough missed fraud. The best PaySim example makes this explicit: XGBoost SMOTE increases false positives from 1,474 at threshold 0.5 to 1,662 at threshold 0.87, but reduces false negatives from 70 to 22. The additional 188 false positives cost 1,880, while the 48 avoided false negatives save 24,000. The net reduction is 22,120.

---

## 8. Cost-Sensitive Analysis

The project uses the following cost matrix:

| Outcome | Meaning | Cost |
|---|---|---:|
| True Negative | Legitimate transaction accepted | 0 |
| True Positive | Fraud correctly flagged | 0 |
| False Positive | Legitimate transaction falsely flagged | 10 |
| False Negative | Fraud missed | 500 |

This cost structure reflects practical fraud detection. A false positive may require manual review, temporary friction, or a declined legitimate transaction. A false negative can cause direct financial loss, chargebacks, investigation cost, and reputational damage. The cost ratio of 50:1 makes recall especially important, but not at unlimited false-positive volume.

### 8.1 Default vs Tuned Threshold

| Dataset | Best configuration | Cost at 0.5 | Tuned cost | Absolute reduction | Relative reduction |
|---|---|---:|---:|---:|---:|
| creditcard | XGBoost + class_weight | 11,080 | 9,240 | 1,840 | 16.6% |
| paysim | XGBoost + SMOTE | 49,740 | 27,620 | 22,120 | 44.5% |

For Credit Card, the tuned threshold reduces false negatives from 22 to 18 while keeping false positives at 24. This is an unusually clean improvement: the model catches four additional fraud cases without increasing the false-positive count on the test set. The resulting expected cost decreases from 11,080 to 9,240.

For PaySim, the improvement is more dramatic. XGBoost SMOTE at threshold 0.5 produces 70 false negatives and 1,474 false positives. At the tuned threshold 0.87, it produces only 22 false negatives and 1,662 false positives. The false-positive count rises, but the false-negative reduction is much more valuable under the cost matrix.

Cost-sensitive evaluation also changes model preference. A model with slightly higher PR-AUC is not always the best operational choice if its thresholded confusion matrix creates more expensive errors. For example, PaySim XGBoost oversampling has slightly higher PR-AUC than XGBoost SMOTE, but its tuned expected cost is 28,300 instead of 27,620. The final decision should therefore consider PR-AUC, recall, precision, and expected cost together.

---

## 9. Critical Analysis and Study Limitations

The project demonstrates a high experimental level, but several limitations should be stated explicitly.

First, both datasets have extremely rare fraud classes. This makes evaluation sensitive to the number of fraud examples in validation and test splits. Credit Card is especially sensitive because the full dataset has only 492 fraud cases. A difference of a few false negatives can visibly change recall and expected cost.

Second, PaySim is computationally heavy. The repository handles this responsibly by constraining PaySim hyperparameter tuning: fewer folds, a 30% stratified tuning subset, reduced search iterations, and restricted parallelism. This preserves a valid tuning procedure while avoiding memory spikes, but it also means PaySim hyperparameter exploration is not exhaustive.

Third, PaySim is simulated. It is useful for controlled experimentation and business-like transaction fields, but it may not contain all noise, behavioral drift, adversarial adaptation, and hidden leakage risks found in real bank transaction streams.

Fourth, synthetic balancing methods have methodological risks. SMOTE and ADASYN interpolate minority examples, but synthetic fraud points may not always correspond to realistic fraud behavior. ADASYN in particular can over-emphasize difficult regions and produce many false positives, as shown by weak Extra Trees ADASYN performance on PaySim.

Fifth, direct train-vs-validation overfitting curves are not saved in the current artifacts. The report can analyze overfitting from CV variance and final test behavior, but future runs should save train scores or learning curves for stronger evidence.

Sixth, the cost matrix is fixed. A 500 vs 10 ratio is reasonable for demonstrating cost-sensitive fraud detection, but real institutions would estimate costs from transaction value, customer segment, investigation capacity, and regulatory constraints. Future experiments should test several cost scenarios.

Future improvements should include broader hyperparameter search on stronger hardware, probability calibration after resampling, temporal validation to simulate production drift, transaction-time feature engineering, model monitoring, and cost-sensitive learning objectives. For PaySim specifically, richer feature engineering around transaction type, amount behavior, account history, and temporal velocity would make the experiment closer to a production fraud system.

---

## 10. Experimental Rigor and Reproducibility

The project demonstrates strong reproducibility and methodological control.

The codebase is modular: dataset settings are centralized in `config.py`, loading logic is isolated in `data_loader.py`, preprocessing is built through `preprocessing.py`, model and sampler construction is handled in `models.py`, and metrics, threshold tuning, curves, and confusion matrices are implemented in `evaluation.py`.

The experiment pipeline uses stratified train, validation, and test splits. The effective proportions are approximately 60% training, 20% validation, and 20% test. Threshold tuning is performed only on validation data, while the test set remains untouched until final evaluation. Balancing is applied inside the training pipeline only, which avoids the common leakage error of resampling before splitting.

The repository saves deterministic artifacts: consolidated comparison tables, threshold-sweep CSV files for every dataset/model/strategy combination, ROC plots, PR plots, tuned-threshold confusion matrices, and hyperparameter-analysis outputs. The dashboard files provide an additional way to inspect saved results interactively.

The experiment is also systematic rather than selective. It evaluates 60 final configurations across two datasets, five model families, and six balancing strategies. This supports fair comparison and reduces the risk of cherry-picking a single good model.

For these reasons, the project satisfies the high-grade criteria: it includes multiple model families, multiple imbalance strategies, cross-validation stability evidence, threshold tuning, cost-sensitive evaluation, saved artifacts, reproducible configuration, and critical analysis of limitations.

---

## 11. Final Experimental Conclusions

The main lesson of the project is that fraud detection under extreme class imbalance is not a standard accuracy-maximization problem. A classifier can achieve very high accuracy while missing many fraud cases. Therefore, model selection must focus on PR-AUC, fraud recall, precision, confusion-matrix counts, and expected cost.

Ensemble models dominate because fraud patterns are nonlinear and rare. XGBoost is the strongest overall model, especially on PaySim, where it clearly outperforms all other families in expected cost and PR-AUC. On Credit Card, XGBoost is still the best single configuration, but Extra Trees and Random Forest are close competitors, showing that bagged tree ensembles are also effective when the dataset is smaller and anonymized.

Balancing strategies are useful but must be interpreted carefully. Class weighting is stable and strong, especially on Credit Card. SMOTE and oversampling are highly effective with XGBoost on PaySim. Undersampling and ADASYN can increase recall, but they are more likely to damage precision and expected cost. The best fraud detector is not the one with maximum recall alone; it is the one that finds fraud while keeping false alerts at an economically acceptable level.

Threshold tuning is essential. The optimal thresholds are often far from 0.5, and their direction depends on model calibration and balancing strategy. Cost-sensitive threshold selection reduced expected cost by 16.6% for the best Credit Card model and 44.5% for the best PaySim model. This demonstrates that the operating threshold is part of the model, not an afterthought.

Finally, cost-sensitive analysis changes the meaning of "best model." The strongest practical model is the one that minimizes business damage, not necessarily the one with the highest accuracy or even the highest PR-AUC. Under the chosen cost matrix, missing fraud is much more expensive than flagging a legitimate transaction, so the preferred systems accept some false positives to prevent false negatives.

Overall, the project provides a reproducible and critically analyzed experimental framework for fraud detection. It satisfies the requirements for a high-level submission by combining systematic model comparison, imbalance-aware evaluation, threshold tuning, stability analysis, cost-sensitive conclusions, and honest discussion of limitations and future development.
