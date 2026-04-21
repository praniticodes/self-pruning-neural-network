# 🧠 Self-Pruning Neural Network

### (Tredence AI Engineering Case Study)

---
## 🔗 Run on Google Colab
[Open Notebook](https://colab.research.google.com/drive/1HzHNQjclTzJEgln88_XtagMZ1LhBaCQ6?usp=sharing)
## 📌 Overview

This project implements a **self-pruning neural network** that dynamically learns which connections to remove during training.

Unlike traditional pruning (post-training), this model integrates pruning into the learning process using **learnable gate parameters**. Each weight is associated with a gate that determines whether the connection is active or pruned.

---

## ⚙️ Methodology

### 🔹 Prunable Layer

* Custom `PrunableLinear` layer implemented from scratch
* Each weight has a corresponding **gate score parameter**
* Gates are computed using a **sigmoid function** to keep values between 0 and 1

### 🔹 Dynamic Pruning Mechanism

* Hard thresholding is applied to enforce pruning
* Straight-through estimator is used to maintain gradient flow
* Effective weights are computed as:

```
pruned_weight = weight × gate
```

---

## 🧠 Loss Function

Total Loss = Classification Loss + λ × Sparsity Loss

* **Classification Loss:** CrossEntropyLoss
* **Sparsity Loss:** Mean of gate activations (L1-style regularization)
* **λ (lambda):** Controls the strength of pruning

---

## 📊 Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 1      | 54.38%   | 21.61%   |
| 50     | 48.68%   | 44.26%   |
| 10000  | 10.00%   | 45.40%   |

---

## 📈 Observations

* Increasing λ increases sparsity due to stronger regularization pressure
* At **λ = 1**, the model retains most connections, achieving higher accuracy with limited pruning
* At **λ = 50**, the model achieves the best balance, pruning nearly half the weights while maintaining reasonable performance
* At **λ = 10000**, the sparsity term dominates the loss, forcing most gates toward zero and leading to model collapse (random accuracy ~10%)

---

## ⚠️ Key Insight

This experiment highlights an important property of self-pruning networks:

* Moderate regularization encourages meaningful pruning
* Excessive regularization leads to **over-pruning and loss of learning capacity**
* There exists an optimal λ range that balances efficiency and accuracy

---

## 📦 Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib

---

## 🚀 Key Highlights

* Custom neural layer design
* Learnable pruning mechanism
* Dynamic architecture adaptation during training
* Implementation of straight-through estimator
* Clear sparsity vs accuracy trade-off analysis

---

## 📊 Visualization

The distribution of gate values shows:

* A spike near **0 → pruned connections**
* A cluster of higher values → important weights retained by the model

---

## ▶️ How to Run

1. Open the notebook:

   ```
   self_pruning_network.ipynb
   ```
2. Run all cells in Google Colab
3. Modify λ values to observe pruning behavior

---

## 🔗 (Optional) Colab Link

Add your Colab notebook link here

---

## 📌 Conclusion

This project demonstrates how neural networks can **learn to prune themselves during training** by identifying and removing less important connections.

It also highlights the importance of **careful regularization tuning**, as overly aggressive pruning can significantly degrade performance.

---

## 🙌 Author

**Praniti Sethi**

---
