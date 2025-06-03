# 🔍 Efficient Fine-Tuning of Tiny BERT Models with LoRA on AG News

## 📚 Dataset Used

- **[AG News](https://huggingface.co/datasets/ag_news)** from Hugging Face
- A 4-class news topic classification task with 120,000 training and 7,600 test examples.

---

## 🧠 Models Used

To maintain **efficiency** and stay within **tight GPU constraints**, I used **extremely lightweight transformer models** (~4.4M parameters):

- [`google/bert_uncased_L-2_H-128_A-2`](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) — A mini BERT variant with 2 layers, 128 hidden size, 2 attention heads.
- [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny) — One of the smallest publicly available BERT models. 

These models were intentionally chosen to explore **how much performance improvement** can be gained from **LoRA-based fine-tuning**, even when starting with minimal model capacity. Large models might perform well by default on AG News, but my goal was to see how far we could push these constrained models with efficient adaptation.

---
## 🧩 What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique designed for large language models (LLMs). Instead of updating the full weight matrices during training (which is memory and compute-intensive), LoRA **freezes the original model weights** and injects **learnable low-rank matrices** into certain layers (typically attention projections like `query` and `value`).

This allows fine-tuning large models with **dramatically fewer trainable parameters** and **lower GPU memory usage**.

---

### 🔢 Forward Pass Equation

Given a frozen weight matrix `W` (e.g., for a linear layer), LoRA adds a low-rank update using two trainable matrices `A` and `B`:

![Screenshot 2025-06-03 at 3 57 51 PM](https://github.com/user-attachments/assets/d89b6917-9c4e-4d29-a2ea-d77577bb62e8)
https://www.determined.ai/blog/lora-parameters

Where:
- W: Original weight matrix(frozen)
- A ∈ R^{d_out × r}
- B ∈ R^{r × d_in}
- $\alpha$: scaling factor
- \( r \): rank (bottleneck size)

This low-rank matrix product approximates a full-rank update to `W` while keeping parameter count low.

---

### 📐 What Does Rank Do?

- `r` is the dimensionality of the low-rank approximation
- Controls the **capacity** of the LoRA module
- Higher `r` means more trainable parameters. Specifically, the number of trainable parameters per layer is:

  `(d_out x r) + (r x d_in)`

  where:  
  - `d_out` = output dimension of the layer  
  - `d_in` = input dimension of the layer  

- Tradeoff: more expressive power vs. more memory and compute

Example:
- If `r = 4`, the update is small and efficient
- If `r = 64`, the update is much larger — closer to full fine-tuning

---

### 🧪 What Does Alpha Do?

- `alpha` is a **scaling factor** that controls the **magnitude** of the low-rank update
- It is **divided by `r`** to normalize updates across different ranks
- A larger `alpha` allows updates of A and B to have a stronger influence on model predictions

---

### 🧠 Summary: What Gets Trained?

With LoRA:
- **Original weights (`W`) are frozen**
- Only the **low-rank matrices `A` and `B`** are updated during training
- Efficient fine-tuning by using only ~0.1-1% of parameters with nearly full performance

---

## 🧪 Research Questions Explored

### 1️⃣ What is the maximum accuracy I can achieve using LoRA fine-tuning under GPU limitations?

#### Setup
- This is done in FinalPredict_FineTune.py
- 
- Fixed LoRA configuration:  
  `rank=16`, `alpha=64`, `dropout=0.1`

#### Results

- **BERT-Uncased Tiny:**  
  - Base accuracy: ~17%  
  - **LoRA fine-tuned accuracy: ~87%**  
  - → **~70% improvement**
  
- **BERT-Tiny:**  
  - Base accuracy: ~24.8%  
  - **LoRA fine-tuned accuracy: ~87%**  
  - → **~62% improvement**

#### Notes

- Training loss curves between base and LoRA-fine-tuned models looked nearly identical.
- Indicates that **loss is not always an indicator of learning** — LoRA shifts representations effectively even when base model "appears" to converge.
- Given more GPU time, accuracy could likely be improved further by scaling model size or experimenting with deeper architectures.


<img width="573" alt="screenshots of accuracies" src="https://github.com/user-attachments/assets/27cce484-c47f-4e14-80d7-b2ea6ab206a1" />
![bert_uncased_trainingLoss](https://github.com/user-attachments/assets/10adec8f-0871-45b2-8997-7e9b80d80a74)
![bert-tiny_trainingLoss](https://github.com/user-attachments/assets/eebe0128-b504-4266-84f3-d17703121f2c)
<img width="1454" alt="Predictions" src="https://github.com/user-attachments/assets/dad10455-5546-4dc3-af2d-65edbe1f8432" />
---

### 2️⃣ How do different LoRA hyperparameters affect model behavior and convergence efficiency?

#### Setup
- This is done in HyperParameterComparison.ipnyb
- Created a grid of:
  - `r ∈ {4, 16, 32}`
  - `alpha ∈ {64, 256, 512}`
  - Fixed dropout = 0.0 for consistency
- Evaluated convergence time, final accuracy, and training stability.

#### Findings

##### BERT-Uncased

- Clear pattern: **higher rank and alpha lead to faster convergence**, up to a point.
- Diminishing returns: Performance **plateaus or regresses beyond rank 16**.
- Best trade-off: `rank=16`, `alpha=512` — strong performance and efficient training.

##### BERT-Tiny

- **Unstable behavior** observed at **high rank + low alpha** settings.
  - Suggests **optimization difficulty due to over-parameterization** relative to base model capacity.
- Best setting: `rank=32`, `alpha=512` — stable convergence and good accuracy.

![ConvergenceStepComparisons](https://github.com/user-attachments/assets/14b55653-ebcf-4d03-a530-86135350b2bd)

---

## ⚙️ Efficiency vs Accuracy Tradeoff

| Model         | Base Accuracy | Best LoRA Accuracy | Trainable Params (LoRA) | Notes                             |
|---------------|----------------|---------------------|---------------------------|-----------------------------------|
| BERT-Uncased  | ~25%           | ~87%                | Low                       | Rapid improvement with LoRA       |
| BERT-Tiny     | ~25%           | ~88%                | Low                       | Sensitive to r/alpha combinations |

- **LoRA enables massive accuracy gains** with very few additional parameters.
- However, **higher ranks increase training time and memory** — diminishing returns must be weighed carefully.


---

## 💡 Key Takeaways

- **LoRA works extremely well** for even the smallest transformer models on classification tasks.
- Even **tiny BERTs can match large model performance** on AG News with the right adaptation strategy.
- **Training dynamics vary drastically** between models — smaller models are more sensitive to LoRA hyperparameters.
- **GPU constraints** force tradeoffs between coverage (number of experiments) and
