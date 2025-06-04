# 🔍 Efficient Fine-Tuning of Tiny BERT Models with LoRA on AG News
## 🏃‍♂️💨 To run files
- HyperParameterComparison.ipnyb should run fine on google collab using t4 GPU
- FinalPredict_FineTune.py requires `pip install torch transformers datasets peft scikit-learn wandb tabulate`.
- In some cases, you may need to also `pip install --upgrade datasets`.


## 📚 Dataset Used

- **[AG News](https://huggingface.co/datasets/ag_news)** from Hugging Face
- A 4-class news topic classification task with 120,000 training and 7,600 test examples.

---

## 🧠 Models Used

To maintain **efficiency** and stay within **tight GPU constraints**, I used **extremely lightweight transformer models** (~4.4M parameters):

- [`google/bert_uncased_L-2_H-128_A-2`](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) — A mini BERT variant with 2 layers, 128 hidden size, 2 attention heads.
- [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny) — One of the smallest publicly available BERT models. 

These models were chosen for their efficiency on limited GPU availability and potential for strong performance. Rather than relying on large models that can complete AG_news tasks easily, the goal was to push smaller models as far as possible using LoRA.

---
## 🧩 What is LoRA(Low-Rank Adaptation)?

[**LoRA**](https://arxiv.org/abs/2106.09685) is a parameter-efficient fine-tuning technique designed for LLMs. Instead of updating the full weight matrices during training, LoRA **freezes the original model weights** and injects **learnable low-rank matrices** into certain layers (typically attention projections like `query` and `value`).

This allows fine-tuning large models with **dramatically fewer trainable parameters** and **lower GPU memory usage**.

---

### 🔢 Forward Pass Equation

Given a frozen weight matrix `W` (e.g., for a linear layer), LoRA adds a low-rank update using two trainable matrices `A` and `B`:

![Screenshot 2025-06-03 at 3 57 51 PM](https://github.com/user-attachments/assets/d89b6917-9c4e-4d29-a2ea-d77577bb62e8)

Taken Screeshot from [this article.](https://www.determined.ai/blog/lora-parameters). Great read!

Where:
- W: Original weight matrix(frozen)
- A ∈ R^{d_out × r} (updated matrices)
- B ∈ R^{r × d_in} (updated matrices)
- $\alpha$: scaling factor
- \( r \): rank (bottleneck size)

Instead of updating the full weight matrix `W`, LoRA learns two smaller matrices: `A`, `B `

- These form a low-rank approximation of the update: A x B
- This update is scaled by a factor `α / r` and added to the original frozen weights:
- This approach drastically reduces the number of trainable parameters while maintaining expressiveness.
- Example: If d_out = 10 and d_in = 10, number of parameters that need to be updated is 100.
  But with LoRA, if we set r to 2, A contains 10x2 = 20 and B contains 2x10 = 20. Total Parameters needed to update is 40. 
In big picture, LoRA would only update 0.1-1% of model's parameters, drastically decreasing GPU usage/memory

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
- In FinalPredict_FineTune.py
- Fixed LoRA configuration:  
  `rank=16`, `alpha=64`, `dropout=0.1`

#### **Results**

- **BERT-Uncased Tiny:**  
  - Base accuracy: ~17%  
  - **LoRA fine-tuned accuracy: ~87%**  :astonished:
  - **~70% improvement**:rocket:



- **BERT-Tiny:**  
  - Base accuracy: ~24.8%  
  - **LoRA fine-tuned accuracy: ~87%**  
  - → **~62% improvement** :rocket:
 
<img width="1449" alt="screenshots of accuracies" src="https://github.com/user-attachments/assets/27cce484-c47f-4e14-80d7-b2ea6ab206a1" />

<p align="center"><b>Accuracy</b></p>


![Screenshot 2025-06-03 at 4 32 27 PM](https://github.com/user-attachments/assets/bea29ec2-941e-42b5-8cc9-35543e08d89d)
<p align="center"><b>Loss Curve</b></p>


![Screenshot 2025-06-03 at 4 34 12 PM](https://github.com/user-attachments/assets/46270590-bda3-4c99-bfaf-c186ae81cde9)


<p align="center"><b>Predictions were done to visually see improvement of fine-tuned model</b></p>
<img width="1449" alt="Predictions_readme2" src="https://github.com/user-attachments/assets/45dcd482-0864-4d28-b043-b1bb7edae70c" />

---

### 2️⃣ How do different LoRA hyperparameters affect model behavior and convergence efficiency?

#### Setup
- In HyperParameterComparison.ipnyb
- Created a grid of:
  - `r ∈ {8, 16, 32}`
  - `alpha ∈ {256, 512}`
  - Fixed dropout = 0.0 for consistency
- Evaluated convergence time, final accuracy, and training stability.

#### **Results**

##### BERT-Uncased

- Clear pattern: **higher rank and alpha lead to faster convergence**, up to a point.
- Diminishing returns: Performance **plateaus or regresses beyond rank 16**.
- Best trade-off: `rank=16`, `alpha=512` — strong performance and efficient training.

##### BERT-Tiny

- **Unstable behavior** observed at **high rank + low alpha** settings.
- Best setting: `rank=32`, `alpha=512` — stable convergence and good accuracy.

![ConvergenceStepComparisons2](https://github.com/user-attachments/assets/073f472f-fc0b-4d4b-8157-02dfdb5ce548)
---

## 💡 Key Takeaways

- **LoRA works extremely well** for even the smallest transformer models on classification tasks.
- Even **tiny BERTs can match large model performance** on AG News with the right adaptation strategy.
- Training loss and accuracy curves between both google and prajjwal1 LoRA-fine-tuned models looked nearly identical. I wonder if this is something I missed when picking models to train...
- It was very interesting that given a range of rank and alphas, not all high rank/high alpha lead to quickest convergence.
- I need more GPU power!💻 🔥⚡️🚀💪
