# Detailed Analysis and Methodology: Representational Alignment Between Vision Models and Human Object Similarity

## 1. Project Overview & Significance

**Project Title:** How Far Can Cheap Linear Transforms Close the Human–Model Visual Similarity Gap?

### 1.1 The Alignment Problem
Modern computer vision models (e.g., CLIP, DINOv2, ViT) have achieved remarkable performance on benchmark tasks, yet their internal representational spaces often diverge from human cognitive structures. When humans judge object similarity, they rely on a complex interplay of visual features, semantic meaning, and categorical hierarchies. Vision models, depending on their training objectives (contrastive, self-supervised, supervised), often over-index on low-level visual features or spurious correlations.

### 1.2 Why it Matters Now (2026 Context)
The pursuit of "human-alignment" is no longer just a cognitive science curiosity; it is recognized as a fundamental driver of model robustness and interpretability. As demonstrated by Muttenthaler et al. (2024), models whose representational spaces closely mirror human judgments exhibit superior generalization and Out-of-Distribution (OOD) robustness. Understanding and bridging this gap is currently a flagship topic at premier AI conferences (e.g., NeurIPS UniReps, ICLR Re-Align).

### 1.3 Novelty and Core Hypothesis
While previous work has evaluated individual models against human data, this project introduces a comprehensive benchmarking across a "zoo" of modern, open-weight models (DINOv2, CLIP, SigLIP, ViT, ResNet). 
The central novelty lies in investigating the **universality of the human alignment gap**: Can a single, computationally inexpensive linear transformation, learned to align one model's embedding space with human judgments, transfer effectively across different model architectures? This tests whether different training paradigms produce fundamentally different misalignment patterns or if they all share a common "skew" relative to human cognition.

---

## 2. Detailed Methodology

### 2.1 Dataset Acquisition and Preparation
* **Images:** Utilize the THINGSplus image dataset, containing 1,854 distinct object images (CC0 licensed).
* **Behavioral Data:** Leverage the THINGS behavioral dataset containing 4.7 million human similarity judgments (odd-one-out triplets) collected from over 14,000 participants.
* **Task Structure:** The behavioral data is structured as triplets (A, B, C). The human subjects identified which object was the "odd one out," implicitly providing similarity judgments (e.g., if C is the odd one out, A and B are more similar to each other than either is to C).

### 2.2 Feature Extraction
1.  **Model Zoo:** Initialize pretrained models via Hugging Face and `timm`, including DINOv2 (Self-supervised), CLIP & SigLIP (Multimodal/Contrastive), standard ViT, and ResNet (Supervised).
2.  **Forward Pass:** Pass the 1,854 THINGS images through each frozen backbone to extract high-dimensional embedding vectors.
3.  **Hardware Acceleration:** Given the M2/MPS stack, map tensors to the MPS device for rapid feature extraction. This step will take mere minutes and comfortably fits within 8GB of unified memory.

### 2.3 Baseline (Zero-Shot) Evaluation
Before training any transformations, establish the baseline alignment for each model.
1.  **Triplet Accuracy:** For every human triplet (A, B, C) where C is the odd-one-out, calculate the cosine similarity between the embeddings of the three images.
2.  **Decision Rule:** The model's prediction is correct if the cosine similarity $S(A, B)$ is greater than both $S(A, C)$ and $S(B, C)$.
3.  **Metric:** Compute the overall zero-shot odd-one-out accuracy across the 4.7 million triplets.

### 2.4 Learning the Linear Transformation (Probing)
To improve alignment, learn a "cheap" transformation matrix $W$.
1.  **Objective:** Transform the original embeddings $X$ to $X' = XW$ such that the triplet accuracy in the new space is maximized.
2.  **Optimization Paradigm:** Use an $L_2$-regularized linear probe. This can be framed as metric learning or optimized using a triplet margin loss. 
    * *Loss Formulation:* Minimize $L = \max(0, 	ext{margin} + D(AW, BW) - D(AW, CW)) + \lambda ||W||_2$, where $D$ is a distance metric (e.g., Euclidean or cosine distance).
3.  **Validation:** Split the human triplets into train, validation, and held-out test sets to ensure the linear transform generalizes to unseen object combinations.

### 2.5 Cross-Model Transferability Testing
This is the core analytical contribution of the project.
1.  **Dimensionality Alignment:** If models have different embedding dimensions, apply PCA to project them into a shared lower-dimensional space (e.g., $d=512$) before learning transforms, or test transferability only between models of identical hidden dimensions.
2.  **Transfer Protocol:** Apply the transformation matrix $W_{model\_A}$ (learned on Model A) to the embeddings of Model B. 
3.  **Evaluation:** Measure the odd-one-out accuracy of Model B using $W_{model\_A}$. 
4.  **Correlation Analysis:** Correlate the baseline alignment and the post-transform alignment with model scale (parameter count) and training objective to identify macro-trends in the model zoo.

### 2.6 Error Analysis and Semantic Profiling
1.  **Isolating Misalignments:** Identify subsets of triplets where the linearly transformed models still fail to match human judgments.
2.  **Semantic Clustering:** Cross-reference these failures with the THINGS semantic metadata. Investigate hypotheses from prior literature (e.g., models failing on highly contextual or abstract concepts like "sports" or "royal" items compared to concrete visual shape matching).
3.  **Representational Similarity Analysis (RSA):** Use `rsatoolbox` to construct and compare Representational Dissimilarity Matrices (RDMs) of the models before and after transformation against the human RDM.

---

## 3. Technical Implementation Profile

* **Compute Requirements:** Extremely light. The 4.7M triplets are represented as integer indices, requiring minimal RAM. The entire pipeline avoids backpropagation through the deep networks, utilizing only frozen features. It is perfectly scoped for local execution on an M2 Mac.
* **Tech Stack:** * `PyTorch` (with `mps` backend) for deep feature extraction and tensor operations.
    * `Hugging Face transformers` / `timm` for managing the model zoo.
    * `scikit-learn` for PCA, cross-validation, and potentially formulating the linear probe as a classifier.
    * `rsatoolbox` for RSA evaluation.
* **Timeline:** 1.5 - 2.5 months (part-time). 
    * *Weeks 1-2:* Data prep and zero-shot baseline extraction.
    * *Weeks 3-5:* Linear probe implementation and tuning.
    * *Weeks 6-8:* Cross-model transfer experiments, RSA, and paper drafting.

---

## 4. Extensions and Future Work

For subsequent iterations or workshop expansions, the project can naturally evolve into the following domains:
1.  **Neural Alignment:** Incorporate the THINGS-fMRI and THINGS-MEG datasets. Compare behavioral alignment (odd-one-out) with neural alignment (brain activity patterns), investigating whether aligning to behavior simultaneously aligns to brain representations.
2.  **Geometric Exploration:** Test linear alignment mapping into hyperbolic space, evaluating if human cognitive structures are better represented on manifolds with negative curvature (matching hierarchical/tree-like human categorization).
3.  **Generative Models:** Extend the benchmark to evaluate the representational spaces of the perceptual backbones of diffusion models (e.g., Stable Diffusion's VAE or UNet bottlenecks) against human visual similarity.
