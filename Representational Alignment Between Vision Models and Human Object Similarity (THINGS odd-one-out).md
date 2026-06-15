# Representational Alignment Between Vision Models and Human Object Similarity (THINGS odd-one-out)

## Direct answer

This is a well-scoped, publication-feasible project: you can implement it with modest compute, obtain clear quantitative results on the THINGS odd-one-out benchmark, and explore a genuinely interesting question about whether a single cheap linear transform can partially close the human–model similarity gap across multiple vision backbones. The work fits strongly with current UniReps/Re-Align themes around human-aligned representations and could realistically yield a NeurIPS/ICLR workshop paper plus a compact cognitive-science style writeup (CCN/CogSci), especially if you add even a small neural RDM or hyperbolic-extension component.

## Big-picture motivation

Human alignment of internal representations has emerged as a key ingredient for robustness, interpretability, and transfer in modern vision and multimodal models. Recent work by Muttenthaler et al. ("Aligning Machine and Human Visual Representations across Abstraction Levels", 2024) shows that representations explicitly optimized to match human similarity structure improve generalization and out-of-distribution robustness on diverse benchmarks. In parallel, the THINGS initiative has provided large-scale behavioral similarity data for thousands of object images, including odd-one-out triplets that operationalize human object similarity structure at scale.

Against this backdrop, your project targets a focused, tractable question: given frozen embeddings from standard pretrained vision models (DINOv2, CLIP, SigLIP, ViT, ResNet, etc.), how far can we push human-model similarity alignment by learning *only* light linear transforms, and do such transforms transfer across architectures? This isolates the "cheap alignment" regime that is particularly attractive for practical deployment and model debugging: we want to know whether the misalignment is mostly due to simple linear distortions of an essentially human-like latent space, or whether it reflects deeper structural differences that cannot be fixed without retraining.

## Key ingredients and prior work

### THINGS dataset and odd-one-out behavior

The THINGS behavioral dataset (Hebart et al., 2023, eLife 82580) provides around 4.7 million human similarity judgments collected from over 12,000 participants for 1,854 object images, using a triplet-based odd-one-out task. On each trial, participants see three images and choose the odd one out; aggregating responses over many subjects yields robust triplet-level similarity statistics and allows derivation of representational dissimilarity matrices (RDMs) that approximate human object similarity structure.

For your project, these odd-one-out triplets become the core supervision signal: given frozen model embeddings for each of the 1,854 images, your model predicts which of the three items is the odd one out using a similarity rule (e.g., cosine distances), and you score how often this choice matches majority human behavior. This is a direct, interpretable metric of representational alignment that avoids arbitrary thresholding or downstream task choices.

### Vision backbones and representational comparisons

Self-supervised and contrastive models such as DINOv2 and CLIP have been shown to learn representations that transfer well across tasks and in some cases correlate with neural and behavioral data. However, these correlations are far from perfect, and different architectures and training objectives can yield quite different alignment patterns (e.g., supervised ResNets vs. self-supervised ViTs).

Muttenthaler et al. 2022/2024 introduce a framework for aligning model and human/neural representations using linear transformations and report that optimized maps can significantly improve correspondence across multiple abstraction levels (from low-level perceptual similarity to high-level semantic categories). Their codebase shows how to compute RDMs, fit linear maps, and evaluate alignment with human fMRI/MEG and behavioral data, which you can adapt for your THINGS odd-one-out pipeline.

## Project structure and methods

### 1. Data preparation and feature extraction

You will first obtain the THINGS image set (CC0 variants via THINGSplus) corresponding to the 1,854 core object images studied in the behavioral work. For each image, you pass it through a set of frozen pretrained backbones: for example, DINOv2 ViT variants, CLIP ViT-B/16 and ViT-L/14, SigLIP models, standard supervised ViTs from timm, and classic ResNet-50 or ResNet-101.

Using PyTorch (with MPS acceleration on Apple Silicon), Hugging Face Transformers, and timm, you extract a single embedding vector per image from a specified layer (typically the CLS token or pooled final-layer features). For each model, this yields an embedding matrix of size 1,854 by d, where d is the model’s feature dimensionality. The compute and memory requirements are modest because the image set is small, and feature extraction is a one-off operation.

### 2. Zero-shot odd-one-out evaluation

In the zero-shot phase, you directly use the raw model embeddings to predict odd-one-out choices for each triplet without any alignment training. Given embeddings for items i, j, and k, you compute pairwise cosine similarities; you then adopt a simple rule such as declaring as the odd one out the item with the largest average distance to the other two. Comparing this prediction to majority human choices across all triplets yields a baseline accuracy score for each backbone.

This step establishes the unaligned human-model similarity gap: you will likely see that some models (e.g., CLIP or DINOv2) perform better than supervised ResNets but still fall significantly short of human consistency levels, especially for higher-level semantic distinctions.

### 3. Learning linear transforms to improve alignment

Next, you learn an L2-regularized linear transform W for each backbone such that transformed embeddings Wx yield higher odd-one-out accuracy on a training subset of triplets. This can be formulated as a multinomial logistic regression or hinge-loss ranking problem, where the model scores candidate odd-one-out choices based on distances in the transformed feature space and is trained to prefer human-consistent choices.

You will split the 4.7M triplets into train, validation, and test sets at the triplet level (or at the image-pair level to avoid leakage) and optimize W using standard scikit-learn or PyTorch optimizers with early stopping based on validation accuracy. The transform remains linear and relatively low-parameter compared to the backbone, ensuring training remains light-weight and unlikely to overfit excessively given the scale of behavioral data.

### 4. Cross-model transfer of alignment transforms

The central novelty of your project is to test whether a linear transform learned on one model transfers to others. To do this, you must map embeddings from different architectures into a common space or at least ensure consistent dimensionality. One approach is to learn W on a chosen reference model (e.g., CLIP ViT-B/16) and then fit a separate linear map M that projects other models’ embeddings into the CLIP space using only image-level correspondences (i.e., same image, different model).

You can then apply the fixed W (trained to align CLIP to humans) on the projected embeddings Mx from other backbones and evaluate odd-one-out accuracy. If performance improves significantly across models relative to their unaligned baselines, this suggests that the human alignment gap has a partially universal linear structure shared across architectures. If improvement is limited or highly model-specific, it implies deeper architectural or training-objective-specific differences.

### 5. Relating alignment to model properties

Because you will benchmark a zoo of models differing in scale (small vs large ViTs), training data (ImageNet vs LAION vs other corpora), and objectives (self-supervised vs contrastive vs supervised classification), you can correlate various metrics with human alignment scores. For example, you can ask whether models trained on more diverse web-scale data (e.g., CLIP, SigLIP) show higher baseline odd-one-out accuracy and require smaller linear adjustments to match humans, compared to purely supervised ImageNet models.

Similarly, you can explore whether model depth, parameter count, or patch size systematically affect alignment after controlling for objective type. This helps answer whether scaling up standard objectives is sufficient for alignment, or whether particular training signals (e.g., image-text contrast) are uniquely helpful for approximating human similarity structure.

### 6. Semantic error analysis and misaligned dimensions

Beyond global accuracy metrics, you can analyze which semantic categories and dimensions remain most misaligned even after linear alignment. Prior work found that models often struggle with high-level conceptual features like social roles, affective valence, and abstract semantic relationships, even when they capture mid-level visual features well.

Using the THINGS ontology and category labels (e.g., animals, tools, vehicles, foods) and any available attribute annotations, you can group triplets by semantic type and compare pre- and post-alignment performance per group. You might find, for instance, that sports-related or social-status-related concepts remain challenging, echoing Muttenthaler et al.’s observations about misalignment in semantic domains such as sports and royal concepts. This gives you interpretable qualitative insight into what alignment can and cannot do in the linear regime.

## Practical feasibility and compute

Because you are working with only 1,854 images and no end-to-end finetuning of backbones, the compute requirements are very light. Extracting embeddings for this many images across a handful of backbones is comfortably feasible on an M2 Mac with MPS acceleration and even tolerable on CPU if necessary. The behavioral dataset’s 4.7M triplets are stored as integer indices and can be streamed or batched with modest memory overhead; training linear models on this scale is standard practice in modern machine learning pipelines.

Your planned stack—PyTorch for model inference, Hugging Face/timm for backbone access, scikit-learn for linear probes, numpy for array handling, and rsatoolbox for RDM computations—matches best practices in this area and aligns with existing reference implementations from Muttenthaler et al. and the THINGS initiative.

## Publication potential and natural extensions

Given the timeliness of human-model representational alignment and the existence of active venues like the NeurIPS UniReps and ICLR Re-Align workshops, a carefully executed study on THINGS odd-one-out with a broad model zoo and cross-model linear transfer experiments stands a realistic chance of workshop acceptance. The behavioral- and representation-focused nature of the work also fits cognitive and computational neuroscience venues such as CCN or CogSci, especially if you add even a modest extension with THINGS-fMRI/MEG representational dissimilarity matrices to compare neural vs behavioral alignment.

Potential extensions include exploring hyperbolic embeddings for better hierarchical semantic representation, as suggested by recent work on evaluating human-machine alignment in hyperbolic space, and testing whether diffusion model features provide different alignment patterns than discriminative backbones. These are optional but could provide additional novelty and depth if time allows.

## How this fits your profile

For you as an early-career engineer with strong ML and systems skills, this project offers an excellent low-barrier entry into human-aligned representation research. It leverages tools and libraries you already know (PyTorch, Hugging Face, scikit-learn), demands careful experimental design and analysis rather than heavy engineering, and connects directly to current open questions about the universality of alignment gaps across architectures.

Over 1.5–2.5 months of part-time work, you can reasonably expect to:

- Implement the full THINGS odd-one-out pipeline for several backbones.
- Quantify zero-shot and linearly aligned human-model similarity gaps.
- Test cross-model transfer of alignment transforms.
- Perform semantic error analysis and, optionally, neural RDM or hyperbolic extensions.

If executed cleanly with thoughtful analysis and clear writing, this should bolster your profile for PhD applications to programs focused on representation learning, cognitive computational neuroscience, and AI safety/robustness.