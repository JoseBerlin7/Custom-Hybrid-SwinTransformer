**Key Idea**

Traditional Conv/Transformer hybrids run both branches always → wasted compute.

We instead do:

    Conv   ↘
            Gate → Selected branch → Output
    Trans ↗


The gate decides per sample whether to run:

    Option	Meaning
    1	Conv only (local features)
    2	Transformer only (global features)
    3	Conv + Transformer (fusion)

Decision is learned end-to-end.

How Gumbel-Softmax gating works

During training:

1. Soft decisions drive gradients

2. Hard argmax decisions control forward routing

        Hard for inference realism
        Soft for gradient learning
        Both are synchronized via straight-through estimator

Example:

soft = [0.3, 0.5, 0.2]  (Conv, Trans, Both)
hard = [0,   0,   1]   (model routed to “Both”)


Even though “Both” ran in forward pass,
Conv & Transformer still get partial gradients via soft weights.

This avoids dead branches.

Temperature is annealed so:

**Stage	Behavior:**
Early	Exploration (soft ≠ hard)
Mid	Sharper decisions
Late	soft ≈ hard (deterministic)
Inference:	pure hard argmax routing

**Benefits:**
Feature	Benefit
Dynamic routing	Skip unnecessary heavy layers
Conv + Transformer choice	Task-adaptive feature fusion
Straight-through gumbel	Stable learning + discrete inference
Soft gradients	All branches train (no dead paths)
Modular	Plug into U-Net, ResNet, SegFormer, etc

**Code Structure:**

    /DHVN
      /models
          conv_block.py
          transformer_branch.py
          hybrid_block.py   
          dhvn_classification.py
          helpers.py
        
    data.py
    train.py
    requirements.py
    utils.py

**Example Outputs:**

Gate Heatmaps (which layer chooses Conv vs Transformer)

FLOPs reduction reports

Per-sample routing visualizations

Coming soon: Routing visualization notebook

**Training:**
1. Customize train.py according to your system

       python train.py --dataset imagenet

**Inference:**

During inference we run only the chosen branch:

choice = torch.argmax(gate, dim=1)
# no soft mixing


Faster, deterministic, no training overhead.


**Related Concepts:**
Method	Similarity
Gumbel-Softmax	Differentiable discrete decisions
Mixture of Experts (MoE)	Learn routing between networks
Dynamic Conv / Dynamic ViT	Adaptive compute per sample
U-Net / SegFormer	Backbone this integrates with

**Citation:**

If you use this repo, please cite:

    @misc{dhvn2025,
      title={Dynamic Hybrid Vision Network: Conv-Transformer Adaptive Routing},
      author={Jose Berlin},
      year={2025},
      note={https://github.com/JoseBerlin7/DHVN.git}
    }

Why this approach?

**Traditional hybrid models:**

1. Always compute both Conv & Transformer
2. Waste compute when one is enough

Our method:

1. Lets model decide dynamically
2. Gets Conv speed + Transformer long-range power
3. Better efficiency & flexibility

Bottom line: Stop forcing hybrid, let the network choose.
