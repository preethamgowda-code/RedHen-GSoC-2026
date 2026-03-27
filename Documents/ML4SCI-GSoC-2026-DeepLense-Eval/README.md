# ML4SCI GSoC 2026: Metacognitive DeepLense Evaluation
**Contributor:** Preetham Gowda R | **Proposal ID:** [Your_ID]

## 🎯 Project Focus
Implementation of a **Metacognitive State Vector (MSV)** layer for truth-verification in gravitational lensing classification. This framework mitigates hallucinations in Vision Transformers (ViT) by monitoring latent entropy during inference.

## 🛠️ Infrastructure
This repository is optimized for **HPC-scale** execution:
* **Containerization:** containers/Singularity.def (Apptainer v1.4.5)
* **Orchestration:** slurm/submit_eval.sh (Slurm Job Arrays)
* **Backbone:** ViT-B/16 with MSV Supervisor logic.

## 📈 Benchmarks
* **Track:** DeepLense Multi-Class Classification (CDM vs Axion vs No-Sub)
* **Performance:** 0.91 AUC (Verified via WandB)
