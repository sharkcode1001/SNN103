# SNN103 — Neuromorphic Rank-Order Spiking Neural Network

[![.NET 10](https://img.shields.io/badge/.NET-10.0-512BD4)](https://dotnet.microsoft.com/)
[![CI Build & Test](https://github.com/sharkcode1001/SNN103/actions/workflows/ci.yml/badge.svg)](https://github.com/sharkcode1001/SNN103/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--2047--9345-green.svg)](https://orcid.org/0009-0007-2047-9345)

**Author:** Erisnel Lora Sugve  
**Affiliation:** Independent Researcher, Santiago de Cuba, Cuba  
**Contact:** `lorasugve@gmail.com` | **ORCID:** [0009-0007-2047-9345](https://orcid.org/0009-0007-2047-9345)  
**Paper (English):** [SNN103-article.md](SNN103-article.md) | **Artículo (Español):** [SNN103-articulo-es.md](SNN103-articulo-es.md)  

---

## English Overview

**SNN103** is an event-driven, rank-order Spiking Neural Network (SNN) implemented in .NET 10. Inspired by the **B103 line code** (where $n$ parallel pulses carry $n!$ arrival permutations), this platform proves that **first-spike arrival order** conveys primary discriminative information rather than secondary cosmetic timing.

### Core Scientific Findings (10-Class Digit Benchmark across 5 Seeds):
* **SNN Order Features ($1/\text{rank}$):** **88.1% ± 1.1%** test accuracy.
* **SNN Binary Features (fired / did not fire):** **70.7% ± 2.3%** test accuracy.
* **Rank-Order Margin:** **+17.4 percentage points** with the identical physical layer and seeds.
* **Event Efficiency:** **~168.7 spikes** per decision (~14× fewer dynamic events than the 2368 continuous Multiply-Accumulate operations of an MLP).
* **Noise Immunity:** Modular 4-neuron subregions achieve **98.9%** exact permutation retention under temporal jitter, whereas monolithic 8-neuron groups degrade to **62.3%**.

---

## Resumen en Español

**SNN103** es una plataforma de redes neuronales de impulsos (*Spiking Neural Networks*) orientada a eventos. Basándose en la hipótesis de que $n$ pulsos paralelos transportan hasta $n!$ zonas o permutaciones de llegada (código de línea B103), el sistema demuestra que el **orden de llegada de los spikes** aporta **+17.4 puntos porcentuales** sobre la codificación binaria convencional con apenas **~169 impulsos por inferencia**.

---

## Architecture & Principles / Arquitectura

1. **Burst-Driven Excitation:** Inputs arrive in periodic bursts, ensuring non-simultaneous threshold crossings and eliminating degenerate ties.
2. **Commitment Dynamics:** Neurons require 4 consecutive steps above threshold before emitting a spike, suppressing high-frequency channel jitter.
3. **Lateral Inhibition by Threshold Jump:** The earliest firing neuron dynamically raises competing thresholds (+0.6), enforcing wide temporal margins.
4. **Modular Subregions ($4 \times 4$):** Spatial partitioning into small 4-neuron groups maximizes noise tolerance under temporal phase noise.
5. **Supervised Readout:** Linear perceptron operating on $1/\text{rank}$ features.

---

## Quick Start & CLI Usage / Ejecución

Clone the repository and run via .NET 10 SDK:

```bash
# Build project
dotnet build

# 1. Run core 10-class ablation experiment (5 seeds)
dotnet run --project SNN103 -- --benchmark

# 2. Run visual 4-class pilot experiment (confusion matrix + raster plot)
dotnet run --project SNN103 -- --pilot

# 3. Run noise robustness demonstration (Section 4.2)
dotnet run --project SNN103 -- --noise

# 4. Run entire suite and export publication-ready figures to plots/
dotnet run --project SNN103 -- --all

# Launch in English or Spanish directly:
dotnet run --project SNN103 -- --lang en
dotnet run --project SNN103 -- --lang es
```

### Interactive Menu / Menú Interactivo
Run without arguments to enter the bilingual interactive terminal:
```bash
dotnet run --project SNN103
```
You can toggle between English and Spanish on the fly by pressing `[L]`.

---

## Exported Figures / Figuras Generadas (`plots/`)

* `benchmark_10classes_ablation.png`: 10-class ablation bar chart with error bars (Order vs Binary vs MLP).
* `confusion_matrix.png`: Test set confusion matrix heatmap.
* `raster_plot.png`: Spike raster plot demonstrating burst-driven rank-order firing.
* `snn_vs_mlp_comparison.png`: Accuracy vs dynamic computational cost tradeoff.

---

## Citation & License

This research and codebase are published under the **GNU General Public License v3.0 (GPLv3)**. See [LICENSE](LICENSE) for details.

* **Open Source / Academic Research:** You are free to copy, modify, and distribute this software under the terms of the GPLv3 (all derivatives must remain open source under GPLv3).
* **Commercial / Closed-Source Licensing:** For proprietary integration or commercial use without GPLv3 copyleft restrictions, contact: `lorasugve@gmail.com`.

```bibtex
@article{sugve2026snn103,
  title={Learning to Use Spike Order: Rank-Order Dictionaries Beat Binary Features by 17 Points at a Fraction of the Events},
  author={Lora Sugve, Erisnel},
  journal={Technical Preprint},
  year={2026},
  url={https://github.com/sharkcode1001/SNN103}
}
```
