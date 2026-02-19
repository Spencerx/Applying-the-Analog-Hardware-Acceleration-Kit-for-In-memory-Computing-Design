# Applying the Analog Hardware Acceleration Kit for In-memory Computing Design

![Python](https://img.shields.io/badge/python-3.6%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
![Stars](https://img.shields.io/github/stars/HungYangChang/Applying-the-Analog-Hardware-Acceleration-Kit-for-In-memory-Computing-Design)

> Simulating and optimizing neural network training on analog resistive processing units (RPUs) using IBM's Analog Hardware Acceleration Kit.

**[Full Report (PDF)](541_Final_report%20.pdf)** · **[Notebook](ECSE_541_Project_.ipynb)**

---

## Problem

Training large deep neural networks is time-consuming and computationally expensive. The Von Neumann bottleneck — constant data movement between storage and computation — prevents real-time, energy-efficient computation on traditional digital hardware.

## Solution

We use IBM's [Analog Hardware Acceleration Kit (AIHWKit)](https://github.com/IBM/aihwkit) to simulate Resistive Processing Units (RPUs) that represent weights and biases directly in analog memory. This eliminates the data movement bottleneck and enables orders-of-magnitude improvements in training efficiency.

## Architecture

```
┌────────────────────────────────────────────────┐
│          Analog Neural Network Training        │
├────────────────────────────────────────────────┤
│                                                │
│   Input (MNIST)                                │
│       │                                        │
│       ▼                                        │
│   ┌──────────────────────────────────────┐     │
│   │   Analog FCNN (RPU-based weights)    │     │
│   │                                      │     │
│   │   Layer 1: 784 → 256 (analog tiles)  │     │
│   │   Layer 2: 256 → 128 (analog tiles)  │     │
│   │   Layer 3: 128 → 10  (analog tiles)  │     │
│   │                                      │     │
│   │   Noise model: SNR = 34 dB           │     │
│   └──────────────────────────────────────┘     │
│       │                                        │
│       ▼                                        │
│   Classification Output (100% accuracy)        │
│                                                │
└────────────────────────────────────────────────┘
```

### Key Design Decisions

- **RPU simulation over real hardware:** IBM's AIHWKit provides realistic non-ideality modeling without requiring physical analog chips.
- **Hyperparameter optimization under noise:** Systematic tuning of activation functions, batch sizes, and learning rates to maintain accuracy despite analog noise.

## Stack

| Layer | Technology |
|-------|-----------|
| Framework | IBM AIHWKit, PyTorch |
| Compute | Google Colab (GPU) |
| Dataset | MNIST |
| Language | Python 3.6+ |

## Getting Started

```bash
# Option 1: Run on Google Colab (recommended)
# Upload ECSE_541_Project_.ipynb to Colab, select GPU runtime

# Option 2: Local setup
pip install aihwkit torch torchvision
jupyter notebook ECSE_541_Project_.ipynb
```

## Results

| Metric | Value | Context |
|--------|-------|---------|
| Test accuracy (noise-free) | **100%** | Analog FCNN on MNIST |
| Test accuracy (SNR 34 dB) | **100%** | Robust to realistic analog noise |
| Minimal architecture | **784-128-10** | 95% accuracy with only 3 layers |
| Accuracy drop (minimal arch) | **3%** | Compared to deeper networks |

## Known Limitations

- Evaluated only on MNIST — more complex datasets (CIFAR-10, ImageNet) would better demonstrate scalability.
- Noise model uses fixed SNR; real analog devices have more complex non-ideality profiles.
- Single-architecture exploration; Neural Architecture Search could further optimize the analog-aware design.

## Related Publication

> **AI Hardware Acceleration with Analog Memory: Micro-architectures for Low Energy at High Speed**
> _IBM Journal of Research and Development_
> H.-Y. Chang, G.W. Burr, P. Narayanan, S. Ambrogio et al.

## License

MIT — see [LICENSE](./LICENSE) for details.

---

Built by [Hung-Yang (James) Chang](https://github.com/HungYangChang) · McGill University / IBM Research
