# ROVER_MRI

This repository contains the implementation of **Rotating-view super-resolution (ROVER)-MRI reconstruction**, developed based on two ISMRM abstracts:

- *Rapid Whole Brain 180µm Mesoscale In-vivo T2w Imaging*, ISMRM 2025 (oral)
- *Rotating-view super-resolution (ROVER)-MRI reconstruction using tailored Implicit Neural Network*, ISMRM 2024 (oral power pitch)

## ✨ Highlights

- Achieves 180 µm isotropic resolution T2w MRI in ~17 minutes
- Leverages 8 rotated views and 5× super-resolution
- Uses multi-resolution hash encoding and implicit neural fields
- Validated on ex-vivo and in-vivo datasets

## 📂 Directory Structure

```
ROVER_MRI/
├── configs/              # Experiment and training configurations
├── fda/                  # Domain adaptation and feature alignment modules
├── bumonkey_hash_v8_*.py # Main training & testing scripts
├── util_args_*.py        # Argument parsers for different experiments
├── utils.py              # Common utility functions
├── README.md             # Project documentation (this file)
```

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 1.11 (tested with 1.13.1 + CUDA 11.7)
- NumPy, SciPy, nibabel, tqdm, etc.

Install dependencies (optional):

```bash
pip install -r requirements.txt
```

### 🔍 Demo

#### Training

Modify the config file in `configs/` and run:

```bash
python bumonkey_hash_v8_noastype.py --config configs/your_config.yaml
```

#### Test

```bash
python bumonkey_hash_v8_test.py --config configs/your_test_config.yaml
```

## 📊 Results

We present 5× super-resolution reconstruction from 8-view low-resolution T2w MRI using our method.

The figure below compares Bicubic, LS-SRR, and our reconstruction on simulated data:

![Comparison](BUMonkey_Results/bumonkey.png)
*Figure: Sagittal reconstructions and corresponding error maps for Bicubic, LS-SRR, and ROVER-MRI on simulated low-resolution data.


## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{rover2025,
  title={Rapid Whole Brain 180µm Mesoscale In-vivo T2w Imaging},
  author={Lyu, J. and Ning, L. and others},
  booktitle={ISMRM},
  year={2025}
}
@inproceedings{lyurotating,
  title={Rotating-view super-resolution (ROVER)-MRI reconstruction using tailored Implicit Neural Network},
  author={Lyu, J. and Ning, L. and others},
  booktitle={ISMRM},
  year={2024}
}
```

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.

---
## 🤝 Contributors

- Jun Lyu
- Lipeng Ning, William Consagra, Qiang Liu, Richard J. Rushmore, Yogesh Rathi

Contact: jlyu1@bwh.harvard.edu
