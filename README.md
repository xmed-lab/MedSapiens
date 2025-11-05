# MedSapiens: Taking a Pose to Rethink Medical Imaging Landmark Detection
Landmark detection in medical imaging plays a pivotal role in diagnostics and surgical planning. Traditional methods often rely on domain-specific models, but the advent of large-scale pretrained models has reshaped the field. MedSapien introduces a groundbreaking framework that leverages LoRA fine-tuning of the SAPIENS model—originally designed for general pose estimation—to achieve state-of-the-art performance in anatomical landmark detection. By bridging the gap between general vision models and medical imaging applications, MedSapien sets a new benchmark for universal landmark detection frameworks.

## Features
- **Universal Framework**: Adaptable for multiple medical imaging datasets.
- **LoRA Fine-Tuning**: Enhances SAPIENS for medical-specific datasets.

| **Method**                                           | **Head Dataset**  | **Hand Dataset**  | **Chest Dataset** | **Leg Dataset**   |
|------------------------------------------------------|-------------------|-------------------|-------------------|-------------------|
| [NFDP](https://doi.org/10.xxxx/NFDP)                 | 1.245 ± 0.276     | 0.673 ± 0.152     | 5.13 ± 1.44       | 2.685 ± 0.617     |
| [UniverDetect](https://doi.org/10.xxxx/UniverDetect) | 1.55 ± 1.74       | 0.71 ± 1.78       | 4.06 ± 3.73       | N/A               |
| [Sapiens]() + LoRA      <br/>                        | 1.246 ± 0.270     | 0.705 ± 0.116     | 3.846 ± 1.27      | 2.647 ± 0.572     |
| **MedSapiens**                                       | 1.275 ± 0.285     | 0.664 ± 0.110     | **3.715 ± 1.31**  | 2.691 ± 0.555     |
| **+ LoRA** 🚀                                        | **1.244 ± 0.276** | **0.638 ± 0.106** | 3.734 ± 1.24      | **2.509 ± 0.556** |



## Getting Started

### Clone the Repository
```bash
git clone https://github.com/xmed-lab/MedSapiens
```

### Installation 
MedSapien follow strictly [**SAPIENS installation pipeline**]().

#### 1. Set up the Environment
Use the provided installation script to create and configure the `sapiens` environment:
```bash
conda create -n sapiens python=3.10 -y
conda activate sapiens
```

#### 2. Install Dependencies
Install PyTorch and CUDA (12.1 or 11.8):
```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
Install additional Python libraries:
```bash
pip install chumpy scipy munkres tqdm cython numpy==1.26.4 pandas fsspec yapf==0.40.1 matplotlib packaging omegaconf ipdb ftfy regex
```
Install MMCV (CUDA: 12.1 or 11.8):
```bash
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

#### 3. Install Custom Modules
Install the required modules in editable mode:
```bash
bash pip_install_editable.sh
```

### Data and Model Weights
#### Download Data
Download the datasets from [here](https://drive.google.com/file/d/1G_3Gir_MJ2Hbm4A2Oqwcy579Mgo2hpYQ/view?usp=sharing
) and extract them to the `data/` directory:

```
gdown --id 1G_3Gir_MJ2Hbm4A2Oqwcy579Mgo2hpYQ -O med_sapien.zip
unzip med_sapien.zip -d data/
```

```
data/
└─ med_sapien/
    ├─ Images/
    └─ [dataset-specific JSON annotation files]
```
#### Download Model Weights
- Download [Med-Sapien Weights here](https://drive.google.com/file/d/1Nxes7MczB3dNvA2JMtGXcSEUEk8gQg4F/view?usp=sharing).
```bash
gdown --id 1Nxes7MczB3dNvA2JMtGXcSEUEk8gQg4F -O checkpoints.zip

unzip checkpoints.zip
```
`
The downloaded weights will have the following directory structure:
```
checkpoints/
└── med_sapien/
    ├──best_EPE_epoch_199.pth
    ├── head/
    │   └── best_EPE_epoch_190.pth
    ├── hand/
    │   └── best_EPE_epoch_207.pth
    ├── chest/
    │   └── best_EPE_epoch_10.pth
    └── legs/
        └── best_EPE_epoch_208.pth
```


## 🎯 Customized MedSapien

### LoRA Fine-Tuning 
Use the `lora_med_sapiens.sh` script to fine-tune the model. Specify the dataset (`chest`, `hand`, `head`, or `legs`).

```bash
bash scripts/train/lora_med_sapiens.sh <DATASET>
```
- Example:
```bash
bash scripts/train/lora/lora_med_sapiens.sh chest
```

### Testing
Use the second `lora_med_sapiens.sh` script to evaluate the model:
```bash
bash scripts/test/lora_med_sapiens.sh <DATASET>
```
- Example:
```bash
bash scripts/test/lora_med_sapiens.sh chest 
```

### Evaluation
To evaluate the model predictions, use the `evaluate.py` script. Specify the required arguments:

```bash
python evaluate.py \
    --annotations path/to/annotations.json \
    --predictions path/to/predictions.pkl \
    --output-dir path/to/output/dir \
    [--keypoint-order path/to/keypoint_order.json] \
    [--save-vis]
```

- Example:
```bash
python evaluation/evaluate.py \
    --annotations data/med_sapien/chest_coco_val_annotations.json \
    --predictions src/pose/Outputs/test_lora/chest/predictions.pkl \
    --output-dir src/pose/Outputs/evaluation/chest \
    --save-vis
```
### Configurations
- Adjust batch sizes, devices, and other parameters directly in the `.sh` scripts as needed.
- Update paths for dataset annotations and checkpoints in the `configs/` directory.

## 🤝 Acknowledgements & Contributions
This project builds on the exceptional work by [SAPIENS](https://github.com/open-mmlab). Contributions and collaborations are welcome! For questions or issues, please open a GitHub issue.

## 📚 Citation
If you find this work useful, please cite the forthcoming paper:
```
@article{medsapien2024,
  title={MedSapien: Taking a Pose to Rethink Medical Imaging Landmark Detection},
  author={},
  arxiv={soon},
  year={2024}
}
```

## 📜 License
This project is licensed under the [SAPIENS License](LICENSE). Portions derived from open-source projects adhere to [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
