# torch-ngp-hotspot

This is a PyTorch-based implementation of NeRF (Neural Radiance Fields) and SDF (Signed Distance Functions) project, supporting various 3D reconstruction and rendering functionalities.

## Project Structure

```
.
├── config/                 # Configuration directory
│   ├── ngp/               # NGP related configurations
│   ├── deepsdf/           # DeepSDF related configurations
│   └── generate_configs.py # Configuration generation script
├── data/                  # Dataset directory
├── gridencoder/          # Grid encoder implementation
├── sdf/                  # SDF related implementation
├── hotspot/              # Hotspot related implementation
├── deepsdf/              # DeepSDF related implementation
├── main.py               # NGP main program
└── main_deepsdf.py       # DeepSDF main program
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Build extensions:
```bash
# Build grid encoder
cd gridencoder
python setup.py build_ext --inplace
pip install .

# Build SDF extension
cd ../sdf
pip install .
```

## Dataset

ModelNet10

## Usage

### NGP Training
```bash
python main.py --config config/hotspot.yaml
```

### DeepSDF Training
```bash
python main_deepsdf.py --config config/deepsdf.yaml
```

## Notes

1. First run will require CUDA extension compilation, which may take some time
2. Please ensure datasets are placed in the `./data` directory
3. Supported data formats are the same as instant-ngp, such as [armadillo](https://github.com/NVlabs/instant-ngp/blob/master/data/sdf/armadillo.obj) and [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox)

## Experiment Logs

The project includes training logs:
- `train_ngp.log`: NGP training log
- `train_deepsdf.log`: DeepSDF training log
