# Online Hadwriting Recognition
This study presents a lightweight neural architecture for online handwriting recognition designed for deployment on resource-constrained devices. The approach builds upon multimodal models by incorporating both trajectory-based and image-based representations during training, while retaining only the trajectory encoder during inference. This design significantly reduces model size and memory requirements while maintaining competitive recognition performance. Experiments on the IAM-OnDB dataset demonstrate that the proposed model achieves a reason- able balance between recognition accuracy and efficiency, confirming the feasibility of lightweight online handwriting recognition for embedded systems.

# PTOHWR (Python Toolkit for Online Handwriting Recognition)

This repo contains a lightweight Col-OLHTR-based model for online handwriting recognition.  
Below you’ll find instructions on how to install dependencies, run training, inference and single-sample examples, and a description of all available command-line parameters.

## Script Overview

The main script (e.g. `app.py`) supports three modes:

- **train**   — train the model  
- **infer**   — run inference on a split file  
- **example** — run inference on a single sample

## Command-Line Usage
```
python app.py \
  --config <PATH_TO_CONFIG> \
  [--mode {train,infer,example}] \
  [--device {mps,cuda,cpu}] \
  [--checkpoint <CKPT_PATH>] \
  [--split <SPLIT_FILE>]
```
Arguments:

• --config       (string, required)  
  Path to the YAML config file (model hyperparameters, data paths, etc.)

• --mode         (string, default: train)  
  Which mode to run:  
  - train  
  - infer  
  - example  

• --device       (string, default: mps)  
  Device for model execution: mps, cuda or cpu

• --checkpoint   (string)  
  Path to a `.pth`/`.pt` checkpoint (required for infer and example)

• --split        (string, default: testset_f.txt)  
  Split file for inference (IAM-OnDB format)

## Examples

1. Train on GPU (or MPS):
```
   python app.py \
     --config configs/train.yaml \
     --mode train \
     --device cuda
```
2. Run inference on the test split:
```
   python app.py \
     --config configs/infer.yaml \
     --mode infer \
     --device cuda \
     --checkpoint outputs/checkpoint_latest.pth \
     --split splits/testset_f.txt
```
   
3. Quick single-sample example:
   
```
   python app.py \
     --config configs/train.yaml \
     --mode example \
     --device cpu \
     --checkpoint outputs/checkpoint_epoch10.pth
```  
