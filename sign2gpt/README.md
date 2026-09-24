# Sign2GPT temporal MoE

Stage-2 sign language translation with MoE FFNs in selected MetaFormer blocks.

Edit the paths in `configs/phoenix2014t.yaml` or `configs/csldaily.yaml` before training. A Stage-1 DINOv2 checkpoint, XGLM-1.7B, and LMDB videos are required.

```bash
pip install torch accelerate transformers peft pyyaml sacrebleu pillow pandas lmdb albumentations
```

Train Phoenix-2014T (e12 k4):

```bash
python scripts/train_stage2_adaptor.py --config configs/phoenix2014t.yaml
```

Train CSL-Daily (e16 k4):

```bash
python scripts/train_stage2_adaptor.py --config configs/csldaily.yaml
```

Resume:

```bash
python scripts/train_stage2_adaptor.py --config configs/phoenix2014t.yaml --resume checkpoints_sign2gpt_phoenix2014t/stage2_latest.pt
```
