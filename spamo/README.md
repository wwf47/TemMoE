# SpaMo temporal MoE

SpaMo sign language translation with a Mixture-of-Experts TemporalConv.

Edit the feature and annotation paths in `configs/phoenix2014t.yaml` or `configs/csldaily.yaml`. Spatial (CLIP) and motion (VideoMAE) features must already be extracted.

```bash
pip install -r requirements.txt
```

Train Phoenix-2014T (e4 k1, Flan-T5-XL):

```bash
python main.py -c configs/phoenix2014t.yaml -n phoenix2014t -e bleu
```

Train CSL-Daily (residual e16 k4, mT0-XL):

```bash
python main.py -c configs/csldaily.yaml -n csldaily -e bleu
```
