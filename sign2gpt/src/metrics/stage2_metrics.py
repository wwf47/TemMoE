"""Stage 2 metrics: BLEU + ROUGE for translation evaluation."""

from typing import List, Dict

from src.metrics.utils.sacrebleu import (
    compute_bleu,
    SMOOTH_VALUE_DEFAULT,
    corpus_bleu_attributes,
)


def rouge_l_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L (longest common subsequence) score.
    """
    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    count = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            continue
        
        lcs = lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        count += 1
    
    if count == 0:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l": 0.0}
    
    return {
        "rouge_l_precision": (total_precision / count) * 100,
        "rouge_l_recall": (total_recall / count) * 100,
        "rouge_l": (total_f1 / count) * 100,
    }


class Stage2Metrics:
    """Accumulates predictions and references for batch evaluation."""
    
    def __init__(self, tokenize: str = "none"):
        """Args:"""
        self.tokenize = tokenize
        self.reset()
    
    def reset(self):
        self.correct = [0, 0, 0, 0]
        self.total = [0, 0, 0, 0]
        self.sys_len = 0
        self.ref_len = 0
        self._predictions: List[str] = []
        self._references: List[str] = []
    
    def update(self, predictions: List[str], references: List[str]):
        attrs = corpus_bleu_attributes(
            sys_stream=predictions,
            ref_streams=[references],
            smooth_method="floor",
            smooth_value=SMOOTH_VALUE_DEFAULT,
            force=True,
            tokenize=self.tokenize,
            use_effective_order=True,
        )
        for i in range(4):
            self.correct[i] += attrs["correct"][i]
            self.total[i] += attrs["total"][i]
        self.sys_len += attrs["sys_len"]
        self.ref_len += attrs["ref_len"]
        self._predictions.extend(predictions)
        self._references.extend(references)
    
    def compute(self) -> Dict[str, float]:
        if self.sys_len == 0:
            return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "rouge_l": 0.0}
        
        bleu_scores = compute_bleu(
            self.correct,
            self.total,
            self.sys_len,
            self.ref_len,
            smooth_method="floor",
            smooth_value=SMOOTH_VALUE_DEFAULT,
            use_effective_order=True,
        ).scores

        bleu = {f"bleu{i+1}": bleu_scores[i] for i in range(len(bleu_scores))}
        rouge = rouge_l_score(self._predictions, self._references)
        
        return {**bleu, "rouge_l": rouge["rouge_l"]}
