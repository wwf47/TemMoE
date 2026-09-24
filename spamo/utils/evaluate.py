from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF, TER


def _lcs_length(left, right) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for left_token in left:
        cur = [0]
        for j, right_token in enumerate(right, start=1):
            if left_token == right_token:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rouge_l_from_tokens(reference_tokens, prediction_tokens):
    if not reference_tokens or not prediction_tokens:
        return 0.0, 0.0, 0.0
    lcs = _lcs_length(reference_tokens, prediction_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _chinese_chars(text: str):
    return [ch for ch in str(text) if not ch.isspace()]


def evaluate_results(predictions, references, split="train", device='cpu', tokenizer='13a'):
    """Evaluate prediction results using BLEU and ROUGE metrics."""
    log_dicts = {}

    bleu4 = BLEU(max_ngram_order=4, tokenize=tokenizer).corpus_score(predictions, [references]).score
    log_dicts[f"{split}/bleu4"] = bleu4

    if split == 'test':
        for i in range(1, 4):
            score = BLEU(max_ngram_order=i, tokenize=tokenizer).corpus_score(predictions, [references]).score
            log_dicts[f"{split}/bleu" + str(i)] = score

        if tokenizer == "zh":
            rouge_scores = [
                _rouge_l_from_tokens(_chinese_chars(ref), _chinese_chars(pred))
                for ref, pred in zip(references, predictions)
            ]
            denom = max(len(rouge_scores), 1)
            avg_precision = sum(score[0] for score in rouge_scores) / denom
            avg_recall = sum(score[1] for score in rouge_scores) / denom
            avg_f1 = sum(score[2] for score in rouge_scores) / denom
        else:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scored = [
                scorer.score(ref, pred)["rougeL"]
                for ref, pred in zip(references, predictions)
            ]
            denom = max(len(scored), 1)
            avg_precision = sum(score.precision for score in scored) / denom
            avg_recall = sum(score.recall for score in scored) / denom
            avg_f1 = sum(score.fmeasure for score in scored) / denom

        log_dicts[f"{split}/rougeL_precision"] = avg_precision
        log_dicts[f"{split}/rougeL_recall"] = avg_recall
        log_dicts[f"{split}/rougeL_f1"] = avg_f1

    return log_dicts
