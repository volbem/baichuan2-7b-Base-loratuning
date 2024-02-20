import os
import json
import argparse
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import BERTScorer

class TextEvaluator:
    def __init__(self, has_action, model_type="用于评分的模型路径", num_layers=9, device="cuda:0"):
        self.has_action=has_action
        self.rouge_evaluator = Rouge()
        self.bert_scorer = BERTScorer(model_type=model_type, num_layers=num_layers, device=device)

    def calculate_bleu_scores(self, predictions, references):
        # 将文本处理为字符序列
        char_predictions = [list(pred) for pred in predictions]
        char_references = [[list(ref)] for ref in references]
        # 计算BLEU分数
        return sum(sentence_bleu(ref, pred) for pred, ref in zip(char_predictions, char_references)) / len(predictions)

    def calculate_rouge_scores(self, predictions, references):
        # 将文本处理为字符序列，然后用空格连接，因为ROUGE期望的是字符串
        char_predictions = [' '.join(list(pred)) for pred in predictions]
        char_references = [' '.join(list(ref)) for ref in references]
        # 计算ROUGE分数
        return self.rouge_evaluator.get_scores(char_predictions, char_references, avg=True)


    def calculate_bert_scores(self, predictions, references):
        P, R, F1 = self.bert_scorer.score(predictions, references)
        # 将Tensor转换为标准Python数值
        P_mean = P.mean().item()
        R_mean = R.mean().item()
        F1_mean = F1.mean().item()
        
        # 构建可序列化的字典
        scores_dict = {
            "Precision": P_mean,
            "Recall": R_mean,
            "F1": F1_mean
        }
        return scores_dict

    def evaluate(self, predictions, references, reference_actions, response_actions):
        #predictions列表中包含非字符串的None，需要转换
        predictions = ['None' if item is None else item for item in predictions]

        bleu = self.calculate_bleu_scores(predictions, references)
        rouge = self.calculate_rouge_scores(predictions, references)
        bert = self.calculate_bert_scores(predictions, references)
        if self.has_action:
            action_prf=self.calculate_batch_prf(reference_actions, response_actions)
            return {
                "BLEU": bleu,
                "ROUGE": rouge,
                "BERTScore": bert,
                "Action_PRF": action_prf
            }
        else:
            return {
                "BLEU": bleu,
                "ROUGE": rouge,
                "BERTScore": bert,
            }
    
    def calculate_batch_prf(self, reference_actions, response_actions):
        # Initialize sums of precision, recall, and f1 score
        total_precision = 0
        total_recall = 0
        total_f1_score = 0

        # Calculate PRF for each sample and sum them up
        for (reference_action, response_action) in zip(reference_actions, response_actions):
            prf = self.calculate_act_prf(reference_action, response_action)
            total_precision += prf["precision"]
            total_recall += prf["recall"]
            total_f1_score += prf["f1_score"]

        # Calculate averages for the batch
        num_samples = len(data)
        avg_precision = total_precision / num_samples if num_samples else 0
        avg_recall = total_recall / num_samples if num_samples else 0
        avg_f1_score = total_f1_score / num_samples if num_samples else 0

        return {
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1_score": avg_f1_score
        }
    
    def calculate_act_prf(self, reference_action, response_action):
        # Convert lists to sets for easier calculation
        reference_set = set(reference_action)
        response_set = set(response_action)
        # Calculate Precision, Recall, and F1 Score
        true_positives = len(reference_set.intersection(response_set))
        precision = true_positives / len(response_set) if response_set else 0
        recall = true_positives / len(reference_set) if reference_set else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Evaluate text generation performance.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test file containing dialogues and responses.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the performance metrics file will be saved.")
    parser.add_argument("--is_action", type=bool, default=True, help="The data has action.")
    args = parser.parse_args()

    # 读取数据
    with open(args.test_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # 提取responses和generated_responses
    responses = [item["reference"] for item in data]
    generated_responses = [item["response"][0] for item in data]
    # 创建评估器实例并评估性能
    evaluator = TextEvaluator(args.is_action)
    reference_actions = []
    response_actions = []
    if args.is_action:
        reference_actions = [item["reference_action"] for item in data]
        response_actions = [item["response_action"] for item in data]
    scores = evaluator.evaluate(generated_responses, responses, reference_actions, response_actions)
    # 打印评分结果
    for metric, score in scores.items():
        print(f"{metric} Score:", score)

    # 保存性能指标
    performance_file = os.path.join(args.output_dir, "performance_metrics.json")
    with open(performance_file, "w", encoding="utf-8") as json_file:
        json.dump(scores, json_file, ensure_ascii=False, indent=4)
