import json
import re

class DialogueExtractor:
    def __init__(self):
        self.action_pattern = r"根据对话上文，我应该(.*?)。因此，"
        self.response_pattern = r"因此，我的回复如下：(.*)"

    def extract_info(self, text):
        action_match = re.search(self.action_pattern, text)
        response_match = re.search(self.response_pattern, text)
        action = action_match.group(1) if action_match else None
        response = response_match.group(1) if response_match else None
        return action.split("、"), response

    def process_json_file(self, input_file_path, output_file_path):

        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        new_data = []
        for sample in data:
            # 清理对话中的 "根据对话上文，我应该XXX"  和 "因此，我的回复如下：”

            new_sample = sample.copy()  # Copy the original sample

            # Rename 'reference' and 'response' to 'original_reference' and 'original_response'
            new_sample['original_reference'] = new_sample.pop('reference', None)
            new_sample['original_response'] = new_sample.pop('response', None)
            
            new_sample['reference_action'], new_sample['reference'] = self.extract_info(sample.get('reference', ''))
            new_sample['response_action'], new_sample['response'] = self.extract_info(sample.get('response', '')[0])
            new_sample['response'] = [new_sample['response']]
            new_data.append(new_sample)

        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)


# 使用示例:
extractor = DialogueExtractor()
extractor.process_json_file('../results_cot_split/output.json', '../results_cot_split/output_clear.json')
#extractor.process_json_file(‘预测结果文件路径’, '清理后的文件路径')