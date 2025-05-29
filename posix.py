import json
from tqdm import tqdm
from tm.qa_models import ImageQAModel, build_prompt_func
from tm.qa_datasets import SingleImageQADataset
from pathlib import Path
import argparse
import torch

# Argument parser for configurable inputs
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llavav1.5-7b", help="Model name to load")
parser.add_argument("--dataset_name", type=str, default="tma-subset", help="Dataset name")
parser.add_argument("--template_path", type=str, default="TemplateMatters/templates", help="Path to template JSON file")
parser.add_argument("--output_path", type=str, default="tma_posix_results.json", help="Output path for results")
args = parser.parse_args()

# Step 1: Load templates
template_path = Path(args.template_path)

with open(template_path) as f:
    templates = json.load(f)["MultiChoiceImageQa"]  # 25 templates

# Step 2: Load model and dataset
vqa_model = ImageQAModel(
    args.model_name, enable_choice_search=True, torch_device=0, precision=torch.bfloat16
)
dataset = SingleImageQADataset(args.dataset_name).get_dataset()

results = []

vqa_model = ImageQAModel(
    args.model_name, enable_choice_search=True, torch_device=0, precision=torch.bfloat16
)
dataset = SingleImageQADataset(args.dataset_name).get_dataset()

results = []

# 辅助函数：计算给定prompt和response的log概率
def calculate_log_probability(model, processor, image, prompt, response):
    """计算给定prompt和response的log概率"""
    # 构建完整文本
    full_prompt = "USER: <image>\n" + prompt + "\nASSISTANT:"
    full_text = full_prompt + response
    
    # 处理输入
    inputs = processor(
        text=full_text, 
        images=image,
        return_tensors='pt',
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    # 计算prompt长度
    prompt_inputs = processor(
        text=full_prompt,
        images=image,
        return_tensors='pt'
    )
    prompt_length = prompt_inputs.input_ids.shape[1]
    
    # 前向传播获取logits
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values
        )
    
    # 计算log softmax
    log_probs = torch.log_softmax(outputs.logits, dim=-1)[0]
    
    # 计算response部分的log概率
    total_log_prob = 0.0
    for k in range(inputs.input_ids.shape[1] - prompt_length):
        # 预测位置是前一个位置
        pred_position = prompt_length + k - 1
        
        # 目标token位置
        token_position = prompt_length + k
        
        # 获取该token的log概率
        token_id = inputs.input_ids[0, token_position].item()
        token_log_prob = log_probs[pred_position, token_id].item()
        total_log_prob += token_log_prob
    
    return total_log_prob

# Step 3: Iterate over the dataset
for item in tqdm(dataset):
    question = item["question"]
    choices = item["choices"]
    answer = item["answer"]
    image = item["image"]

    responses = []
    logprobs = []

    # Step 4: Iterate over templates, build prompt_func, call multiple_choice_qa
    for template in templates:
        prompt_func = build_prompt_func(template)
        result = vqa_model.multiple_choice_qa(
            image=image,
            question=question,
            choices=choices,
            answer=answer,
            prompt_func=prompt_func,
        )
        responses.append(result.get("response", ""))  # Store response text
        logprobs.append(result.get("log_probability", 0.0))  # Extract log probability

    # Step 5: Construct logprob matrix (cross-scoring)
    N = len(templates)
    logprob_matrix = []

    for i in range(N):
        row = []
        for j in range(N):
            # prompt_i scoring response_j
            current_prompt = build_prompt_func(templates[i])(question, choices)
            log_prob = calculate_log_probability(
                model=vqa_model.model,
                processor=vqa_model.processor,
                prompt_text=current_prompt,
                response_text=responses[j]
            )
            row.append(log_prob)
        logprob_matrix.append(row)

    # Step 6: Compute POSIX score
    psi = 0.0
    for i in range(N):
        for j in range(N):
            psi += abs(logprob_matrix[j][i] - logprob_matrix[j][j]) / 200
    posix = psi / (N * (N - 1))

    # Step 7: Store result
    results.append({
        "question": question,
        "answer": answer,
        "posix": posix,
        "responses": responses,
        "logprob_matrix": logprob_matrix,
    })
all_posix_values = [entry["posix"] for entry in results]
average_posix = sum(all_posix_values) / len(all_posix_values)

# 只保存一个平均值
with open("{args.model_name}_{args.dataset_name}_average_posix.json", "w") as f:
    json.dump({"average_posix": average_posix}, f, indent=2)