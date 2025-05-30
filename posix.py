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

# Step 3: Iterate over the dataset
for item in tqdm(dataset, desc="Processing dataset"):  # <-- tqdm added here
    question = item["question"]
    choices = item["choices"]
    answer = item["answer"]
    image = item["image"]

    responses = []
    logprobs = []

    # Step 4: Iterate over templates, build prompt_func, call multiple_choice_qa
    for template in tqdm(templates, desc="Generating responses", leave=False):  # <-- tqdm added here
        prompt_func = build_prompt_func(template)
        result = vqa_model.multiple_choice_qa(
            image=image,
            question=question,
            choices=choices,
            answer=answer,
            prompt_func=prompt_func,
        )
        responses.append(result.get("response", ""))
        logprobs.append(result.get("log_probability", 0.0))

    # Step 5: Construct logprob matrix (cross-scoring)
    N = len(templates)
    logprob_matrix = []

    for i in tqdm(range(N), desc="Scoring templates", leave=False):  # <-- tqdm added here
        row = []
        for j in range(N):
            current_prompt = build_prompt_func(templates[i])(question, choices)
            response = responses[j]
            full_prompt = "USER: <image>\n" + current_prompt + "\nASSISTANT:"
            full_text = full_prompt + response

            inputs = vqa_model.model.processor(
                text=full_text,
                images=image,
                return_tensors='pt',
                truncation=True,
                max_length=2048
            ).to(vqa_model.model.model.device)

            prompt_inputs = vqa_model.model.processor(
                text=full_prompt,
                images=image,
                return_tensors='pt'
            )
            prompt_length = prompt_inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = vqa_model.model.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.pixel_values
                )

            log_probs = torch.log_softmax(outputs.logits, dim=-1)[0]

            total_log_prob = 0.0
            for k in range(inputs.input_ids.shape[1] - prompt_length):
                pred_position = prompt_length + k - 1
                token_position = prompt_length + k
                token_id = inputs.input_ids[0, token_position].item()
                token_log_prob = log_probs[pred_position, token_id].item()
                total_log_prob += token_log_prob

            row.append(total_log_prob)
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

# Step 8: Save average POSIX
all_posix_values = [entry["posix"] for entry in results]
average_posix = sum(all_posix_values) / len(all_posix_values)

with open(f"{args.model_name}_{args.dataset_name}_average_posix.json", "w") as f:  # <-- f-string fixed
    json.dump({"average_posix": average_posix}, f, indent=2)
