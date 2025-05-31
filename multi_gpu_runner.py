# multi_gpu_runner.py

import torch
import argparse
import os
import json
from multiprocessing import Process, set_start_method
from pathlib import Path
from tm.qa_models import ImageQAModel, build_prompt_func
from tqdm import tqdm
from pathlib import Path



set_start_method("spawn", force=True)

def run_inference(args, full_dataset, start_idx, end_idx):
    template_path = Path(args.template_path)
    with open(template_path) as f:
        templates = json.load(f)["MultiChoiceImageQa"]

    vqa_model = ImageQAModel(
        args.model_name, enable_choice_search=True,
        torch_device=args.torch_device, precision=torch.bfloat16
    )

    results = []
    # 只处理指定范围的数据
    for i in tqdm(range(start_idx, end_idx), desc=f"GPU {args.torch_device} Processing"):
        item = full_dataset[i]
        print(type(item))
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
            responses.append(result.get("free_form_answer", ""))
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

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

def calculate_split_indices(total_len, num_parts, part_idx):
    """计算每个部分的起始和结束索引"""
    part_len = total_len // num_parts
    start_idx = part_idx * part_len
    end_idx = total_len if part_idx == num_parts - 1 else start_idx + part_len
    return start_idx, end_idx

def run_on_gpu(rank, gpu_id, args, full_dataset, start_idx, end_idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args.torch_device = 0  # 每个进程只有一个可见卡，编号就是 0
    args.output_path = f"{args.model_name}_{args.dataset_name}_results_rank{rank}.json"
    run_inference(args, full_dataset, start_idx, end_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="final_result.json")
    parser.add_argument("--gpu_ids", type=str, required=True, help="Comma separated GPU ids, e.g., 0,2,3")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    num_gpus = len(gpu_ids)

    from tm.qa_datasets import SingleImageQADataset
    full_dataset = SingleImageQADataset(args.dataset_name).get_dataset()
    total_len = len(full_dataset)

    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx, end_idx = calculate_split_indices(total_len, num_gpus, i)
        p = Process(target=run_on_gpu, args=(i, gpu_id, args, full_dataset, start_idx, end_idx))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 合并所有结果
    all_results = []
    for i in range(num_gpus):
        partial_path = f"{args.model_name}_{args.dataset_name}_results_rank{i}.json"
        with open(partial_path) as f:
            all_results.extend(json.load(f))

    # 计算平均 POSIX
    all_posix_values = [entry["posix"] for entry in all_results]
    average_posix = sum(all_posix_values) / len(all_posix_values)

    # 保存最终合并结果
    with open(args.output_path, "w") as f:
        json.dump({
            "average_posix": average_posix,
            "results": all_results
        }, f, indent=2)