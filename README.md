<h2 align="center"> <a href="">Template Matters: Understanding the Role of Instruction Templates in Multimodal Language Model Evaluation and Training
</a></h2>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub.  </h2>

<p align="center">
    <img src="assets/teaser.png" width="1000" style="margin-bottom: 0.2;"/>
<p>


**The left (a) illustrates the high sensitivity of Multimodal Language Models (MLMs) to variations in instruction templates.** 
We compare the best and worst accuracy of eight prominent MLMs across 100 different instruction templates on the MMBench dataset. The accuracy gaps are marked in red bold; **The right (b) shows that visual instruction tuning with diverse instruction templates significantly improves MLM's performance and reduces the performance variance.** LLaVA-1.5-7B trained with diverse instruction templates achieves the highest average performance and the lowest performance variance among similar-scale MLMs on the SeedBench dataset, evaluated across 25 instruction templates that are not included in the training. 

## 🔔News
  
 **🔥[2024-12-04]: Code released!**

## What's TemplateMatters ?
We propose a programmatic instruction template generator, aimed at enhancing the understanding of the critical role instruction templates play in large Multimodal Language Model (MLM) evaluation and training.

## Abstract

Current MLM evaluation and training approaches overlook the influence of instruction format, presenting an elephant-in-the-room problem. Previous research deals with this problem by manually crafting instructions, failing to yield significant insights due to limitations in diversity and scalability. In this work, we propose a programmatic instruction template generator capable of producing over 39B unique template combinations by filling randomly sampled positional synonyms into weighted sampled meta templates, enabling us to comprehensively examine the MLM's performance across diverse instruction templates. Our experiments across eight common MLMs on five benchmark datasets reveal that MLMs have high template sensitivities with at most 29% performance gaps between different templates. We further augment the instruction tuning dataset of LLaVA-1.5 with our template generator and perform instruction tuning on LLaVA-1.5-7B and LLaVA-1.5-13B. Models tuned on our augmented dataset achieve the best overall performance when compared with the same scale MLMs tuned on at most 75 times the scale of our augmented dataset, highlighting the importance of instruction templates in the instruction tuning process.



## Install
You can easily download the repo and set up the environments via:
```
git clone https://github.com/shijian2001/TemplateMatters
cd ./TemplateMatters

conda create -n template python==3.10
conda activate template
pip install -r requirements.txt
```

## Instruction Template Generator
We provide three easy-to-use interfaces: `QuestionTemplateGenerator`, `ChoiceTemplateGenerator`, `VQATemplateGenerator`. You can easily use them to generate diverse instruction templates as follows：
```python
from tm.template_generator import VQATemplateGenerator, generate_templates_set

print(VQATemplateGenerator().num_all_potential_templates)
# 3939857075

## Randomly generate template
template = VQATemplateGenerator().generate()
print(template)
# The question about the provided picture asks for an response: {question}
# Available options are listed below and you should pick the best answer:
# {choices}
prompt = template.format(
    question="How many cants are there in the image?"
    choices="(A) 1 (B) 2 (C) 3 (D) 4"
)

## Generate a specified number of non-repeating templates
vqa_templates_set = generate_templates_set(VQATemplateGenerator, num_templates=1000)
print(len(vqa_templates_set))
# 1000
```

## Evaluation

### Dataset
We support the following five datasets: 
- `SingleImageQADataset`: BLINK, MMBench, SeedBench, Task-Me-Anything, MMMU

We offer a unified interface to load and process VQA datasets in a standard format. You can load a VQA dataset easily as follows:
```python
from tm.qa_datasets import SingleImageQADataset

tma = SingleImageQADataset("tma-subset").get_dataset()
tma
# Dataset({
#     features: ['id', 'image', 'question', 'choices', 'answer'],
#     num_rows: 100
# })
```

**The subsets used in our paper are available 🤗[here](https://huggingface.co/collections/shijianS01/templatematters-dataset-674f22fed0110c9d450624d2).**

### Model
We support the following eight models: 
- `ImageQAModel`: llavav1.5-7b, llavav1.5-13b, llavav1.6-7b, llavav1.6-13b, qwenvl-chat, qwenvl, idefics2-8b, internvl-chat-v1.5-24b

You can use our unified VQA interface for inference:

```python
from tm.qa_models import ImageQAModel, build_prompt_func
from tm.qa_datasets import SingleImageQADataset
import torch

vqa_model = ImageQAModel("llavav1.5-7b", enable_choice_search=True, torch_device=0, precision=torch.bfloat16)
tma = SingleImageQADataset("tma-subset").get_dataset()
test = tma[0]

result = vqa_model.multiple_choice_qa(
    image=test["image"],
    question=test["question"],
    choices=test["choices"],
    answer=test["answer"],
    prompt_func=build_prompt_func("Question: {question}\nSelect from the following choices: {choices}")
)
result

## Example Result
# {'prompt': 'Question: How many textile mat are there in the image?\nSelect from the following choices: (A) 8 (B) 5 (C) 4 (D) 1',
#  'free_form_answer': 'D',
#  'multiple_choice_answer': '1',
#  'answer': '4',
#  'accuracy': 0}
```

### Instruction Templates for Evaluation
The two instruction template sets used in our paper are available below:

[Simple](https://github.com/shijian2001/TemplateMatters/blob/main/templates/simple_templates.json): three commonly used simple templates

[Complex](https://github.com/shijian2001/TemplateMatters/blob/main/templates/complex_templates.json): 100 instruction templates randomly generated by our template generator

## Training

### Traing Resources
We trained five 7B and five 13B models based on LLaVA-1.5 resources. Follow [here](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning) to prepare your data and training scripts.

### Training Templates
You can prepare your training instruction templates like follows:

```python
from tm.template_generator import QuestionTemplateGenerator, generate_templates_set, assign_templates

# Generate 15000 templates and assign to all data
training_templates = assign_templates(
    num_data=665000, 
    templates_set=generate_templates_set(
        QuestionTemplateGenerator, 
        num_templates=15000
    )
)
print(len(training_templates))
# 665000
```
Then you should add the templates to the instruction part of your insturction-tuning dataset.

### Checkpoints
**The 10 model checkpoints involved in our paper can be found 🤗[here](https://huggingface.co/collections/shijianS01/templatematters-model-674dd4469a2382bb17bb3460).**

We also support these models, you can simply load the model as follows:
```python
from tm.qa_models import ImageQAModel
import torch

## 7b models
# llavav1.5-7b-100-templated, llavav1.5-7b-1k-templated, llavav1.5-7b-5k-templated, llavav1.5-7b-10k-templated, llavav1.5-7b-15k-templated

## 13b models
# llavav1.5-13b-100-templated, llavav1.5-13b-1k-templated, llavav1.5-13b-5k-templated, llavav1.5-13b-10k-templated, llavav1.5-13b-15k-templated

template_tuned_model = ImageQAModel("llavav1.5-7b-100-templated", enable_choice_search=True, torch_device=0, precision=torch.bfloat16)
```

### Evaluating the template-tuned models
We tested our tuned models on 100 generator-created templates ([Complex](https://github.com/shijian2001/TemplateMatters/blob/main/templates/complex_templates.json)), 3 common used templates ([Simple](https://github.com/shijian2001/TemplateMatters/blob/main/templates/simple_templates.json)), and 25 handwritten held-out templates available [here](https://github.com/shijian2001/TemplateMatters/blob/main/templates/heldout_templates.json).

## Contact
- Shijian Wang: shijian@seu.edu.cn