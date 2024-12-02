<h2 align="center"> <a href="">🤗Template Matters🤗</a></h2>


## 🌟 What's TemplateMatters ?
We propose a programmatic instruction template generator, aimed at enhancing the understanding of the critical role instruction templates play in large Multimodal Language Model (MLM) evaluation and training.

## 🤗 Example
```python
from tm.qa_models import ImageQAModel, build_prompt_func
from tm.qa_datasets import SingleImageQADataset
import torch

vqa_model = ImageQAModel("llavav1.5-7b", enable_choice_search=True, torch_device=0, precision=torch.bfloat16)
tma = SingleImageQADataset("tma-subset").get_dataset()

result = vqa_model.multiple_choice_qa(
    image=tma[0]["image"],
    question=tma[0]["question"],
    choices=tma[0]["choices"],
    answer=tma[0]["answer"],
    prompt_func=build_prompt_func("Question: {question}\nSelect from the following choices: {choices}")
)

result

## Example Result
# {'prompt': 'Question: How many textile mat are there in the image?\nSelect from the following choices: (A) 8 (B) 5 (C) 4 (D) 1',
#  'choices': ['8', '5', '4', '1'],
#  'free_form_answer': 'D',
#  'multiple_choice_answer': '1',
#  'answer': '4',
#  'accuracy': 0}
```
