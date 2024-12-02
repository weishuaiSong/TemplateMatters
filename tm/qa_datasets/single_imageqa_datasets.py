from datasets import load_dataset, concatenate_datasets
import ast

from .base_vqa_datasets import SingleVQADatasetInstance, BaseSingleVQADataset

single_image_qa_datasets = {

    "blink-subset": ("BLINK", "shijianS01/blink-subset"),
    "mmbench-subset": ("MMBench", "shijianS01/mmbench-subset"),
    "seedbench1-subset": ("SeedBench1", "shijianS01/seedbench-subset"),
    "tma-subset": ("TaskMeAnything", "shijianS01/tma-subset"),
    "mmmu-subset": ("MMMU", "shijianS01/mmmu-subset"),

    "blink-dev-all-single-images": ("BLINK", "BLINK-Benchmark/BLINK"),
    "seedbench1-all-single-images": ("SeedBench1", "lmms-lab/SEED-Bench"),
    "mmbench-en-dev-all-single-images": ("MMBench", "lmms-lab/MMBench"),
    "tma-all": ("TaskMeAnything", "weikaih/TaskMeAnything-v1-imageqa-random"),
    "mmmu-dev-val-all-single-images": ("MMMU", "lmms-lab/MMMU")
}


def set_imageqa_dataset_key(model_name, key):
    single_image_qa_datasets[model_name] = (
        single_image_qa_datasets[model_name][0], key)


def list_imageqa_datasets():
    return list(single_image_qa_datasets.keys())


class SingleImageQADataset(BaseSingleVQADataset):
    def __init__(
        self,
        dataset_name: str,
        dataset: SingleVQADatasetInstance = None
    ):
        super().__init__(dataset_name)

        if dataset is None:
            print(f"Loading {dataset_name}...")
            class_name, dataset_path = single_image_qa_datasets[dataset_name]
            self.dataset = eval(class_name)(dataset_path)
            print(f"Finish loading {dataset_name}")


class BLINK(SingleVQADatasetInstance):

    def __init__(self, dataset_path):
        if dataset_path == "shijianS01/blink-subset":
            self.dataset = load_dataset(dataset_path)["eval"]
        else:
            datasets = []
            for single_image_task in ['Counting', 'IQ_Test', 'Object_Localization', 'Relative_Depth', 'Relative_Reflectance', 'Spatial_Relation']:
                subset = load_dataset(
                    "BLINK-Benchmark/BLINK", single_image_task, split="val")
                datasets.append(subset)
            combined_dataset = concatenate_datasets(datasets)
            self.dataset = combined_dataset

    def get_standard_dataset(self):

        standard_dataset = self.dataset.select_columns(
            ["image_1", "question", "choices", "answer"]).rename_column("image_1", "image")

        def _process_data(sample):

            sample["context"] = ""

            # change answer from A/B/C to the concrete value
            answer_to_index = {"(A)": 0, "(B)": 1, "(C)": 2, "(D)": 3}
            index = answer_to_index[sample["answer"]]
            sample["answer"] = sample["choices"][index]

            return sample

        standard_dataset = standard_dataset.map(_process_data)

        return standard_dataset


class MMBench(SingleVQADatasetInstance):

    def __init__(self, dataset_path):
        if dataset_path == "shijianS01/mmbench-subset":
            self.dataset = load_dataset(dataset_path)["eval"]
        else:
            self.dataset = load_dataset(dataset_path, "en")["dev"]

    def get_standard_dataset(self):

        standard_dataset = self.dataset.select_columns(
            ["index", "image", "question", "hint", "answer", "A", "B", "C", "D"]).rename_column("hint", "context")

        def _process_data(sample):
            sample["choices"] = [sample[option]
                                 for option in ["A", "B", "C", "D"] if sample[option] != "nan"]

            answer_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
            index = answer_to_index[sample["answer"]]

            sample["answer"] = sample["choices"][index]

            return sample

        standard_dataset = standard_dataset.map(_process_data)
        standard_dataset = standard_dataset.remove_columns(
            ["A", "B", "C", "D"])
        return standard_dataset


class SeedBench1(SingleVQADatasetInstance):

    def __init__(self, dataset_path):
        if dataset_path == "shijianS01/seedbench-subset":
            self.dataset = load_dataset(dataset_path)["eval"]
        else:
            self.dataset = load_dataset(dataset_path)[
                "test"].select(range(14233))

    def get_standard_dataset(self):

        standard_dataset = self.dataset.select_columns(
            ["image", "question", "answer", "choice_a", "choice_b", "choice_c", "choice_d"]).rename_column("image", "image_list")

        def _process_data(sample):

            sample["context"] = ""

            assert len(sample["image_list"]) == 1
            sample["image"] = sample["image_list"][0]
            sample["choices"] = [sample[option]
                                 for option in ["choice_a", "choice_b", "choice_c", "choice_d"]]

            answer_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
            index = answer_to_index[sample["answer"]]

            sample["answer"] = sample["choices"][index]

            return sample

        standard_dataset = standard_dataset.map(_process_data)
        standard_dataset = standard_dataset.remove_columns(
            ["choice_a", "choice_b", "choice_c", "choice_d", "image_list"])
        return standard_dataset


class TaskMeAnything(SingleVQADatasetInstance):

    def __init__(self, dataset_path):
        if dataset_path == "shijianS01/tma-subset":
            self.dataset = load_dataset(dataset_path)["eval"]
        else:
            tma = load_dataset(dataset_path)
            combined_dataset = concatenate_datasets(
                [tma[split] for split in tma.keys()])
            self.dataset = combined_dataset

    def get_standard_dataset(self):

        standard_dataset = self.dataset.select_columns(
            ["id", "image", "question", "options", "answer"]).rename_column("options", "choices")
        return standard_dataset


class MMMU(SingleVQADatasetInstance):

    def __init__(self, dataset_path):
        if dataset_path == "shijianS01/mmmu-subset":
            self.dataset = load_dataset(dataset_path)["eval"]
        else:
            mmmu = load_dataset(dataset_path)
            dev = mmmu["dev"].filter(
                lambda x: x["image_2"] is None and x["question_type"] == "multiple-choice")
            val = mmmu["validation"].filter(
                lambda x: x["image_2"] is None and x["question_type"] == "multiple-choice")
            combined_dataset = concatenate_datasets([dev, val])
            self.dataset = combined_dataset

    def get_standard_dataset(self):

        standard_dataset = self.dataset.select_columns(["id", "image_1", "question", "options", "answer"]).rename_columns({
            "image_1": "image",
            "options": "choices"
        })

        def _process_data(sample):
            sample["choices"] = ast.literal_eval(sample["choices"])
            answer_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            index = answer_to_index[sample["answer"]]
            sample["answer"] = sample["choices"][index]

            return sample

        standard_dataset = standard_dataset.map(_process_data)
        return standard_dataset
