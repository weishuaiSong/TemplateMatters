from .base_vqa_datasets import SingleVQADatsetInstance, BaseSingleVQADataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
import ast

single_image_qa_datasets = {
    "blink-subset": ("BLINK", "./subset/blink", "local"),
    "mmbench-subset": ("MMBench", "./subset/mmbench", "local"),
    "seedbench1-subset": ("SeedBench1", "./subset/seedbench1", "local"),
    "blink-dev-all-single-images": ("BLINK", "BLINK-Benchmark/BLINK", "hf-all"),
    "seedbench1-all-single-images": ("SeedBench1", "lmms-lab/SEED-Bench", "hf-all"),
    "mmbench-en-dev-all": ("MMBench", "lmms-lab/MMBench", "hf-all"),
    "tma-subset": ("TaskMeAnything", "./subset/tma", "local"),
    "tma-all": ("TaskMeAnything", "weikaih/TaskMeAnything-v1-imageqa-random", "hf-all"),
    "mmmu-subset": ("MMMU", "./subset/mmmu_dev", "local"),
    "mmmu-dev-val-all": ("MMMU", "lmms-lab/MMMU", "hf-all")
}


class SingleImageQADataset(BaseSingleVQADataset):
    def __init__(
        self,
        dataset_name: str,
        dataset: SingleVQADatsetInstance = None
    ):
        super().__init__(dataset_name)

        if dataset is None:
            print(f"Loading {dataset_name}...")
            class_name, dataset_path, dataset_type = single_image_qa_datasets[dataset_name]
            self.dataset = eval(class_name)(dataset_path, dataset_type)
            print(f"Finish loading {dataset_name}")


class BLINK(SingleVQADatsetInstance):

    def __init__(self, dataset_path, dataset_type):
        if dataset_type == "local":
            self.dataset = load_from_disk(dataset_path)
        elif dataset_type == "hf-all":
            datasets = []
            for single_image_task in ['Counting', 'IQ_Test', 'Object_Localization', 'Relative_Depth', 'Relative_Reflectance', 'Spatial_Relation']:
                subset = load_dataset(
                    "BLINK-Benchmark/BLINK", single_image_task, split="val")
                datasets.append(subset)
            combined_dataset = concatenate_datasets(datasets)
            self.dataset = combined_dataset
        else:
            return ValueError("Havn't support the dataset type")

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


class MMBench(SingleVQADatsetInstance):

    def __init__(self, dataset_path, dataset_type):
        if dataset_type == "local":
            self.dataset = load_from_disk(dataset_path)
        elif dataset_type == "hf-all":
            self.dataset = load_dataset(dataset_path, "en")["dev"]
        else:
            return ValueError("Havn't support the dataset type")

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


class SeedBench1(SingleVQADatsetInstance):

    def __init__(self, dataset_path, dataset_type):
        if dataset_type == "local":
            self.dataset = load_from_disk(dataset_path)
        elif dataset_type == "hf-all":
            self.dataset = load_dataset(dataset_path)[
                "test"].select(range(14233))
        else:
            return ValueError("Havn't support the dataset type")

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


class TaskMeAnything(SingleVQADatsetInstance):

    def __init__(self, dataset_path, dataset_type):
        if dataset_type == "local":
            self.dataset = load_from_disk(dataset_path)
        elif dataset_type == "hf-all":
            tma = load_dataset(dataset_path)
            combined_dataset = concatenate_datasets(
                [tma[split] for split in tma.keys()])
            self.dataset = combined_dataset
        else:
            return ValueError("Havn't support the dataset type")

    def get_standard_dataset(self):

        standard_dataset = self.dataset.select_columns(
            ["id", "image", "question", "options", "answer"]).rename_column("options", "choices")
        return standard_dataset


class MMMU(SingleVQADatsetInstance):

    def __init__(self, dataset_path, dataset_type):
        if dataset_type == "local":
            self.dataset = load_from_disk(dataset_path)
        elif dataset_type == "hf-all":
            mmmu = load_dataset(dataset_path)
            dev = mmmu["dev"].filter(
                lambda x: x["image_2"] is None and x["question_type"] == "multiple-choice")
            val = mmmu["validation"].filter(
                lambda x: x["image_2"] is None and x["question_type"] == "multiple-choice")
            combined_dataset = concatenate_datasets([dev, val])
            self.dataset = combined_dataset
        else:
            return ValueError("Havn't support the dataset type")

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
