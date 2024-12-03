#### Util Functions for Template Generator ####

from typing import *
import random
from tqdm import tqdm
from .base import TemplateGenerator
from .vqa_meta_data import QuestionMetaTemplates, ChoiceMetaTemplates


class QuestionTemplateGenerator:
    def __init__(self, enable_balanced: bool = True):
        self.generator = TemplateGenerator(
            QuestionMetaTemplates, enable_balanced=enable_balanced)

    def generate(self):
        return self.generator.generate()

    @property
    def num_all_potential_templates(self):
        return self.generator.num_all_potential_templates


class ChoiceTemplateGenerator:
    def __init__(self, enable_balanced: bool = True):
        self.generator = TemplateGenerator(
            ChoiceMetaTemplates, enable_balanced=enable_balanced)

    def generate(self):
        return self.generator.generate()

    @property
    def num_all_potential_templates(self):
        return self.generator.num_all_potential_templates


class VQATemplateGenerator:
    def __init__(self, enable_shuffle: bool = False, enable_balanced: bool = True):
        self.enable_shuffle = enable_shuffle
        self.question_template_generator = QuestionTemplateGenerator(
            enable_balanced)
        self.choices_template_generator = ChoiceTemplateGenerator(
            enable_balanced)

    def generate(self):
        question_template = self.question_template_generator.generate()
        choices_template = self.choices_template_generator.generate()

        templates = [question_template, choices_template]

        if self.enable_shuffle:
            random.shuffle(templates)

        return '\n'.join(templates)

    @property
    def num_all_potential_templates(self):
        total = self.question_template_generator.num_all_potential_templates * \
            self.choices_template_generator.num_all_potential_templates
        return total


def generate_templates_set(template_generator, num_templates: int):
    """Generate a specified number of templates with no duplicate elements"""
    assert num_templates <= template_generator().num_all_potential_templates, (
        "The number of generated templates should be less than or equal to the capacity of the template generator."
    )
    templates_set = set()
    with tqdm(total=num_templates, desc="Generating templates") as pbar:
        while len(templates_set) < num_templates:
            template = template_generator().generate()
            if template not in templates_set:
                templates_set.add(template)
                pbar.update(1)
    return list(templates_set)


def assign_templates(num_data: int, templates_set: List[str]) -> List[str]:
    """Assign a fixed set of prompt templates to the data and ensure that each template is sampled."""
    assert num_data >= len(templates_set), (
        "The number of data should be greater than or equal to the number of templates."
    )
    randomized_templates = random.sample(templates_set, len(templates_set))
    all_templates = randomized_templates + \
        random.choices(templates_set, k=num_data - len(templates_set))
    random.shuffle(all_templates)
    assert set(all_templates) == set(
        templates_set), "Not all templates have been used."
    return all_templates
