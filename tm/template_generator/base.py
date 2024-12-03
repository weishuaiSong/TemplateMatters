from functools import reduce
from typing import List, Optional, Union
import random
import re


class PositionalSynonyms:
    def __init__(self, name: str, candidates: List[str], comment: str = "none"):
        self.name = name
        self.candidates = candidates
        self.comment = comment  # comments about the element

    @property
    def random_candidate(self):
        return random.choice(self.candidates)

    @property
    def all_candidates(self):
        return self.candidates

    @property
    def num_candidates(self):
        return len(self.candidates)


class MetaTemplate:
    """
    - Example meta_template: {verb} the{is_following}image, answer the question: {{question}}.
    - Only extract {}, instead of {{}}
    - Designed for template of template
    """

    def __init__(self, meta_template: str, positional_synonyms: Optional[List[PositionalSynonyms]] = None):
        self.meta_template = meta_template
        self.positional_synonyms = positional_synonyms or []
        self._check_meta_template()

    def _is_duplicate(self, placeholders: List[str]) -> bool:
        return len(placeholders) != len(set(placeholders))

    def _check_meta_template(self):
        self.placeholders = re.findall(
            r'(?<!\{)\{([^{}]*)\}(?!\})', self.meta_template)
        if self._is_duplicate(self.placeholders):
            raise ValueError("Duplicate placeholders are not allowed")
        if self.positional_synonyms:
            positional_synonyms_names = [
                element.name for element in self.positional_synonyms]
            if (set(self.placeholders) != set(positional_synonyms_names)) or (len(self.placeholders) != len(positional_synonyms_names)):
                raise ValueError(
                    "MetaTemplate placeholders do not match positional_synonyms names")

    @property
    def num_placeholders(self):
        return len(self.placeholders)

    @property
    def num_potential_templates(self):
        return reduce(lambda x, y: x * y.num_candidates, self.positional_synonyms or [], 1)

    def fit_meta_template(self):
        if not self.positional_synonyms:
            return self.meta_template.format()
        element_dict = {
            element.name: element.random_candidate for element in self.positional_synonyms}
        # Ensured that the first letter of the sentence is capitalized
        # Ensured that the generated senetence is striped
        fited = self.meta_template.format(**element_dict).strip()
        return fited[0].upper() + fited[1:]


class Node:
    def __init__(self, name: str, meta_template: Optional['MetaTemplate'] = None):
        self.name = name
        self.children: List[Node] = []
        self.meta_template = meta_template  # Only leaf nodes have meta_template
        self.weight = 1

    def add_child(self, child: 'Node'):
        self.children.append(child)

    def is_leaf(self) -> bool:
        return self.meta_template is not None

    def balance_weights(self) -> float:
        if self.is_leaf():
            # Weights are based on the number of placeholders of the meta_template
            # Add-one smoothing
            # self.weight = self.meta_template.num_placeholders + 1

            # Weights are based on the number of potentail generated templates of the meta_template
            self.weight = self.meta_template.num_potential_templates
        else:
            self.weight = sum(child.balance_weights()
                              for child in self.children)
        return self.weight

    def traverse(self) -> 'Node':
        if self.is_leaf():
            return self
        next_node = random.choices(self.children, weights=[
                                   child.weight for child in self.children], k=1)[0]
        return next_node.traverse()


class TemplateGenerator:
    def __init__(self, data: Union[dict, list], name: str = '', enable_balanced: bool = True):
        self.root = self._build_taxonomy(data, name)
        if enable_balanced:
            self.root.balance_weights()

    @property
    def num_all_potential_templates(self) -> int:
        return self._get_total_templates(self.root)

    def _get_total_templates(self, node: Node) -> int:
        if node.is_leaf():
            return node.meta_template.num_potential_templates
        else:
            return sum(self._get_total_templates(child) for child in node.children)

    def _build_taxonomy(self, data: Union[dict, list], name: str = '') -> Node:
        if isinstance(data, dict):
            node = Node(name=name)
            for key, value in data.items():
                child_node = self._build_taxonomy(value, name=key)
                node.add_child(child_node)
            return node
        elif isinstance(data, list):
            parent_node = Node(name=name)
            for i, meta_template in enumerate(data):
                meta_template_str = f"meta_template_{i+1}"
                meta_template_node = Node(
                    name=meta_template_str, meta_template=meta_template)
                parent_node.add_child(meta_template_node)
            return parent_node
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _find_node_by_path(self, path: str, node: Optional[Node] = None) -> Optional[Node]:
        """
        - Example Path: Taxonomy/Declarative/Simple
        """
        if node is None:
            node = self.root
        parts = path.split('/')
        if parts[0] != node.name:
            return None
        if len(parts) == 1:
            return node
        for child in node.children:
            result = self._find_node_by_path('/'.join(parts[1:]), child)
            if result:
                return result
        return None

    def generate(self, path: Optional[str] = None):
        start_node = self.root
        if path:
            start_node = self._find_node_by_path(path)
            if not start_node:
                raise ValueError(f"Node with path '{path}' not found.")
        leaf = start_node.traverse()
        if leaf.meta_template:
            return leaf.meta_template.fit_meta_template()
        else:
            raise ValueError(
                "Traversal did not result in a valid meta_template.")

    def visualize_taxonomy(self, node: Optional[Node] = None, level: int = 0):
        if node is None:
            node = self.root
        indent = " " * (level * 4)
        if node.is_leaf():
            print(
                f"{indent}- {node.name} (weight: {node.weight}): {node.meta_template.meta_template}")
        else:
            print(f"{indent}+ {node.name} (weight: {node.weight})")
            for child in node.children:
                self.visualize_taxonomy(child, level + 1)
