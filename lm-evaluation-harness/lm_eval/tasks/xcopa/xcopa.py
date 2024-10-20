"""
XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning
https://ducdauge.github.io/files/xcopa.pdf

The Cross-lingual Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across languages.
The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around the globe.
The dataset is challenging as it requires both the command of world knowledge and the ability to generalise to new languages.
All the details about the creation of XCOPA and the implementation of the baselines are available in the paper.

Homepage: https://github.com/cambridgeltl/xcopa
"""
import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.metrics import mean

_CITATION = """
@inproceedings{ponti2020xcopa,
  title={{XCOPA: A} Multilingual Dataset for Causal Commonsense Reasoning},
  author={Edoardo M. Ponti, Goran Glava\v{s}, Olga Majewska, Qianchu Liu, Ivan Vuli\'{c} and Anna Korhonen},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020},
  url={https://ducdauge.github.io/files/xcopa.pdf}
}
"""

class Copa(ConfigurableTask):
    VERSION = 0
    DATASET_PATH = "/online1/ycsc_chenkh/hitici_02/data/eval_data/copa"
    # DATASET_NAME = "copa"

    # ORCA_SYSTEM = (
    #     "While answering a multiple choice question, first output the correct answer(s). "
    # )

    ORCA_SYSTEM = (
        "You should describe the task and explain your answer. "
        "While answering a multiple choice question, first output the correct answer(s). "
        "Then explain why other answers are wrong. Think like you are answering to a five year old."
    )
    INSTRUCTION = (
        "Which of the two options is most likely to be the {question} of the premise?\n\n"
        "Premise: {premise}\n"
        "A. {choice1}\n"
        "B. {choice2}\n\n"
        "Answer: "
    )

    CHOICES = ['A.', 'B.']

    def __init__(self, config=None):
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return self.doc_to_text_with_instruction(doc)

    def doc_to_text_with_instruction(self, doc):
        instruction = self.INSTRUCTION.format(
            question=doc["question"],
            premise=doc["premise"],
            choice1=doc["choice1"],
            choice2=doc["choice2"],
        )
        template = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        return template.format(
            system_message=self.ORCA_SYSTEM,
            user_message=instruction,
        )

    def doc_to_target(self, doc):
        return self.doc_to_target_with_instruction(doc)

    def doc_to_target_with_instruction(self, doc):
        correct_choice = 'A.' if doc["label"] == 0 else 'B.'
        return correct_choice

    def aggregation(self):
        return {"em": mean, "acc": mean, "acc_norm": mean}

    def higher_is_better(self):
        return {"em": True, "acc": True, "acc_norm": True}

    def process_results(self, doc, results):
        gold = doc["label"]

        lls, _ = zip(*results)
        acc = 1.0 if np.argmax(lls) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in self.CHOICES])
        acc_norm = 1.0 if np.argmax(lls / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
            "em": acc_norm * 100.0,
        }

    def construct_requests(self, doc, ctx, **kwargs):
        request_list = [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, " {}".format(choice)),
                idx=i,
                **kwargs,
            )
            for i, choice in enumerate(self.CHOICES)
        ]
        return request_list

class XCopa(Copa):
    VERSION = 0
    DATASET_PATH = "/home/export/base/ycsc_chenkh/hitici_02/online1/data/eval_data/xcopa"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]


class XCopaEt(XCopa):
    DATASET_NAME = "et"
    CAUSE = "sest"
    EFFECT = "seetõttu"


class XCopaHt(XCopa):
    DATASET_NAME = "ht"
    CAUSE = "poukisa"
    EFFECT = "donk sa"


class XCopaIt(XCopa):
    DATASET_NAME = "it"
    CAUSE = "perché"
    EFFECT = "quindi"


class XCopaId(XCopa):
    DATASET_NAME = "id"
    CAUSE = "karena"
    EFFECT = "maka"


class XCopaQu(XCopa):
    DATASET_NAME = "qu"
    CAUSE = "imataq"
    EFFECT = "chaymi"


class XCopaSw(XCopa):
    DATASET_NAME = "sw"
    CAUSE = "kwa sababu"
    EFFECT = "kwa hiyo"


class XCopaZh(XCopa):
    DATASET_NAME = "zh"
    CAUSE = "因为"
    EFFECT = "所以"


class XCopaTa(XCopa):
    DATASET_NAME = "ta"
    CAUSE = "காரணமாக"
    EFFECT = "எனவே"


class XCopaTh(XCopa):
    DATASET_NAME = "th"
    CAUSE = "เพราะ"
    EFFECT = "ดังนั้น"


class XCopaTr(XCopa):
    DATASET_NAME = "tr"
    CAUSE = "çünkü"
    EFFECT = "bu yüzden"


class XCopaVi(XCopa):
    DATASET_NAME = "vi"
    CAUSE = "bởi vì"
    EFFECT = "vì vậy"


LANGS = ["et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"]

LANG_CLASSES = [
    XCopaEt,
    XCopaHt,
    XCopaIt,
    XCopaId,
    XCopaQu,
    XCopaSw,
    XCopaZh,
    XCopaTa,
    XCopaTh,
    XCopaTr,
    XCopaVi,
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"orca_xcopa_{lang}"] = lang_class
    return tasks
