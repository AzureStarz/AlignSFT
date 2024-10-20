"""
Know What You Don’t Know: Unanswerable Questions for SQuAD
https://arxiv.org/pdf/1806.03822.pdf

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset,
consisting of questions posed by crowdworkers on a set of Wikipedia articles,
where the answer to every question is a segment of text, or span, from the
corresponding reading passage, or the question might be unanswerable.
SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable
questions written adversarially by crowdworkers to look similar to answerable ones.
To do well on SQuAD2.0, systems must not only answer questions when possible, but
also determine when no answer is supported by the paragraph and abstain from answering.

Homepage: https://rajpurkar.github.io/SQuAD-explorer/
"""

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.metrics import mean


_CITATION = """
@inproceedings{NEURIPS2019_4496bf24,
    author = {Wang, Alex and Pruksachatkun, Yada and Nangia, Nikita and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
    pages = {},
    publisher = {Curran Associates, Inc.},
    title = {SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems},
    url = {https://proceedings.neurips.cc/paper/2019/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf},
    volume = {32},
    year = {2019}
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