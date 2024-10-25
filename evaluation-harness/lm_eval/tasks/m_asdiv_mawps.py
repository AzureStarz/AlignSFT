"""
Language Models are Multilingual Chain-of-Thought Reasoners
https://arxiv.org/abs/2210.03057

Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems, proposed in the paper [Language models are multilingual chain-of-thought reasoners](http://arxiv.org/abs/2210.03057).

The same 250 problems from [GSM8K](https://arxiv.org/abs/2110.14168) are each translated via human annotators in 10 languages. The 10 languages are:
- Spanish
- French
- German
- Russian
- Chinese
- Japanese
- Thai
- Swahili
- Bengali
- Telugu

GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.

You can find the input and targets for each of the ten languages (and English) as `.tsv` files.
We also include few-shot exemplars that are also manually translated from each language in `exemplars.py`.

Homepage: https://github.com/google-research/url-nlp/tree/main/mgsm
"""
import re
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
import datasets
from lm_eval.utils import InstructionTemplates


_CITATION = """
@misc{cobbe2021training,
    title={Training Verifiers to Solve Math Word Problems},
    author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
    year={2021},
    eprint={2110.14168},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
@misc{shi2022language,
    title={Language Models are Multilingual Chain-of-Thought Reasoners},
    author={Freda Shi and Mirac Suzgun and Markus Freitag and Xuezhi Wang and Suraj Srivats and Soroush Vosoughi and Hyung Won Chung and Yi Tay and Sebastian Ruder and Denny Zhou and Dipanjan Das and Jason Wei},
    year={2022},
    eprint={2210.03057},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

ANS_RE = re.compile(r"(\-?\d+)")
INVALID_ANS = "[invalid]"


class M_ASDIV_MAWPS(Task):
    VERSION = 0
    DATASET_PATH = "/home/export/base/ycsc_chenkh/hitici_02/online1/data/eval_data/m_asdiv_mawps"
    DATASET_NAME = None
    QUESTION = "Question:"
    ANSWER = "Answer Choices:"

    ORCA_SYSTEM = (
        "You are an AI assistant. User will you give you a task. "
        "Your goal is to complete the task as faithfully as you can. "
        "While performing the task think step-by-step and justify your steps."
    )

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = datasets.load_dataset(
            self.DATASET_PATH,
            self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )
        # if self.DATASET_NAME == "en":
        #     return
        # self.en_dataset = datasets.load_dataset(
        #     self.DATASET_PATH,
        #     "en",
        #     data_dir=data_dir,
        #     cache_dir=cache_dir,
        #     download_mode=download_mode,
        # )

        # self.dataset['train'] = self.dataset['train'].remove_columns('answer')
        # self.dataset['train'] = self.dataset['train'].add_column(
        #     'answer', self.en_dataset['train']['answer'])

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        raise NotImplementedError

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc, instruction_template=None):
        # if doc["answer"] is not None:
        #     text = doc["question"]
        # else:
        text = self.QUESTION + " " + doc["question"]

        if not instruction_template:
            text = text + "\n" + self.ANSWER

        if instruction_template:
            template = InstructionTemplates.get_template(instruction_template)
            if instruction_template == "orca":
                text = template.format(
                    system_message=self.ORCA_SYSTEM,
                    user_message=text)
            elif instruction_template == 'metamath' or instruction_template == 'wizardmath' or instruction_template == 'mammoth':
                text = template.format(
                    user_message=text)
            elif instruction_template == 'mathoctopus':
                text = template.format(
                    input_lang=self.LANG_NAME,
                    output_lang=self.LANG_NAME,
                    user_message=text)
            elif instruction_template == 'mcot':
                text = doc["question"]
                # Define prompts for different languages
                prompts = {
                    "bn": "আসুন ধাপে ধাপে চিন্তা করি।",
                    "de": "Denken wir Schritt für Schritt.",
                    "en": "Let's think step by step.",
                    "es": "Pensemos paso a paso.",
                    "fr": "Réfléchissons étape par étape.",
                    "ja": "段階的に考えてみましょう。",
                    "ru": "Давайте думать поэтапно.",
                    "sw": "Hebu fikiria hatua kwa hatua.",
                    "te": "అంచెలంచెలుగా ఆలోచిద్దాం.",
                    "th": "ลองคิดทีละขั้นตอน",
                    "zh": "让我们一步步思考。"
                }
                text = template.format(
                    language=prompts[self.LANG_NAME] if self.LANG_NAME in prompts.keys() else prompts['en'],
                    user_message=text)
            else:
                raise NotImplementedError
        return text

    def doc_to_target(self, doc, instruction_template=None):
        if doc["answer"] is not None:
            return " " + doc["answer"][len(self.ANSWER) + 1:] + '[END]'
        else:
            return " " + str(doc["answer_number"]) + '[END]'

    def construct_requests(self, doc, ctx, instruction_template=None):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        if instruction_template:
            completion = rf.greedy_until(
                ctx, {"until": [self.QUESTION, '[END]', '</s>', '<|im_end|>']})
        else:
            completion = rf.greedy_until(
                ctx, {"until": [self.QUESTION, '[END]']})
        return completion

    def _extract_answer(self, completion):
        # code copied from MathOctopus, the original regex in lm_eval is wrong
        completion = re.sub(r"(\d),(\d)", "\g<1>\g<2>",
                            completion)  # 123,456
        res = re.findall(r"(\d+(\.\d+)?)", completion)  # 123456.789
        if len(res) > 0:
            num_str = res[-1][0]
            return float(num_str)
        else:
            return 0.0

    # def _extract_choice(self, completion, direct_answer_trigger: tuple):
    #     # model may generate "The answer is choice (a)"
    #     completion = completion.strip('\n')
    #     completion = re.split('|'.join(direct_answer_trigger), completion)[-1]
    #     completion = completion.strip('\n').rstrip('.').rstrip('/').strip(' ')
    #     pred = re.findall(r'\b(A|B|C|D|E|F|G|H|I|J)\b', completion.upper())
    #     if pred is None:
    #         pred = ""
    #     if len(pred) > 0:
    #         pred = pred[-1]
    #     # Remove the period at the end, again!
    #     pred = pred.rstrip('.').rstrip('/')
    #     return pred

    def _is_correct(self, completion, answer):
        gold = answer
        assert gold != INVALID_ANS, "No ground truth answer found in the document."
        # direct_answer_trigger = ('####', 'The answer is')
        # return self._extract_answer(completion) == float(gold)
        return abs(self._extract_answer(completion) - float(gold)) < 0.001

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        completion = results[0]
        answer = doc["answer_number"]
        return {"acc": self._is_correct(completion, answer)}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"acc": mean}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"acc": True}


class M_ASDIV_MAWPS_Arabic(M_ASDIV_MAWPS):
    DATASET_NAME = "ar"
    LANG_NAME = "Arabic"
    QUESTION = "السؤال:"

class M_ASDIV_MAWPS_Afrikaans(M_ASDIV_MAWPS):
    DATASET_NAME = "af"
    LANG_NAME = "Afrikaans"
    QUESTION = "Vraag:"

class M_ASDIV_MAWPS_Bulgarian(M_ASDIV_MAWPS):
    DATASET_NAME = "bg"
    LANG_NAME = "Bulgarian"
    QUESTION = "Въпрос:"

class M_ASDIV_MAWPS_Catalan(M_ASDIV_MAWPS):
    DATASET_NAME = "ca"
    LANG_NAME = "Catalan"
    QUESTION = "Pregunta:"

class M_ASDIV_MAWPS_Czech(M_ASDIV_MAWPS):
    DATASET_NAME = "cs"
    LANG_NAME = "Czech"
    QUESTION = "Otázka:"

class M_ASDIV_MAWPS_German(M_ASDIV_MAWPS):
    DATASET_NAME = "de"
    LANG_NAME = "German"
    QUESTION = "Frage:"

class M_ASDIV_MAWPS_Spanish(M_ASDIV_MAWPS):
    DATASET_NAME = "es"
    LANG_NAME = "Spanish"
    QUESTION = "Pregunta:"

class M_ASDIV_MAWPS_French(M_ASDIV_MAWPS):
    DATASET_NAME = "fr"
    LANG_NAME = "French"
    QUESTION = "Question:"

class M_ASDIV_MAWPS_Finnish(M_ASDIV_MAWPS):
    DATASET_NAME = "fi"
    LANG_NAME = "Finnish"
    QUESTION = "Kysymys:"

class M_ASDIV_MAWPS_Hindi(M_ASDIV_MAWPS):
    DATASET_NAME = "hi"
    LANG_NAME = "Hindi"
    QUESTION = "प्रश्न:"

class M_ASDIV_MAWPS_Hungarian(M_ASDIV_MAWPS):
    DATASET_NAME = "hu"
    LANG_NAME = "Hungarian"
    QUESTION = "Kérdés:"

class M_ASDIV_MAWPS_Indonesian(M_ASDIV_MAWPS):
    DATASET_NAME = "id"
    LANG_NAME = "Indonesian"
    QUESTION = "Pertanyaan:"

class M_ASDIV_MAWPS_Italian(M_ASDIV_MAWPS):
    DATASET_NAME = "it"
    LANG_NAME = "Italian"
    QUESTION = "Domanda:"

class M_ASDIV_MAWPS_Korean(M_ASDIV_MAWPS):
    DATASET_NAME = "ko"
    LANG_NAME = "Korean"
    QUESTION = "질문:"

class M_ASDIV_MAWPS_Kannada(M_ASDIV_MAWPS):
    DATASET_NAME = "kn"
    LANG_NAME = "Kannada"
    QUESTION = "ಪ್ರಶ್ನೆ:"

class M_ASDIV_MAWPS_Norwegian(M_ASDIV_MAWPS):
    DATASET_NAME = "nb"
    LANG_NAME = "Norwegian"
    QUESTION = "Spørsmål:"

class M_ASDIV_MAWPS_Dutch(M_ASDIV_MAWPS):
    DATASET_NAME = "nl"
    LANG_NAME = "Dutch"
    QUESTION = "Vraag:"

class M_ASDIV_MAWPS_Romanian(M_ASDIV_MAWPS):
    DATASET_NAME = "ro"
    LANG_NAME = "Romanian"
    QUESTION = "Întrebare:"

class M_ASDIV_MAWPS_Slovenian(M_ASDIV_MAWPS):
    DATASET_NAME = "sl"
    LANG_NAME = "Slovenian"
    QUESTION = "Vprašanje:"

class M_ASDIV_MAWPS_Slovak(M_ASDIV_MAWPS):
    DATASET_NAME = "sk"
    LANG_NAME = "Slovak"
    QUESTION = "Otázka:"

class M_ASDIV_MAWPS_Swedish(M_ASDIV_MAWPS):
    DATASET_NAME = "sw"
    LANG_NAME = "Swedish"
    QUESTION = "Fråga:"

class M_ASDIV_MAWPS_Telugu(M_ASDIV_MAWPS):
    DATASET_NAME = "te"
    LANG_NAME = "Telugu"
    QUESTION = "ప్రశ్న:"

class M_ASDIV_MAWPS_Ukrainian(M_ASDIV_MAWPS):
    DATASET_NAME = "uk"
    LANG_NAME = "Ukrainian"
    QUESTION = "Запитання:"

# class M_ASDIV_MAWPS_ChineseTraditional(M_ASDIV_MAWPS):
#     DATASET_NAME = "zh"
#     LANG_NAME = "Chinese Traditional"
#     QUESTION = "問題:"

class M_ASDIV_MAWPS_Bengali(M_ASDIV_MAWPS):
    DATASET_NAME = "bn"
    LANG_NAME = "Bengali"
    QUESTION = "প্রশ্ন:"

class M_ASDIV_MAWPS_Belarusian(M_ASDIV_MAWPS):
    DATASET_NAME = "be"
    LANG_NAME = "Belarusian"
    QUESTION = "Пытанне:"

class M_ASDIV_MAWPS_Danish(M_ASDIV_MAWPS):
    DATASET_NAME = "da"
    LANG_NAME = "Danish"
    QUESTION = "Spørgsmål:"

class M_ASDIV_MAWPS_English(M_ASDIV_MAWPS):
    DATASET_NAME = "en"
    LANG_NAME = "English"
    QUESTION = "Question:"

class M_ASDIV_MAWPS_Basque(M_ASDIV_MAWPS):
    DATASET_NAME = "eu"
    LANG_NAME = "Basque"
    QUESTION = "Galdera:"

class M_ASDIV_MAWPS_Gujarati(M_ASDIV_MAWPS):
    DATASET_NAME = "gu"
    LANG_NAME = "Gujarati"
    QUESTION = "પ્રશ્ન:"

class M_ASDIV_MAWPS_Hausa(M_ASDIV_MAWPS):
    DATASET_NAME = "ha"
    LANG_NAME = "Hausa"
    QUESTION = "Tambaya:"

class M_ASDIV_MAWPS_Croatian(M_ASDIV_MAWPS):
    DATASET_NAME = "hr"
    LANG_NAME = "Croatian"
    QUESTION = "Pitanje:"

class M_ASDIV_MAWPS_Armenian(M_ASDIV_MAWPS):
    DATASET_NAME = "hy"
    LANG_NAME = "Armenian"
    QUESTION = "Հարց:"

class M_ASDIV_MAWPS_Icelandic(M_ASDIV_MAWPS):
    DATASET_NAME = "is"
    LANG_NAME = "Icelandic"
    QUESTION = "Spurning:"

class M_ASDIV_MAWPS_Japanese(M_ASDIV_MAWPS):
    DATASET_NAME = "ja"
    LANG_NAME = "Japanese"
    QUESTION = "質問:"

class M_ASDIV_MAWPS_Malayalam(M_ASDIV_MAWPS):
    DATASET_NAME = "ml"
    LANG_NAME = "Malayalam"
    QUESTION = "പ്രശ്നം:"

class M_ASDIV_MAWPS_Marathi(M_ASDIV_MAWPS):
    DATASET_NAME = "mr"
    LANG_NAME = "Marathi"
    QUESTION = "प्रश्न:"

class M_ASDIV_MAWPS_Nepali(M_ASDIV_MAWPS):
    DATASET_NAME = "ne"
    LANG_NAME = "Nepali"
    QUESTION = "प्रश्न:"

class M_ASDIV_MAWPS_Polish(M_ASDIV_MAWPS):
    DATASET_NAME = "pl"
    LANG_NAME = "Polish"
    QUESTION = "Pytanie:"

class M_ASDIV_MAWPS_Portuguese(M_ASDIV_MAWPS):
    DATASET_NAME = "pt"
    LANG_NAME = "Portuguese"
    QUESTION = "Pergunta:"

class M_ASDIV_MAWPS_Russian(M_ASDIV_MAWPS):
    DATASET_NAME = "ru"
    LANG_NAME = "Russian"
    QUESTION = "Вопрос:"

class M_ASDIV_MAWPS_Serbian(M_ASDIV_MAWPS):
    DATASET_NAME = "sr"
    LANG_NAME = "Serbian"
    QUESTION = "Питање:"

class M_ASDIV_MAWPS_Swahili(M_ASDIV_MAWPS):
    DATASET_NAME = "sv"
    LANG_NAME = "Swahili"
    QUESTION = "Swali:"

class M_ASDIV_MAWPS_Tamil(M_ASDIV_MAWPS):
    DATASET_NAME = "ta"
    LANG_NAME = "Tamil"
    QUESTION = "கேள்வி:"

class M_ASDIV_MAWPS_Thai(M_ASDIV_MAWPS):
    DATASET_NAME = "th"
    LANG_NAME = "Thai"
    QUESTION = "คำถาม:"

class M_ASDIV_MAWPS_Vietnamese(M_ASDIV_MAWPS):
    DATASET_NAME = "vi"
    LANG_NAME = "Vietnamese"
    QUESTION = "Câu hỏi:"

class M_ASDIV_MAWPS_Macedonian(M_ASDIV_MAWPS):
    DATASET_NAME = "mk"
    LANG_NAME = "Macedonian"
    QUESTION = "Прашање:"

class M_ASDIV_MAWPS_Luxembourgish(M_ASDIV_MAWPS):
    DATASET_NAME = "lb"
    LANG_NAME = "Luxembourgish"
    QUESTION = "Fro:"

class M_ASDIV_MAWPS_Chinese(M_ASDIV_MAWPS):
    DATASET_NAME = "zh"
    LANG_NAME = "Chinese"
    QUESTION = "问题:"

LANGS = """
ar  af  bg  ca  cs  de  es  fr  fi  hi  hu  id  it  ko  kn  nb  nl  ro  sl  sk  sw  te  uk  zh  
bn  be  da  en  eu  gu  ha  hr  hy  is  ja  ml  mr  ne  pl  pt  ru  sr  sv  ta  th  vi  mk  lb  
""".split()

LANG_CLASSES = [
    M_ASDIV_MAWPS_Arabic,
    M_ASDIV_MAWPS_Afrikaans,
    M_ASDIV_MAWPS_Bulgarian,
    M_ASDIV_MAWPS_Catalan,
    M_ASDIV_MAWPS_Czech,
    M_ASDIV_MAWPS_German,
    M_ASDIV_MAWPS_Spanish,
    M_ASDIV_MAWPS_French,
    M_ASDIV_MAWPS_Finnish,
    M_ASDIV_MAWPS_Hindi,
    M_ASDIV_MAWPS_Hungarian,
    M_ASDIV_MAWPS_Indonesian,
    M_ASDIV_MAWPS_Italian,
    M_ASDIV_MAWPS_Korean,
    M_ASDIV_MAWPS_Kannada,
    M_ASDIV_MAWPS_Norwegian,
    M_ASDIV_MAWPS_Dutch,
    M_ASDIV_MAWPS_Romanian,
    M_ASDIV_MAWPS_Slovenian,
    M_ASDIV_MAWPS_Slovak,
    M_ASDIV_MAWPS_Swahili,
    M_ASDIV_MAWPS_Telugu,
    M_ASDIV_MAWPS_Ukrainian,
    M_ASDIV_MAWPS_Chinese,
    M_ASDIV_MAWPS_Bengali,
    M_ASDIV_MAWPS_Belarusian,
    M_ASDIV_MAWPS_Danish,
    M_ASDIV_MAWPS_English,
    M_ASDIV_MAWPS_Basque,
    M_ASDIV_MAWPS_Gujarati,
    M_ASDIV_MAWPS_Hausa,
    M_ASDIV_MAWPS_Croatian,
    M_ASDIV_MAWPS_Armenian,
    M_ASDIV_MAWPS_Icelandic,
    M_ASDIV_MAWPS_Japanese,
    M_ASDIV_MAWPS_Malayalam,
    M_ASDIV_MAWPS_Marathi,
    M_ASDIV_MAWPS_Nepali,
    M_ASDIV_MAWPS_Polish,
    M_ASDIV_MAWPS_Portuguese,
    M_ASDIV_MAWPS_Russian,
    M_ASDIV_MAWPS_Serbian,
    M_ASDIV_MAWPS_Swedish,
    M_ASDIV_MAWPS_Tamil,
    M_ASDIV_MAWPS_Thai,
    M_ASDIV_MAWPS_Vietnamese,
    M_ASDIV_MAWPS_Macedonian,
    M_ASDIV_MAWPS_Luxembourgish,
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"m_asdiv_mawps_{lang}"] = lang_class
    return tasks
