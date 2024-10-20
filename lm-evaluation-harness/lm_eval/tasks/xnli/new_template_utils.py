import argparse

import yaml


# Different languages that are part of xnli.
# These correspond to dataset names (Subsets) on HuggingFace.
# A yaml file is generated by this script for each language.

LANGUAGES = {
    "ar": {  # Arabic
        "QUESTION_WORD": "صحيح",
        "ENTAILMENT_LABEL": "نعم",
        "NEUTRAL_LABEL": "لذا",
        "CONTRADICTION_LABEL": "رقم",
    },
    "bg": {  # Bulgarian
        "QUESTION_WORD": "правилно",
        "ENTAILMENT_LABEL": "да",
        "NEUTRAL_LABEL": "така",
        "CONTRADICTION_LABEL": "не",
    },
    "de": {  # German
        "QUESTION_WORD": "richtig",
        "ENTAILMENT_LABEL": "Ja",
        "NEUTRAL_LABEL": "Auch",
        "CONTRADICTION_LABEL": "Nein",
    },
    "el": {  # Greek
        "QUESTION_WORD": "σωστός",
        "ENTAILMENT_LABEL": "Ναί",
        "NEUTRAL_LABEL": "Έτσι",
        "CONTRADICTION_LABEL": "όχι",
    },
    "en": {  # English
        "QUESTION_WORD": "right",
        "ENTAILMENT_LABEL": "Yes",
        "NEUTRAL_LABEL": "Also",
        "CONTRADICTION_LABEL": "No",
    },
    "es": {  # Spanish
        "QUESTION_WORD": "correcto",
        "ENTAILMENT_LABEL": "Sí",
        "NEUTRAL_LABEL": "Asi que",
        "CONTRADICTION_LABEL": "No",
    },
    "fr": {  # French
        "QUESTION_WORD": "correct",
        "ENTAILMENT_LABEL": "Oui",
        "NEUTRAL_LABEL": "Aussi",
        "CONTRADICTION_LABEL": "Non",
    },
    "hi": {  # Hindi
        "QUESTION_WORD": "सही",
        "ENTAILMENT_LABEL": "हाँ",
        "NEUTRAL_LABEL": "इसलिए",
        "CONTRADICTION_LABEL": "नहीं",
    },
    "ru": {  # Russian
        "QUESTION_WORD": "правильно",
        "ENTAILMENT_LABEL": "Да",
        "NEUTRAL_LABEL": "Так",
        "CONTRADICTION_LABEL": "Нет",
    },
    "sw": {  # Swahili
        "QUESTION_WORD": "sahihi",
        "ENTAILMENT_LABEL": "Ndiyo",
        "NEUTRAL_LABEL": "Hivyo",
        "CONTRADICTION_LABEL": "Hapana",
    },
    "th": {  # Thai
        "QUESTION_WORD": "ถูกต้อง",
        "ENTAILMENT_LABEL": "ใช่",
        "NEUTRAL_LABEL": "ดังนั้น",
        "CONTRADICTION_LABEL": "ไม่",
    },
    "tr": {  # Turkish
        "QUESTION_WORD": "doğru",
        "ENTAILMENT_LABEL": "Evet",
        "NEUTRAL_LABEL": "Böylece",
        "CONTRADICTION_LABEL": "Hayır",
    },
    "ur": {  # Urdu
        "QUESTION_WORD": "صحیح",
        "ENTAILMENT_LABEL": "جی ہاں",
        "NEUTRAL_LABEL": "اس لئے",
        "CONTRADICTION_LABEL": "نہیں",
    },
    "vi": {  # Vietnamese
        "QUESTION_WORD": "đúng",
        "ENTAILMENT_LABEL": "Vâng",
        "NEUTRAL_LABEL": "Vì vậy",
        "CONTRADICTION_LABEL": "Không",
    },
    "zh": {  # Chinese
        "QUESTION_WORD": "正确",
        "ENTAILMENT_LABEL": "是的",
        "NEUTRAL_LABEL": "所以",
        "CONTRADICTION_LABEL": "不是的",
    },
}


def gen_lang_yamls(output_dir: str, overwrite: bool) -> None:
    """
    Generate a yaml file for each language.

    :param output_dir: The directory to output the files to.
    :param overwrite: Whether to overwrite files if they already exist.
    """
    err = []
    for lang in LANGUAGES.keys():
        file_name = f"new_xnli_{lang}.yaml"
        try:
            # QUESTION_WORD = LANGUAGES[lang]["QUESTION_WORD"]
            # ENTAILMENT_LABEL = LANGUAGES[lang]["ENTAILMENT_LABEL"]
            # NEUTRAL_LABEL = LANGUAGES[lang]["NEUTRAL_LABEL"]
            # CONTRADICTION_LABEL = LANGUAGES[lang]["CONTRADICTION_LABEL"]
            with open(
                f"{output_dir}/{file_name}", "w" if overwrite else "x", encoding="utf8"
            ) as f:
                f.write("# Generated by utils.py\n")
                yaml.dump(
                    {
                        "include": "xnli_common_yaml",
                        "dataset_name": lang,
                        "task": f"new_xnli_{lang}",
                        "doc_to_text": f"{{{{"
                        f"""\"I will give you a premise and a hypothesis. Choose the most appropriate relationship from the following options:\nA) Entailment\nB) Neutral\nC) Contradiction\n\nPremise: \"+premise+\"\nHypothesis: \"+hypothesis+\"\nAnswer: \""""
                        f"}}}}",
                        "doc_to_choice": f"{{{{"
                        f"""\"Yes\", \"No\", \"Maybe\""""
                        f"}}}}",
                    },
                    f,
                    allow_unicode=True,
                )
        except FileExistsError:
            err.append(file_name)

    if len(err) > 0:
        raise FileExistsError(
            "Files were not created because they already exist (use --overwrite flag):"
            f" {', '.join(err)}"
        )


def main() -> None:
    """Parse CLI args and generate language-specific yaml files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite files if they already exist",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory to write yaml files to"
    )
    args = parser.parse_args()

    gen_lang_yamls(output_dir=args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()