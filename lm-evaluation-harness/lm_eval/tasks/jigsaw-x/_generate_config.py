import argparse

import yaml


# Different languages that are part of xnli.
# These correspond to dataset names (Subsets) on HuggingFace.
# A yaml file is generated by this script for each language.

LANGUAGES = {
    "it": {  # German
        "QUESTION_WORD": "È un commento tossico?",
        "YES": "SÌ",
        "NO": "NO",
    },
    "pt": {  # English
        "QUESTION_WORD": "Este é um comentário tóxico?",
        "YES": "Sim",
        "NO": "Não",
    },
    "es": {  # Spanish
        "QUESTION_WORD": "¿Es este un comentario tóxico?",
        "YES": "Sí",
        "NO": "No",
    },
    "fr": {  # French
        "QUESTION_WORD": "Est-ce un commentaire toxique ?",
        "YES": "Oui",
        "NO": "No",
    },
    "ru": {  # Japanese
        "QUESTION_WORD": "Это токсичный комментарий?",
        "YES": "Да",
        "NO": "Нет",
    },
    "tr": {  # Korean
        "QUESTION_WORD": "zehirli bir yorum mu?",
        "YES": "Evet",
        "NO": "HAYIR",
    }
}


def gen_lang_yamls(output_dir: str, overwrite: bool) -> None:
    """
    Generate a yaml file for each language.

    :param output_dir: The directory to output the files to.
    :param overwrite: Whether to overwrite files if they already exist.
    """
    err = []
    for lang in LANGUAGES.keys():
        file_name = f"jigsaw_{lang}.yaml"
        try:
            QUESTION_WORD = LANGUAGES[lang]["QUESTION_WORD"]
            YES = LANGUAGES[lang]["YES"]
            NO = LANGUAGES[lang]["NO"]
            with open(
                f"{output_dir}/{file_name}", "w" if overwrite else "x", encoding="utf8"
            ) as f:
                f.write("# Generated by utils.py\n")
                yaml.dump(
                    {
                        "include": "jigsawx_template_yaml",
                        "dataset_name": lang,
                        "task": f"jigsaw_{lang}",
                        "doc_to_text": "",
                        "doc_to_choice": f"{{{{["
                        f"""comment_text+\", {QUESTION_WORD}? {NO}\","""
                        f""" comment_text+\", {QUESTION_WORD}? {YES}\""""
                        f"]}}}}",
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