from functools import partial

def doc_to_text(doc, connector):
    return (
        f"{connector['Background']}: "
        + doc["context"]
        + "\n\n"
        + f"{connector['Question']}: "
        + doc["question"]
        + "\n\n"
        + f"{connector['Answer']}:"
    )

doc_to_text_en = partial(
    doc_to_text,
    connector={
        "Background": "Background",
        "Question": "Question",
        "Answer": "Answer",
    },
)

doc_to_text_zh = partial(
    doc_to_text,
    connector={
        "Background": "背景",
        "Question": "问题",
        "Answer": "答案",
    },
)

# Arabic
doc_to_text_ar = partial(
    doc_to_text,
    connector={
        "Background": "خلفية",
        "Question": "سؤال",
        "Answer": "إجابة",
    },
)

# German
doc_to_text_de = partial(
    doc_to_text,
    connector={
        "Background": "Hintergrund",
        "Question": "Frage",
        "Answer": "Antwort",
    },
)

# Greek
doc_to_text_el = partial(
    doc_to_text,
    connector={
        "Background": "Υπόβαθρο",
        "Question": "Ερώτηση",
        "Answer": "Απάντηση",
    },
)

# Spanish
doc_to_text_es = partial(
    doc_to_text,
    connector={
        "Background": "Antecedentes",
        "Question": "Pregunta",
        "Answer": "Respuesta",
    },
)

# Hindi
doc_to_text_hi = partial(
    doc_to_text,
    connector={
        "Background": "पृष्ठभूमि",
        "Question": "प्रश्न",
        "Answer": "उत्तर",
    },
)

# Romanian
doc_to_text_ro = partial(
    doc_to_text,
    connector={
        "Background": "Fundal",
        "Question": "Întrebare",
        "Answer": "Răspuns",
    },
)

# Russian
doc_to_text_ru = partial(
    doc_to_text,
    connector={
        "Background": "Фон",
        "Question": "Вопрос",
        "Answer": "Ответ",
    },
)

# Thai
doc_to_text_th = partial(
    doc_to_text,
    connector={
        "Background": "พื้นหลัง",
        "Question": "คำถาม",
        "Answer": "คำตอบ",
    },
)

# Turkish
doc_to_text_tr = partial(
    doc_to_text,
    connector={
        "Background": "Arka plan",
        "Question": "Soru",
        "Answer": "Cevap",
    },
)

# Vietnamese
doc_to_text_vi = partial(
    doc_to_text,
    connector={
        "Background": "Bối cảnh",
        "Question": "Câu hỏi",
        "Answer": "Câu trả lời",
    },
)