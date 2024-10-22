# -*- coding:utf-8 _*-

import re
import os
import json
import argparse
import torch
from vllm import LLM, SamplingParams
import util

LANGUAGE_DICT = {
    'sw': 'Swahili',
    'bn': 'Bengali',
    'te': 'Telugu',
    'th': 'Thai',
    'ja': 'Japanese',
    'zh': 'Chinese',
    'ru': 'Russian',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'en': 'English'
}

MCOT_PROMPTS = {
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

class InstructionTemplates:
    @staticmethod
    def get_template(template_name):
        if 'orca' in template_name:
            return "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        elif 'metamath' in template_name:
            return (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{user_message}\n\n### Response: Let's think step by step."
            )
        elif 'bactrian' in template_name:
            return (
                "### Input:\n{user_message}\n\n### Output:\n"
            )
        elif 'mathoctopus' in template_name:
            return (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request in {input_lang}. Please answer in {output_lang}.\n\n"
                "### Instruction:\n{user_message}\n\n### Response:"
            )
        elif 'wizardmath' in template_name:
            return (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{user_message}\n\n### Response: Let's think step by step."
            )
        elif 'mammoth' in template_name:
            return (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{user_message}\n\n### Response:"
            )
        elif 'mcot' in template_name:
            return "Question: \n{user_message} \nAnswer: \n{language}\n"
        else:
            raise NotImplementedError

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return s

def extract_answer(sentence):
    sentence = str(sentence).replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*|\d+(?:\s+\d+)?', sentence)]
    if not pred:
        return float('inf')
    pred_answer = float(pred[-1])

    return pred_answer

def _extract_choice(completion, direct_answer_trigger: tuple):
    # model may generate "The answer is choice (a)"
    completion = completion.strip('\n')
    completion = re.split('|'.join(direct_answer_trigger), completion)[-1]
    completion = completion.strip('\n').rstrip('.').rstrip('/').strip(' ')
    pred = re.findall(r'\b(A|B|C|D|E|F|G|H|I|J)\b', completion.upper())
    if pred is None:
        pred = ""
    if len(pred) > 0:
        pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip('.').rstrip('/')
    return pred

def _is_correct(completion, answer, answer_choice=None):
    direct_answer_trigger = ('####', 'The answer is')
    answer_number_eval = test_answers(completion, answer)
    is_correct, pred, gold = answer_number_eval
    if is_correct:
        return is_correct, pred, gold
    elif answer_choice:
        answer_choice_eval = _extract_choice(completion, direct_answer_trigger)
        pred = answer_choice_eval
        gold = answer_choice
        if answer_choice_eval == answer_choice:
            is_correct = True
    return is_correct, pred, gold

def test_answers(pred_str, answer):
    pred = remove_boxed(util.last_boxed_only_string(pred_str))
    if util.is_equiv(pred, answer):
        return True, pred, answer
    else:
        try:
            if isinstance(pred, str):
                pred = extract_answer(pred)
            answer = float(answer.replace(',',''))
            if abs(pred - answer)<0.001:
                return True, pred, answer
            else:
                return False, pred, answer
        except:
            return False, pred, answer


def read_mmlu_inputs(lang, file_path, template_name):
    answer_choices = []
    answer_numbers = []
    questions = []
    
    prompt = InstructionTemplates.get_template(template_name)
    
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            instruction = doc['instruction'] + f'\nAnswer Choices: (A) {doc["option_a"]} (B) {doc["option_b"]} (C) {doc["option_c"]} (D) {doc["option_d"]}'
            if 'mathoctopus' in template_name:
                question = prompt.format(
                    input_lang=LANGUAGE_DICT[lang],
                    output_lang=LANGUAGE_DICT[lang],
                    user_message=instruction.strip(),
                )
            elif 'mcot' in template_name:
                question = prompt.format(
                    language=MCOT_PROMPTS[lang].strip(),
                    user_message=instruction.strip(),
                )
            else:
                question = prompt.format(
                    user_message=instruction.strip(),
                )
            questions.append(question)
            answer_choices.append(doc['answer'])
            answer_numbers.append(doc['answer_number'])

    return questions, answer_numbers, answer_choices

def read_inputs(lang, file_path, template_name):
    answers = []
    questions = []
    
    prompt = InstructionTemplates.get_template(template_name)
    
    with open(file_path, 'r', encoding="utf-8") as f:
        f=json.load(f)
        for line in f:
            if 'mathoctopus' in template_name:
                question = prompt.format(
                    input_lang=LANGUAGE_DICT[lang],
                    output_lang=LANGUAGE_DICT[lang],
                    user_message=line['question'].strip(),
                )
            elif 'mcot' in template_name:
                question = prompt.format(
                    language=MCOT_PROMPTS[lang].strip(),
                    user_message=line['question'].strip(),
                )
            else:
                question = prompt.format(
                    user_message=line['question'].strip(),
                )
            questions.append(question)
            answers.append(line['answer'])

    return questions, answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--template_name', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inp_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--stop', type=str, nargs='+', default=['</s>', '[END]', '<|im_end|>'])
    parser.add_argument('--max_num_batched_tokens', type=int, default=4096)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)
    args = parser.parse_args()

    # num_gpus = torch.cuda.device_count()
    print(f'args.num_gpus: {args.num_gpus}')
    another_args = {
        'max_num_batched_tokens': args.max_num_batched_tokens,
        'max_model_len': args.max_tokens,
    }

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=torch.bfloat16,
    )
    print('[info] Model loaded')

    # Sampling params
    sampling_params = SamplingParams(
        top_p=args.top_p,
        stop=args.stop,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty
    )

    if args.task == 'mgsm':
        langs = "bn	de	en	es	fr	ja	ru	sw	te	th	zh".split()
    elif args.task == 'msvamp':
        langs = "bn	de	en	es	fr	ja	ru	sw	th	zh".split()
    elif args.task == 'm_mmlu_math':
        langs = """
        ar  ca  de  es  fr  hi  hu  id  it  kn  nb  nl  ro  sk  sw  te  uk  zh
        bn  da  en  eu  gu  hr  hy  is  ja  ml  mr  ne  pt  ru  sr  sv  ta  th  vi
        """.split()
    else:
        raise NotImplementedError

    accuracies = {}

    for lang in langs:
        test_data_path = os.path.join(args.inp_path, f'{args.task}', f'test_{lang}.json')
        # Reading data with prompt template
        answer_choices = None
        if args.task == "m_mmlu_math":
            questions, answers, answer_choices = read_mmlu_inputs(lang, test_data_path, args.template_name)
        else:
            questions, answers = read_inputs(lang, test_data_path, args.template_name)
        print('[info] Data loaded')
        print(f'[info] Data Sample: {questions[0]}')

        # Generate outputs
        outputs = llm.generate(questions, sampling_params)
        print(f'[info] Data output: {outputs[0].outputs[0].text}')
        
        sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
        print('[info] Generation done')

        acc = 0
        # Write outputs
        with open(os.path.join(args.out_path, f'{args.model.split("/")[-1]}_{args.task}_{lang}.json'), "w", encoding="utf-8") as f:
            for i, output in enumerate(sorted_outputs):
                result, pred, gold = _is_correct(output.outputs[0].text, answers[i], answer_choice=answer_choices[i] if answer_choices else None)
                acc += int(result)
                f.write(json.dumps({'question': output.prompt,
                                    'response': output.outputs[0].text,
                                    'prediction': pred,
                                    'answer': gold,
                                    'result': result}, ensure_ascii=False) + '\n')

        # print('[info] %s | Accuracy: %.4f | Correct %d | Total %d' % (
        #     lang, float(acc / len(answers)), acc, len(answers)))
        accuracies[lang] = acc / len(answers)
        print(f'[info] {lang} | Accuracy: {accuracies[lang]:.4f} | Correct {acc} | Total {len(answers)}')

# 打印所有语言的准确率
print("\n[info] All Language Accuracies:")
print("Languages: ", "\t".join(langs))
print("Accuracies: ", "\t".join(f'{accuracies[lang]:.4f}' for lang in langs))