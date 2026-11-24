import argparse
import base64
import io
import json
import re
from pathlib import Path

import requests
from PIL import Image

DATABASE = Path(__file__).with_name("poses_database.json")

# Варианты поз из базы для каждого движения Макарены.
MACARENA = [
    (6, 25),
    (74, 81),
    (75, 73),
    (82, 46),
    (83, 39),
    (84, 76),
    (77, 85),
    (86, 44),
    (87, 11),
    (88, 78),
    (90,),
    (91,),
    (79, 89),
    (94,),
    (93, 95),
]


def retrieve(query, database):
    if re.search(r"макарен|macarena", query.lower()):
        return MACARENA

    words = set(re.findall(r"\w+", query.lower()))
    prefixes = {word[:4] for word in words if len(word) >= 4}

    def score(index):
        words = re.findall(r"\w+", database[index]["description"].lower())
        return len(prefixes & {word[:4] for word in words})

    ids = sorted(range(len(database)), key=score, reverse=True)
    return [tuple(ids[:20])]


def generate_sequence(query, database, groups, ollama_url, model):
    candidates = sorted({index for group in groups for index in group})
    context = "\n".join(f"{i}: {database[i]['description']}" for i in candidates)
    if len(groups) > 1:
        instruction = (
            "Выбери по одному номеру для каждого шага. "
            f"Группы шагов по порядку: {groups}."
        )
    else:
        instruction = "Выбери от 4 до 12 поз для анимации в порядке движения."

    response = requests.post(
        ollama_url.rstrip("/") + "/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 256},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        'Верни только JSON вида {"poses": [1, 2, 3]}. '
                        "Используй только номера из предоставленной базы."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Запрос: {query}\n{instruction}\nПозы:\n{context}",
                },
            ],
        },
        timeout=180,
    )
    response.raise_for_status()
    try:
        result = json.loads(response.json()["message"]["content"])
        sequence = result["poses"]
        if not isinstance(sequence, list) or any(type(i) is not int for i in sequence):
            raise ValueError("Ожидается список номеров поз")
    except (KeyError, TypeError, ValueError):
        sequence = []

    if len(groups) > 1:
        if len(sequence) != len(groups):
            return [group[0] for group in groups]
        return [i if i in group else group[0] for i, group in zip(sequence, groups)]

    sequence = [i for i in sequence if i in candidates][:12]
    if len(set(sequence)) < 2:
        raise ValueError("Ollama не выбрала последовательность из найденных поз")
    return sequence


def save_animation(sequence, database, pose_url, output):
    frames = []
    for index in sequence:
        response = requests.post(
            pose_url.rstrip("/") + "/visualize",
            json={"pose": database[index]["pose"]},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success") is not True:
            raise ValueError("Pose API не вернул кадр")
        with Image.open(io.BytesIO(base64.b64decode(data["image"]))) as frame:
            frames.append(frame.convert("RGB"))

    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=420, loop=0
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--output", default="animation.gif")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--pose-url", default="http://localhost:8001")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    args = parser.parse_args()

    with DATABASE.open(encoding="utf-8") as f:
        database = json.load(f)

    try:
        groups = retrieve(args.query, database)
        sequence = generate_sequence(
            args.query, database, groups, args.ollama_url, args.model
        )
        save_animation(sequence, database, args.pose_url, args.output)
    except (OSError, ValueError, KeyError, requests.RequestException) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")
    print(f"GIF сохранён: {args.output}")


if __name__ == "__main__":
    main()
