import json

class Task:
    def __init__(self, id, difficulty, domain, task, answer):
        self.id = id
        self.difficulty = difficulty
        self.domain = domain
        self.task = task
        self.answer = answer


ALLOWED_DIFFICULTY = {"simple", "medium", "hard"}
ALLOWED_DOMAIN = {"math", "algorithms", "logic", "text_reasoning"}


def parse_task(json_string):
    data = json.loads(json_string)

    required_fields = ["id", "difficulty", "domain", "task", "answer"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing field: {field}")

    if not isinstance(data["id"], str):
        raise TypeError("id must be string")

    if data["difficulty"] not in ALLOWED_DIFFICULTY:
        raise ValueError("Invalid difficulty")

    if data["domain"] not in ALLOWED_DOMAIN:
        raise ValueError("Invalid domain")

    if not isinstance(data["task"], str):
        raise TypeError("task must be string")

    if not isinstance(data["answer"], str):
        raise TypeError("answer must be string")

    return Task(
        id=data["id"],
        difficulty=data["difficulty"],
        domain=data["domain"],
        task=data["task"],
        answer=data["answer"]
    )