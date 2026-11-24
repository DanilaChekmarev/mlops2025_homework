import json
from datetime import datetime, timezone
from pathlib import Path

COLORS = {"normal": "\033[32m", "warning": "\033[33m", "critical": "\033[31m"}


class MonitorLogger:
    def __init__(self, config):
        self.console_colors = config["console_colors"]
        self.log_file = Path(config["log_file"])
        self.metrics_file = Path(config["metrics_file"])

        for path in (self.log_file, self.metrics_file):
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event, level="normal", **fields):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "level": level,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        if event == "metrics":
            with self.metrics_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        print(f"{COLORS[level]}{line}\033[0m" if self.console_colors else line)
