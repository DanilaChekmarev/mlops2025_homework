import argparse
import time
from pathlib import Path

from src.config import load_config
from src.monitor import ServiceMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/monitoring_config.yaml")
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        default=[Path("step2_fastapi_inference/test_images/img.jpg")],
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    for image in args.images:
        if not image.is_file():
            parser.error(f"Image not found: {image}")

    config = load_config(args.config)
    monitor = ServiceMonitor(config, args.images)
    try:
        while True:
            monitor.check()
            if args.once:
                break
            time.sleep(config["monitoring"]["check_interval_seconds"])
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
