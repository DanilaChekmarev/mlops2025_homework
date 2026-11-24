import math
import mimetypes
import time
from itertools import cycle
from statistics import mean

import requests

from .logger import MonitorLogger


class ServiceMonitor:
    def __init__(self, config, images):
        self.config = config
        service = config["service"]
        self.base_url = (
            service.get("base_url") or f"http://{service['host']}:{service['port']}"
        ).rstrip("/")
        self.timeout = config["monitoring"]["request_timeout_seconds"]
        self.images = cycle(images)
        self.logger = MonitorLogger(config["logging"])
        self.failures = {"/health": 0, "/predict": 0}
        self.last_alert = {}

    def request(self, endpoint, image=None):
        start = time.perf_counter()
        fields = {"endpoint": endpoint}
        if image is not None:
            fields["image"] = image.name
        try:
            if image is None:
                response = requests.get(self.base_url + endpoint, timeout=self.timeout)
            else:
                mime = mimetypes.guess_type(image.name)[0]
                if not mime or not mime.startswith("image/"):
                    mime = "image/" + image.suffix.lstrip(".").lower()
                with image.open("rb") as f:
                    response = requests.post(
                        self.base_url + endpoint,
                        files={"file": (image.name, f, mime)},
                        timeout=self.timeout,
                    )

            fields["http_status"] = response.status_code
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")

            if endpoint == "/health":
                if (
                    data.get("status") != "healthy"
                    or data.get("model_loaded") is not True
                ):
                    raise ValueError("Model is unhealthy")
            else:
                result = data.get("result")
                if data.get("success") is not True or not isinstance(result, dict):
                    raise ValueError("Invalid prediction response")
                if result.get("success") is not True:
                    raise ValueError(result.get("prediction") or "Inference failed")
                prediction = result.get("prediction")
                timing = result.get("timing")
                if not isinstance(prediction, str) or not prediction.strip():
                    raise ValueError("Empty or invalid prediction")
                if not isinstance(timing, dict):
                    raise ValueError("Missing processing time")
                processing_ms = timing.get("total_ms")
                if type(processing_ms) not in (float, int) or not (
                    math.isfinite(processing_ms) and processing_ms >= 0
                ):
                    raise ValueError("Invalid processing time")
                fields.update(prediction=prediction, processing_time_ms=processing_ms)

            ok = True
        except (requests.RequestException, ValueError, OSError) as error:
            fields["error"] = str(error)
            ok = False

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.failures[endpoint] = 0 if ok else self.failures[endpoint] + 1
        self.logger.log(
            "request",
            "normal" if ok else "warning",
            ok=ok,
            response_time_ms=round(elapsed_ms, 2),
            **fields,
        )
        return ok, elapsed_ms

    def check(self):
        healthy, health_time = self.request("/health")
        results = [(healthy, health_time)]
        for _ in range(self.config["monitoring"]["samples_per_check"]):
            results.append(self.request("/predict", next(self.images)))

        times = sorted(elapsed for _, elapsed in results)
        # P95 по методу ближайшего ранга.
        metrics = {
            "response_time_ms": mean(times),
            "p95_latency_ms": times[math.ceil(0.95 * len(times)) - 1],
            "error_rate_percent": 100 * sum(not ok for ok, _ in results) / len(results),
            "consecutive_failures": max(self.failures.values()),
            "health_status": "healthy" if healthy else "unhealthy",
        }
        reasons = {"warning": [], "critical": []}
        if not healthy:
            reasons["critical"].append("health_status")
        for metric, limits in self.config["thresholds"].items():
            if metrics[metric] >= limits["critical"]:
                reasons["critical"].append(metric)
            elif metrics[metric] >= limits["warning"]:
                reasons["warning"].append(metric)

        level = "normal"
        if reasons["critical"]:
            level = "critical"
        elif reasons["warning"]:
            level = "warning"

        self.logger.log("metrics", level, **metrics)
        alerts = self.config["alerts"]
        if alerts["enabled"] and level != "normal":
            now = time.monotonic()
            cooldown = alerts["cooldown_minutes"] * 60
            previous = self.last_alert.get(level, float("-inf"))
            if now - previous >= cooldown:
                self.logger.log("alert", level, reasons=reasons[level])
                self.last_alert[level] = now
