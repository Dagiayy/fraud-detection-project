# src/api/rate_limiter.py
import time

class RateLimiter:
    def __init__(self, max_requests: int = 100):
        self.max_requests = max_requests
        self.requests = []

    def is_allowed(self) -> bool:
        now = time.time()
        self.requests = [t for t in self.requests if now - t < 60.0]
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
