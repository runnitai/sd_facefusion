import time


class StopWatch:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.times = {}
            self._initialized = True

    def start(self, key: str = None):
        start_time = time.time()
        if key is None:
            key = f"start_{len(self.times)}"
        if key in self.times:
            key_idx = 1
            while f"{key}_{key_idx}" in self.times:
                key_idx += 1
            key = f"{key}_{key_idx}"
        self.times[key] = time.time()

    def stop(self, key: str = None):
        if key is None:
            key = f"stop_{len(self.times)}"
        if key in self.times:
            raise ValueError(f"Key '{key}' already exists. Choose a unique key.")
        self.times[key] = time.time()

    def next(self, key: str):
        self.start(key)

    def get(self, key: str):
        keys = list(self.times.keys())
        if key not in keys:
            raise KeyError(f"Key '{key}' not found.")
        key_idx = keys.index(key)
        if key_idx + 1 >= len(keys):
            return None
        next_key = keys[key_idx + 1]
        return self.times[next_key] - self.times[key]

    def total(self):
        if len(self.times) < 2:
            return 0
        keys = list(self.times.keys())
        return self.times[keys[-1]] - self.times[keys[0]]

    def report(self):
        keys = list(self.times.keys())
        for i, key in enumerate(keys):
            if i + 1 < len(keys):
                next_key = keys[i + 1]
                duration = self.times[next_key] - self.times[key]
                print(f"{key} -> {next_key}: {duration:.4f} seconds")
            else:
                print(f"{key}: {self.times[key]:.4f}")
        print(f"Total: {self.total():.4f} seconds")
        self.times = {}
