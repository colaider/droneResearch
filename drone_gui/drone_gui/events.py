"""An append-only event log that outlives the GUI process.

The point of the panel is to answer "what happened while I was not watching": the board
rebooted, the node died, the drone was armed. Keeping that in memory would lose it on every
reload, so events go to a JSONL file and are read back on start-up.
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time

MAX_SHOWN = 200
MAX_FILE_BYTES = 2 * 1024 * 1024        # rotate once, keep one .1 file


class EventLog:
    def __init__(self, state_dir, keep=MAX_SHOWN):
        self.path = os.path.join(state_dir, "events.jsonl")
        self.lock = threading.Lock()
        self.recent = collections.deque(maxlen=keep)
        os.makedirs(state_dir, exist_ok=True)
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as handle:
                for line in handle.readlines()[-self.recent.maxlen:]:
                    try:
                        self.recent.append(json.loads(line))
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass

    def add(self, message, level="info"):
        entry = {"t": time.time(), "level": level, "message": str(message)}
        with self.lock:
            self.recent.append(entry)
            try:
                if os.path.exists(self.path) and os.path.getsize(self.path) > MAX_FILE_BYTES:
                    os.replace(self.path, self.path + ".1")
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
            except OSError:
                pass            # a full or read-only disk must not take the panel down
        return entry

    def items(self, newest_first=True):
        with self.lock:
            out = list(self.recent)
        return list(reversed(out)) if newest_first else out
