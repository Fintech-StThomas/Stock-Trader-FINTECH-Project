import os

os.environ.setdefault("LIGHTHOUSE_OFFLINE", "1")      # the collectors refuse live requests in tests
