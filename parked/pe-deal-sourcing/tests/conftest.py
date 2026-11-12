import os

os.environ.setdefault("LIGHTHOUSE_OFFLINE", "1")      # the collector refuses live requests in tests
