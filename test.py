import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "build"))

import example

print(example.add(5, 7))  # Output: 12
