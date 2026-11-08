"""`python -m veridx` 等价于 `veridx`。"""

import sys

from veridx.cli import main

if __name__ == "__main__":
    sys.exit(main())
