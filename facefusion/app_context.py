import os
import sys

from facefusion.typing import AppContext


def detect_app_context() -> AppContext:
    # Need to force this for AUTO
    return 'ui'
