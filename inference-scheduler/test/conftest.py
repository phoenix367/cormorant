# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

import os
import sys
# Make the inference-scheduler root importable in all test files
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
