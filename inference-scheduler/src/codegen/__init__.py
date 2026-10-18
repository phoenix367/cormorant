# Copyright 2025 GradeBuilder SL
# SPDX-License-Identifier: Apache-2.0

"""C code generator package for the inference scheduler."""

from ._core     import _CoreMixin
from ._header   import _HeaderMixin
from ._source   import _SourceMixin
from ._buf_impl import _BufImplMixin
from ._simulate import _SimulateMixin
from ._test     import _TestMixin
from ._cmake    import _CmakeMixin


class CodeGenerator(
    _HeaderMixin, _SourceMixin, _BufImplMixin,
    _SimulateMixin, _TestMixin, _CmakeMixin,
    _CoreMixin,
):
    """C code generator for the inference scheduler.  See _core.py for __init__."""
