# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import keyword
import os
import re
import threading
import uuid
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Dict, List, Set


class IndentedBuffer:
    """A buffer for building indented text output (e.g. generated code).

    Maintains a list of lines and a current indentation level. Lines are
    written with automatic indentation prefix applied.
    """

    tabwidth = 4

    def __init__(self, initial_indent: int = 0):
        self._lines: List[str] = []
        self._level = initial_indent

    # ---- output ----

    def getvalue(self) -> str:
        buf = StringIO()
        for line in self._lines:
            buf.write(line)
            buf.write("\n")
        return buf.getvalue()

    def clear(self) -> None:
        self._lines.clear()

    def __bool__(self) -> bool:
        return bool(self._lines)

    # ---- prefix ----

    def _prefix(self) -> str:
        return " " * (self._level * self.tabwidth)

    def prefix(self) -> str:
        return self._prefix()

    # ---- write helpers ----

    def newline(self) -> None:
        """Write a blank line (no indentation)."""
        self._lines.append("")

    def writeline(self, line: str) -> None:
        """Write a single line. Non-blank lines are prefixed with current indent."""
        if line.strip():
            self._lines.append(f"{self._prefix()}{line}")
        else:
            self._lines.append("")

    def writelines(self, lines) -> None:
        """Write a sequence of lines."""
        for line in lines:
            self.writeline(line)

    def writemultiline(self, s: str) -> None:
        """Write a multi-line string, splitting on newlines."""
        self.writelines(s.splitlines())

    def tpl(self, format_str: str, **kwargs) -> None:
        """Write a template string: format it, then write each line."""
        formatted = format_str.format(**kwargs)
        for line in formatted.strip().splitlines():
            self.writeline(line.replace("\t", " " * self.tabwidth))

    # ---- indent ----

    def indent(self, offset: int = 1):
        """Context manager that temporarily increases indentation level.

        Usage::

            with buf.indent():
                buf.writeline("inside")
        """
        return self._IndentCtx(self, offset)

    class _IndentCtx:
        def __init__(self, buf: "IndentedBuffer", offset: int):
            self._buf = buf
            self._offset = offset

        def __enter__(self):
            self._buf._level += self._offset
            return self

        def __exit__(self, *args):
            self._buf._level -= self._offset


class NameSpace:
    """Generates unique, valid Python identifiers from candidate names.

    Sanitizes input, avoids Python keywords and builtins, and appends numeric
    suffixes to prevent collisions.
    """

    def __init__(self):
        self._used: Set[str] = set()
        self._counters: Dict[str, int] = defaultdict(int)
        self._sanitize_re = re.compile(r"[^0-9a-zA-Z_]+")
        self._suffix_re = re.compile(r"(.*)_(\d+)$")

    def create_name(self, candidate: str) -> str:
        """Return a unique valid Python identifier based on *candidate*."""
        # strip illegal characters
        candidate = self._sanitize_re.sub("_", candidate)

        if not candidate:
            candidate = "_unnamed"

        if candidate[0].isdigit():
            candidate = f"_{candidate}"

        # parse base name and optional numeric suffix
        m = self._suffix_re.match(candidate)
        if m is None:
            base, num = candidate, None
        else:
            base, num_str = m.group(1, 2)
            num = int(num_str)

        candidate = base if num is None else f"{base}_{num}"
        if num is None:
            num = self._counters[base]

        # advance until unique and legal
        while candidate in self._used or self._illegal(candidate):
            num += 1
            candidate = f"{base}_{num}"

        self._used.add(candidate)
        self._counters[base] = num
        return candidate

    def _illegal(self, name: str) -> bool:
        if name in keyword.kwlist:
            return True
        if name in builtins.__dict__:
            return True
        return False


def write_atomic(
    path_: str,
    content: str,
    make_dirs: bool = False,
    encoding: str = "utf-8",
) -> None:
    """Atomically write *content* to *path_* via a temporary file + rename."""
    path = Path(path_)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = (
        path.parent / f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    with tmp_path.open("wt", encoding=encoding) as f:
        f.write(content)
    tmp_path.replace(path)
