#!/usr/bin/env python3

import platform
from setuptools import Extension, setup
from pathlib import Path
import pdb
import sys

link_flags = {
    'Linux': ["-shared", "-lm", "-pthread"],
    'Windows': ["-shared", "-lm", "-pthread"],
    'Darwin': ["-lm", "-pthread"]
}

current_os = platform.system()

if current_os in link_flags:
    extra_link = link_flags[current_os]
else:
    extra_link = []

rp = Path("./AMaFiL")

header_dirs = ["./" + str(rp / p) for p in ("Common/BaseMath",    "Common/BasePhysics",
                                     "Common/General",     "Common/IDL",
                                     "Common/Multithread", "Common/Utils", "sources")]

source_dirs = [rp / p for p in ("Common/BaseMath",
                                "Common/BasePhysics",
                                "Common/Multithread",
                                "Common/Utils", "sources")]

source_files = []
for d in source_dirs:
    source_files.extend([str(d / x.name) for x in sorted(d.glob("*.cpp"))])
#pdb.set_trace()
source_files.append("pyinit.cpp")

setup(
    ext_modules=[
        Extension(
            name="pyAMaFiL.WWNLFFFReconstruction",
            sources = source_files,
            include_dirs = header_dirs,
            export_symbols = ["utilInitialize", "utilSetInt", "utilSetDouble", "utilGetVersion", "mfoNLFFFCore", "mfoGetLines"],
            language = "c++",
            extra_compile_args = ["-std=c++11", "-fPIC", "-O2", "-fpermissive"],
            extra_link_args = extra_link
        ),
    ]
)