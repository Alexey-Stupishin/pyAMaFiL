#!/usr/bin/env python3
from setuptools import Extension, setup
from pathlib import Path
import pdb
import sys

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
#sys.exit(1)

setup(
    ext_modules=[
        Extension(
            name="pyAMaFiL.WWNLFFFReconstruction",
            sources = source_files,
            include_dirs = header_dirs,
            language = "c++",
            extra_compile_args = ["-std=c++11", "-fPIC", "-O2", "-fpermissive"],
            extra_link_args = ["-shared", "-lm", "-pthread"]
        ),
    ]
)