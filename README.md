# pyAMaFiL

#### Testing development version

```bash
git submodule update --init --remote --recursive
pip install .
```

Or, to install an [editable](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#development-mode) version:

```bash
pip install --editable .
```

Importing class from Python

```python
from pyAMaFiL.mag_field_wrapper import MagFieldWrapper

maglib = MagFieldWrapper()
```