import subprocess
import tempfile
import pathlib

import pytest
import numpy as np
import pandas as pd

from RootInteractive.Tools.compressArray import compressArray

@pytest.mark.feature("ENC.compression.delta")
@pytest.mark.feature("ENC.compression.sinh")
@pytest.mark.backend("node")
@pytest.mark.layer("unit")
def test_serializationutils():
    temp_dir = tempfile.gettempdir()
    cwd = pathlib.Path(__file__).parent.resolve()
    subprocess.run(["node", "node_modules/typescript/bin/tsc",
                    '--module', 'None',
                    '--target', 'ES2020',
                    '--outDir', str(temp_dir),
                    f'{cwd}/SerializationUtils.ts'], check=True, cwd=cwd)
    result = subprocess.run(['node', '--experimental-modules',
                             f'{cwd}/test_serializationutils.mjs', str(temp_dir)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("All serializationutils tests passed")
    else:
        raise RuntimeError(result.stderr)

if __name__ == "__main__":
    test_serializationutils()