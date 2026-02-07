import subprocess
import tempfile
import pathlib
import json
import base64
import sys

import pytest
import numpy as np

from RootInteractive.Tools.compressArray import compressArray
    
rng = np.random.default_rng(42)
TEST_CASES = {
    "default": rng.random(100) + .5,
    "integers": rng.integers(0,100,100).astype(np.int32),
    "all_nan": np.array([np.nan] * 100),
    "has_infinity": np.array([0.1,1e7,-14,np.nan,np.inf,-np.inf]).astype(np.float64)
}

test_b64 = {i: base64.b64encode(TEST_CASES[i]).decode('utf-8') for i in TEST_CASES.keys()}
dtypes = {i: testcase.dtype.name for i, testcase in TEST_CASES.items()}
lengths = {i: len(testcase) for i, testcase in TEST_CASES.items()}

def run_test_in_node(data, temp_dir):
    cwd = pathlib.Path(__file__).parent.resolve()
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_json_file:
        json.dump(data, temp_json_file)
        temp_json_file.flush()
        result = subprocess.run(['node', f'{cwd}/test_serialization_integration.mjs', str(temp_dir), temp_json_file.name],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"JavaScript test failed:\n{result.stderr}")


@pytest.mark.feature("ENC.compression.delta")
@pytest.mark.feature("ENC.compression.sinh")
@pytest.mark.feature("ENC.compression.zip")
@pytest.mark.backend("node")
@pytest.mark.layer("integration")
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
    
@pytest.mark.feature("ENC.base64.float64")
@pytest.mark.feature("ENC.base64.int32")
@pytest.mark.feature("ENC.compression.zip")
@pytest.mark.feature("ENC.compression.roundtrip")
@pytest.mark.backend("node")
@pytest.mark.layer("invariance")
def test_compression_simple():
    temp_dir = tempfile.gettempdir()
    cwd = pathlib.Path(__file__).parent.resolve()
    subprocess.run(["node", "node_modules/typescript/bin/tsc",
                    '--module', 'None',
                    '--target', 'ES2020',
                    '--outDir', str(temp_dir),
                    f'{cwd}/SerializationUtils.ts'], check=True, cwd=cwd) 
    
    arrayCompression = ["zip", "base64"]
    compressed = {i: compressArray(testcase, arrayCompression) for i, testcase in TEST_CASES.items()}
    data_exported = [{
        "meta":{
            "name": i,
            "enableDithering": False,
            "byteorder": sys.byteorder,
            "dtype": dtypes[i],
            "length": lengths[i],
            "abs_tol": 0,
            "rel_tol": 0,
            "nan_equal": True
        },
        "pipeline": compressed[i]["history"],
        "data_compressed": compressed[i]["array"],
        "data_ref": test_b64[i]
    } for i, testcase in TEST_CASES.items()]
    run_test_in_node(data_exported, temp_dir)


@pytest.mark.feature("ENC.base64.float64")
@pytest.mark.feature("ENC.base64.int32")
@pytest.mark.feature("ENC.compression.zip")
@pytest.mark.feature("ENC.compression.roundtrip")
@pytest.mark.feature("ENC.compression.relative")
@pytest.mark.backend("node")
@pytest.mark.layer("invariance")
def test_compression_relative16():
    temp_dir = tempfile.gettempdir()
    cwd = pathlib.Path(__file__).parent.resolve()
    subprocess.run(["node", "node_modules/typescript/bin/tsc",
                    '--module', 'None',
                    '--target', 'ES2020',
                    '--outDir', str(temp_dir),
                    f'{cwd}/SerializationUtils.ts'], check=True, cwd=cwd) 
    
    arrayCompression = [("relative", 16), "zip", "base64"]
    compressed = {i: compressArray(testcase, arrayCompression) for i, testcase in TEST_CASES.items()}
    data_exported = [{
        "meta":{
            "name": i,
            "enableDithering": False,
            "byteorder": sys.byteorder,
            "dtype": dtypes[i],
            "length": lengths[i],
            "abs_tol": 1e-10,
            "rel_tol": 1/2**16,
            "nan_equal": True
        },
        "pipeline": compressed[i]["history"],
        "data_compressed": compressed[i]["array"],
        "data_ref": test_b64[i]
    } for i, testcase in TEST_CASES.items()]    
    run_test_in_node(data_exported, temp_dir)

@pytest.mark.feature("ENC.base64.float64")
@pytest.mark.feature("ENC.base64.int32")
@pytest.mark.feature("ENC.compression.zip")
@pytest.mark.feature("ENC.compression.roundtrip")
@pytest.mark.feature("ENC.compression.delta")
@pytest.mark.backend("node")
@pytest.mark.layer("invariance")
def test_compression_delta():
    temp_dir = tempfile.gettempdir()
    cwd = pathlib.Path(__file__).parent.resolve()
    subprocess.run(["node", "node_modules/typescript/bin/tsc",
                    '--module', 'None',
                    '--target', 'ES2020',
                    '--outDir', str(temp_dir),
                    f'{cwd}/SerializationUtils.ts'], check=True, cwd=cwd) 
    
    arrayCompression = [("delta", .3), "zip", "base64"]
    compressed = {i: compressArray(testcase, arrayCompression) for i, testcase in TEST_CASES.items()}
    data_exported = [{
        "meta":{
            "name": i,
            "enableDithering": False,
            "byteorder": sys.byteorder,
            "dtype": dtypes[i],
            "length": lengths[i],
            "abs_tol": .15,
            "rel_tol": 1e-10,
            "nan_equal": True
        },
        "pipeline": compressed[i]["history"],
        "data_compressed": compressed[i]["array"],
        "data_ref": test_b64[i]
    } for i, testcase in TEST_CASES.items()]    
    run_test_in_node(data_exported, temp_dir)


@pytest.mark.feature("ENC.base64.float64")
@pytest.mark.feature("ENC.base64.int32")
@pytest.mark.feature("ENC.compression.zip")
@pytest.mark.feature("ENC.compression.roundtrip")
@pytest.mark.feature("ENC.compression.sinh")
@pytest.mark.backend("node")
@pytest.mark.layer("invariance")
def test_compression_sinh():
    temp_dir = tempfile.gettempdir()
    cwd = pathlib.Path(__file__).parent.resolve()
    subprocess.run(["node", "node_modules/typescript/bin/tsc",
                    '--module', 'None',
                    '--target', 'ES2020',
                    '--outDir', str(temp_dir),
                    f'{cwd}/SerializationUtils.ts'], check=True, cwd=cwd) 
    
    sigma0 = 1e-3
    sigma1 = 10
    arrayCompression = [("sqrt_scaling",sigma0,sigma1), "zip", "base64"]
    compressed = {i: compressArray(testcase, arrayCompression) for i, testcase in TEST_CASES.items()}
    data_exported = [{
        "meta":{
            "name": i,
            "enableDithering": False,
            "byteorder": sys.byteorder,
            "dtype": dtypes[i],
            "length": lengths[i],
            "abs_tol": .5 * sigma0 * sigma0 * sigma1,
            "rel_tol": .5 * sigma0,
            "nan_equal": True
        },
        "pipeline": compressed[i]["history"],
        "data_compressed": compressed[i]["array"],
        "data_ref": test_b64[i]
    } for i, testcase in TEST_CASES.items()]    
    run_test_in_node(data_exported, temp_dir)


if __name__ == "__main__":
    #test_serializationutils()
    #test_compression_simple()
    #test_compression_relative16()
    test_compression_sinh()
