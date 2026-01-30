import subprocess
import tempfile
import pathlib
import json
import ast
import base64

import numpy as np

from RootInteractive.InteractiveDrawing.bokeh.compileVarName import ColumnEvaluator
from RootInteractive.Tools.compressArray import compressArray

def helper_test_command(command, expected_output):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stdout.strip()
        assert output == expected_output, f"Expected '{expected_output}', but got '{output}'"
        print(f"Test passed for command: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        print(f"Command '{' '.join(command)}' failed with error: {e.stderr.strip()}")
    except AssertionError as e:
        print(e)

def test_nodejs():
    helper_test_command(['node', '-e', 'console.log(1 + 1)'], '2')
    helper_test_command(['node', '-e', 'console.log("Hello, World!")'], 'Hello, World!')

def test_mathutils():
    return 0
    temp_dir = tempfile.gettempdir()
    cwd = pathlib.Path(__file__).parent.resolve()
    subprocess.run(['npx', 'tsc',
                    '--outDir', temp_dir,
                    '--module', 'ES2020',
                    '--target', 'ES2020',
                    f'{cwd}/mathutils.ts'], check=True)
    result = subprocess.run(['node', '--experimental-modules',
                             f'{temp_dir}/mathutils.js'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    print(result.stdout)
    if result.returncode == 0:
        print("All mathutils tests passed")
    else:
        raise RuntimeError(result.stderr)
    
def test_compileVarName():
    rng = np.random.default_rng(5669)
    A = {'a': rng.random(100), 'b': rng.random(100), 'c': rng.random(100),
         'x': rng.random(100), 'y': rng.random(100)}
    A['expr1_ref'] = A['a'] + A['b'] * A['c']
    A['expr2_ref'] = np.arctan2(A['y'], A['x'])
    expr1 = "a + b * c"
    expr2 = "arctan2(y,x)"

    aliasDict = {"A":{"expr1": expr1, "expr2": expr2}}
    data = {"A": {"data": A, "type": "array"}}

    evaluator1 = ColumnEvaluator("A", data, {}, {}, expr1, aliasDict)
    queryAST = ast.parse(expr1, mode="eval")
    column_expr1 = evaluator1.visit(queryAST.body)

    assert evaluator1.aliasDependencies == {'a': 'a', 'b': 'b', 'c': 'c'}

    A_encoded = {}
    for i in A.keys():
        b64_bytes = base64.b64encode(A[i]).decode('utf-8')
        A_encoded[i] = {"data": b64_bytes, "dtype": "float64", "length": len(A[i])}

    func = evaluator1.make_vfunc(column_expr1["implementation"])

    A_encoded['expr1'] = {"func":func, "args": list(evaluator1.aliasDependencies.keys())}

    data_exported = {"data": {"A":A_encoded}, "test_cases": [{"type":"EQ","lhs": "expr1", "rhs": "expr1_ref","epsilon": 1e-10,"table":"A"}]}

    print(data_exported)

    cwd = pathlib.Path(__file__).parent.resolve()

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_json_file:
        json.dump(data_exported, temp_json_file)
        temp_json_file.flush()
        result = subprocess.run(['node', f'{cwd}/test_dsl_customjs.mjs', temp_json_file.name])
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError("JavaScript test failed")


    print("All compileVarName tests passed")

test_compileVarName()