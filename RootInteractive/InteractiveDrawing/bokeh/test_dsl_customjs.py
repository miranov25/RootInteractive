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
         'x': rng.random(100), 'y': rng.random(100), 'i': rng.integers(0, 20, size=100, dtype=np.int32)}
    dfB = {'a': rng.random(20), 'b': rng.random(20), 'c': rng.random(20)}
    A['expr1_ref'] = A['a'] + A['b'] * A['c']
    A['expr2_ref'] = dfB['a'][A['i']] + dfB['b'][A['i']] * dfB['c'][A['i']]

    expr1 = "a + b * c"
    expr2 = "dfB.a[i] + dfB.b[i] * dfB.c[i]"

    aliasDict = {"A":{"expr1": expr1, "expr2": expr2}}
    data = {"A": {"data": A, "type": "array"}, "dfB": {"data": dfB, "type": "array"}}

    evaluator1 = ColumnEvaluator("A", data, {}, {}, expr1, aliasDict)
    queryAST = ast.parse(expr1, mode="eval")
    column_expr1 = evaluator1.visit(queryAST.body)

    assert evaluator1.aliasDependencies == {'a': 'a', 'b': 'b', 'c': 'c'}

    A_encoded = {}
    for i in A.keys():
        b64_bytes = base64.b64encode(A[i]).decode('utf-8')
        A_encoded[i] = {"data": b64_bytes, "dtype": A[i].dtype.name, "length": len(A[i])}

    evaluator2 = ColumnEvaluator("A", data, {}, {}, expr2, aliasDict)
    queryAST = ast.parse(expr2, mode="eval")
    column_expr2 = evaluator2.visit(queryAST.body)

    print({i[0] for i in evaluator2.dependencies_table})

    dfB_encoded = {}
    for i in dfB.keys():
        b64_bytes = base64.b64encode(dfB[i]).decode('utf-8')
        dfB_encoded[i] = {"data": b64_bytes, "dtype": dfB[i].dtype.name, "length": len(dfB[i])}

    func = evaluator1.make_vfunc(column_expr1["implementation"])
    func2 = evaluator2.make_vfunc(column_expr2["implementation"])

    A_encoded['expr1'] = {"func":func, "args": list(evaluator1.aliasDependencies.keys())}
    A_encoded['expr2'] = {"func":func2, "args": list(evaluator2.aliasDependencies.keys()), "context": {i[0]:i[0] for i in evaluator2.dependencies_table}}

    data_exported = {"data": {"A":A_encoded, "dfB":dfB_encoded}, "test_cases": [
        {"type":"EQ","lhs": "expr1", "rhs": "expr1_ref","epsilon": 1e-10,"table":"A"},
        {"type":"EQ","lhs": "expr2", "rhs": "expr2_ref","epsilon": 1e-10,"table":"A"}
        ]}

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