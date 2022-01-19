from argparse import ArgumentError
import ast
from msilib.schema import Error

JAVASCRIPT_GLOBALS = {
    "sin": "Math.sin"
}

# This seems really stupid and overengineered
class ColumnEvaluator:
    def __init__(self, context, cdsDict, paramDict, funcDict, code, useEval=True):
        self.cdsDict = cdsDict
        self.paramDict = paramDict
        self.funcDict = funcDict
        self.context = context
        self.dependencies = set()
        self.code = code
        self.useEval = useEval
        self.isSource = True 

    def visit(self, node):
        if isinstance(node, ast.Attribute):
            return self.visit_Attribute(node)
        elif isinstance(node, ast.Call):
            return self.visit_Call(node)
        elif isinstance(node, ast.Name):
            return self.visit_Name(node)
        elif isinstance(node, ast.Num):
            return self.visit_Num(node)
        else:
            return self.eval_fallback(node)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Functions in variables list can only be specified by names")
        if node.func.id in self.funcDict:
            args = []
            for iArg in node.args:
                args.append(self.visit(iArg))
            return {
                "value": {
                    "func": node.func.id,
                    "args": args
                },
                "type": "client_function",
                "name": self.code
            }
        if node.func.id in JAVASCRIPT_GLOBALS and not self.useEval:
            #TODO: Add code generation
            args = []
            implementation = JAVASCRIPT_GLOBALS[node.func.id] + '('
            for iArg in node.args:
                args.append(self.visit(iArg))
                implementation += args[-1]["implementation"]
            implementation += ')'
        return self.eval_fallback(node)

    def visit_Num(self, node: ast.Num):
        # Kept for compatibility with old Python
        self.isSource = False
        return {
            "name": self.code,
            "type": "constant",
            "value": node.n
        }

    def visit_Attribute(self, node: ast.Attribute):
        # TODO: When adding joins on client, expand the functionality of this
        if not isinstance(node.value, ast.Name):
            raise ValueError("Columns can only be selected from X")
        if self.context is not None:
            if node.value.id != self.context:
                raise ValueError("Cannot jump out of already entered context within vairable parsing")
        if node.value.id not in self.cdsDict:
            raise KeyError("Column not found")
        self.context = node.value.id
        if self.cdsDict[self.context]["type"] in ["histogram", "histo2d", "histoNd"]:
            return self.visit_Name_histogram(node.attr)
        self.dependencies.add((self.context, node.attr))
        return {
            "name": node.attr,
            "type": "column"
        }

    def visit_Name(self, node: ast.Name):
        if node.id in self.paramDict:
            self.isSource = False
            if "options" in self.paramDict[node.id]:
                for iOption in self.paramDict[node.id]["options"]:
                    self.dependencies.add((self.context, iOption))
            return {
                "name": node.id,
                "type": "parameter"
            }
        if "data" in self.cdsDict[self.context] and node.id not in self.cdsDict[self.context]["data"]:
            raise NameError("Column not defined: " + node.id)
        if self.cdsDict[self.context]["type"] in ["histogram", "histo2d", "histoNd"]:
            return self.visit_Name_histogram(node.id)
        self.dependencies.add((self.context, node.id))
        return {
            "name": node.id,
            "type": "column"
        }
    
    def visit_Name_histogram(self, id: str):
        self.isSource = False
        histogram = self.cdsDict[self.context]
        for i in histogram["variables"]:
            self.dependencies.add((histogram["source"], i))
        if "weights" in histogram:
            self.dependencies.add((histogram["source"], histogram["weights"]))
        if "histograms" in histogram and id in histogram["histograms"]:
            self.dependencies.add((histogram["source"], histogram["histograms"][id]["weights"]))
        return {
            "name": id,
            "type": "column"
        }        

    def eval_fallback(self, node):
        if "data" not in self.cdsDict[self.context] or not self.useEval:
            raise NotImplementedError("Feature not implemented for tables on client: " + self.code)
        self.dependencies.add((self.context, self.code[node.col_offset:node.end_col_offset]))
        code = compile(ast.Expression(body=node), self.code, "eval")
        return {
            "name": self.code,
            "value": eval(code, {}, self.cdsDict[self.context]["data"]),
            "type": "server_derived_column"
            }
        

def getOrMakeColumns(variableNames, context = None, cdsDict: dict = {None: {}}, paramDict: dict = {}, funcDict: dict = {},
                     memoizedColumns: dict = {}, forbiddenColumns: set = set()):
    if variableNames is None:
        return variableNames, context, memoizedColumns, set()
    if not isinstance(variableNames, list):
        variableNames = [variableNames]
    if not isinstance(context, list):
        context = [context]
    nvars = len(variableNames)
    n_context = len(context)
    variables = []
    ctx_updated = []
    used_names = set()
    for i in range(max(len(variableNames), len(context))):
        i_var = variableNames[i % nvars]
        i_context = context[i % n_context] 
        if i_context in memoizedColumns and i_var in memoizedColumns[i_context]:
            variables.append(memoizedColumns[i_context][i_var])
            ctx_updated.append(i_context)
            continue
        if (i_context, i_var) in forbiddenColumns:
            # Unresolvable cyclic dependency in table - stack trace should tell exactly what went wrong
            raise RuntimeError("Cyclic dependency detected")
        queryAST = ast.parse(i_var, mode="eval")
        evaluator = ColumnEvaluator(i_context, cdsDict, paramDict, funcDict, i_var)
        column = evaluator.visit(queryAST.body)
        variables.append(column)
        i_context = evaluator.context
        if evaluator.isSource:
            used_names.update(evaluator.dependencies)
        else:
            direct_dependencies = list(evaluator.dependencies)
            dependency_columns = [i[1] for i in direct_dependencies]
            dependency_tables = [i[0] for i in direct_dependencies]
            _, _, memoizedColumns, sources_local = getOrMakeColumns(dependency_columns, dependency_tables, cdsDict, paramDict, funcDict, 
                                                                    memoizedColumns, forbiddenColumns | {(i_context, i_var)})
            used_names.update(sources_local)
        ctx_updated.append(i_context)
        if i_context in memoizedColumns:
            memoizedColumns[i_context][i_var] = column
        else:
            memoizedColumns[i_context] = {i_var: column}
        
    return variables, ctx_updated, memoizedColumns, used_names