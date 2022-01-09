import ast

from bokeh.models.layouts import Panel

# This seems really stupid and overengineered, might as well use LLVM
class ColumnEvaluator:
    def __init__(self, context, cdsDict, paramDict, funcDict, code, fallback_eval=True):
        self.cdsDict = cdsDict
        self.paramDict = paramDict
        self.funcDict = funcDict
        self.context = context
        self.usedNames = set()
        self.code = code
        self.javascript = ""

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
            self.usedNames.add(self.code)
            code = compile(ast.Expression(body=node), self.code, "eval")
            return {
                "name": self.code,
                "value": eval(code, {}, self.cdsDict[self.context]),
                "type": "server_derived_column"
                }

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
        else:
            self.usedNames.add(self.code)
            code = compile(ast.Expression(body=node), self.code, "eval")
            return {
                "name": self.code,
                "value": eval(code, {}, self.cdsDict[self.context]),
                "type": "server_derived_column"
                }

    def visit_Num(self, node: ast.Num):
        # This is deprecated, kept for compatibility with old Python
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
        self.usedNames.add(node.attr)
        return {
            "name": node.attr,
            "type": "column"
        }

    def visit_Name(self, node: ast.Name):
        if node.id in self.paramDict:
            return {
                "name": node.id,
                "type": "parameter"
            }
        self.usedNames.add(node.id)
        return {
            "name": node.id,
            "type": "column"
        }
    
    def eval_fallback(self, node):
        self.usedNames.add(self.code)
        code = compile(ast.Expression(body=node), self.code, "eval")
        return {
            "name": self.code,
            "value": eval(code, {}, self.cdsDict[self.context]),
            "type": "server_derived_column"
            }
        

def getOrMakeColumns(variableNames, context = None, cdsDict: dict = {None: {}}, paramDict: dict = {}, funcDict: dict = {}, memoizedColumns: dict = {}):
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
        queryAST = ast.parse(i_var, mode="eval")
        evaluator = ColumnEvaluator(i_context, cdsDict, paramDict, funcDict, i_var)
        column = evaluator.visit(queryAST.body)
        variables.append(column)
        i_context = evaluator.context
        used_names.update({(i_context, i) for  i in evaluator.usedNames})
        ctx_updated.append(i_context)
        if i_context in memoizedColumns:
            memoizedColumns[i_context][i_var] = column
        else:
            memoizedColumns[i_context] = {i_var: column}
        
    return variables, ctx_updated, memoizedColumns, used_names