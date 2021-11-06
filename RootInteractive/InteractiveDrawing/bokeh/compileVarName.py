import ast

class ColumnEvaluator:
    def __init__(self, context, cdsDict, paramDict, aliasDict):
        self.cdsDict = cdsDict
        self.paramDict = paramDict
        self.aliasDict = aliasDict
        self.context = context

    def visit(self, node):
        if isinstance(node, ast.Attribute):
            return self.visit_Attribute(node)
        elif isinstance(node, ast.Call):
            return self.visit_Call(node)
        elif isinstance(node, ast.Name):
            return self.visit_Name(node)
        else:
            code = compile(ast.Expression(body=node), "<string>", "eval")
            return {
                "value": eval(code, globals(), self.cdsDict[self.context]),
                "type": "server_derived_column"
                }        

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Functions in variables list can only be specified by names")
        if node.func.id in self.aliasDict:
            args = []
            for iArg in node.args:
                args.append(self.visit(iArg))
            return {
                "value": {
                    "func": node.func.id,
                    "args": args
                },
                "type": "client_function"
            }
        else:
            code = compile(ast.Expression(body=node), "<string>", "eval")
            return {
                "name": "anonymous",
                "value": eval(code, globals(), self.cdsDict[self.context]),
                "type": "server_derived_column"
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
        return {
            "name": node.id,
            "type": "column"
        }
        
        

def getOrMakeColumns(variableNames, context = None, cdsDict: dict = {None: {}}, paramDict: dict = {}, aliasDict: dict = {}, memoizedColumns: dict = {}):
    if not isinstance(variableNames, list):
        variableNames = [variableNames]
    if not isinstance(context, list):
        context = [context]
    nvars = len(variableNames)
    n_context = len(context)
    variables = []
    ctx_updated = []
    for i in range(max(len(variableNames), len(context))):
        i_var = variableNames[i % nvars]
        i_context = context[i % n_context] 
        if (i_var, i_context) in memoizedColumns:
            variables.append(memoizedColumns[(i_var, i_context)])
        queryAST = ast.parse(i_var, mode="eval")
        evaluator = ColumnEvaluator(i_context, cdsDict, paramDict, aliasDict)
        column = evaluator.visit(queryAST.body)
        variables.append(column)
        i_context = evaluator.context
        ctx_updated.append(i_context)
        memoizedColumns[(i_var, i_context)] = column
        
    return variables, ctx_updated, memoizedColumns