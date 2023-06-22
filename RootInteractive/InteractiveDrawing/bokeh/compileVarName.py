import ast
import numpy as np

from RootInteractive.InteractiveDrawing.bokeh.CustomJSNAryFunction import CustomJSNAryFunction
from bokeh.models.callbacks import CustomJS

JAVASCRIPT_GLOBALS = {
    "sin": "Math.sin",
    "cos": "Math.cos",
    "arcsin": "Math.asin",
    "arccos": "Math.acos",
    "arctan": "Math.atan",
    "exp": "Math.exp",
    "log": "Math.log",
    "log2": "Math.log2",
    "log10": "Math.log10",
    "sqrt": "Math.sqrt",
    "abs": "Math.abs",
    "sinh": "Math.sinh",
    "cosh": "Math.cosh",
    "arctan2": "Math.atan2",
    "arccosh": "Math.acosh",
    "arcsinh": "Math.asinh",
    "arctanh": "Math.atanh"
}

JAVASCRIPT_N_ARGS = {
    "sin": 1,
    "cos": 1,
    "arcsin": 1,
    "arccos": 1,
    "arctan": 1,
    "exp": 1,
    "log": 1,
    "log2": 1,
    "log10": 1,
    "sqrt": 1,
    "abs": 1,
    "sinh": 1,
    "cosh": 1,
    "arctan2": 2,
    "arccosh": 1,
    "arcsinh": 1,
    "arctanh": 1
}

math_functions = {
    "sin": np.sin,
    "cos": np.cos,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "exp": np.exp,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.absolute,
    "sinh": np.sinh,
    "cosh": np.sinh,
    "arctan2": np.arctan2,
    "arccosh": np.arccosh,
    "arcsinh": np.arcsinh,
    "arctanh": np.arctanh
}

class ColumnEvaluator:
    # This class walks the Python abstract syntax tree of the expressions to detect its dependencies
    def __init__(self, context, cdsDict, paramDict, funcDict, code, aliasDict, firstGeneratedID=0):
        self.cdsDict = cdsDict
        self.paramDict = paramDict
        self.funcDict = funcDict
        self.context = context
        self.dependencies = set()
        self.paramDependencies = set()
        self.aliasDependencies = set()
        self.firstGeneratedID = firstGeneratedID
        self.code = code
        self.isSource = True 
        self.aliasDict = aliasDict
        self.isAuto = False
        self.locals = []

    def visit(self, node):
        if isinstance(node, ast.Attribute):
            return self.visit_Attribute(node)
        elif isinstance(node, ast.Call):
            return self.visit_Call(node)
        elif isinstance(node, ast.Name):
            return self.visit_Name(node)
        elif isinstance(node, ast.Num):
            return self.visit_Num(node)
        elif isinstance(node, ast.BinOp):
            return self.visit_BinOp(node)
        elif isinstance(node, ast.UnaryOp):
            return self.visit_UnaryOp(node)
        elif isinstance(node, ast.Compare):
            return self.visit_Compare(node)
        elif isinstance(node, ast.BoolOp):
            return self.visit_BoolOp(node)
        elif isinstance(node, ast.IfExp):
            return self.visit_IfExp(node)
        elif isinstance(node, ast.Lambda):
            return self.visit_Lambda(node)
        else:
            return self.eval_fallback(node)

    def visit_Call(self, node: ast.Call):
        left = self.visit(node.func)
        if left["type"] == "parameter":
            # TODO: Make a parameter unswitcher - should help improve performance - but same can also apply to other scalars?
            pass
        args = []
        implementation = left['implementation'] + '('
        for iArg in node.args:
            args.append(self.visit(iArg))
        implementation += ", ".join([i["implementation"] for i in args])
        implementation += ')'
        return {
            "implementation": implementation,
            "type": "javascript",
            "name": self.code
        }

    def visit_Num(self, node: ast.Num):
        # Kept for compatibility with old Python
        return {
            "name": str(node.n),
            "implementation": str(node.n),
            "type": "constant",
            "value": node.n
        }

    def visit_Attribute(self, node: ast.Attribute):
        if self.context in self.aliasDict and node.attr in self.aliasDict[self.context]:
            # We have an alias in aliasDict
            self.isSource = False
            if "expr" in self.aliasDict[self.context][node.attr]:
                self.dependencies.add((self.context, self.aliasDict[self.context][node.attr]["expr"]))
                return {
                    "name": node.attr,
                    "implementation": node.attr,
                    "type": "alias"
                }
            elif self.aliasDict[self.context][node.attr].get("fields", None) is not None:
                for i in self.aliasDict[self.context][node.attr]["fields"]:
                    self.dependencies.add((self.context, i))
            self.aliasDependencies.add(node.attr)
            return {
                "name": node.attr,
                "implementation": node.attr,
                "type": "alias"
            }
        if self.cdsDict[self.context]["type"] == "join":
            # In this case, we can have chained attributes
            attrChain = []
            if isinstance(node.value, ast.Attribute) or isinstance(node.value, ast.Name):
                attrChain = self.visit(node.value)["attrChain"]
            if node.attr in self.cdsDict:
                return {
                    "name": node.attr,
                    "implementation": node.attr,
                    "type": "table",
                    "attrChain": attrChain + [node.attr]
                }
            self.isSource = False
            cds = self.cdsDict[self.context]
            # Joins always depend on the join key
            for i in cds["left_on"]:
                self.dependencies.add((cds["left"], i))
            for i in cds["right_on"]:
                self.dependencies.add((cds["right"], i))
            cds_used = 0
            if attrChain[0] == self.context:
                cds_used = 1
            if(cds_used < len(attrChain)):
                if attrChain[cds_used] in [cds["left"], cds["right"]]:
                    self.dependencies.add((attrChain[cds_used], node.attr))
            self.aliasDependencies.add(node.attr)
            return {
                "name": node.attr,
                "implementation": node.attr,
                "type": "column"
            }
        if not isinstance(node.value, ast.Name):
            raise ValueError("Column data source name cannot be a function call")
        if node.value.id != "self":
            if self.context is not None:
                if node.value.id != self.context:
                    raise ValueError("Incompatible data sources: " + node.value.id + "." + node.attr + ", " + self.context)
            if node.value.id not in self.cdsDict:
                raise KeyError("Data source not found: " + node.value.id)
            self.context = node.value.id
        if self.context in self.cdsDict and self.cdsDict[self.context]["type"] == "stack":
            self.isSource = False
            if node.attr != "$source_index":
                for i in self.cdsDict[self.context]["sources_all"]:
                    self.dependencies.add((i, node.attr))
        if self.cdsDict[self.context]["type"] in ["histogram", "histo2d", "histoNd"]:
            return self.visit_Name_histogram(node.attr)
        if self.cdsDict[self.context]["type"] == "projection":
            self.isSource = False
            projection = self.cdsDict[self.context]
            self.dependencies.add((projection["source"], "bin_count"))
        if "data" in self.cdsDict[self.context] and node.attr not in self.cdsDict[self.context]["data"]:
            raise KeyError("Column " + node.attr + " not found in data source " + str(self.cdsDict[self.context]["name"]))
            #return {
            #    "error": KeyError,
            #    "msg": "Column " + id + " not found in data source " + self.cdsDict[self.context]["name"]
            #}           
        self.aliasDependencies.add(node.attr)
        try:
            is_boolean = "data" in self.cdsDict[self.context] and self.cdsDict[self.context]["data"][node.attr].dtype.kind == "b"
        except AttributeError:
            is_boolean = False
        return {
            "name": node.attr,
            "implementation": node.attr,
            "type": "column",
            "is_boolean": is_boolean
        }

    def visit_Name(self, node: ast.Name):
        # There are two cases, either we are selecting the namespace or the column from the current one
        if node.id == "auto":
            self.isSource = False
            self.isAuto = True
            return {
                "name": node.id,
                "implementation": node.id,
                "type": "auto"
            }
        if self.locals and node.id in self.locals[-1]:
            return {
                "name": node.id,
                "implementation": node.id,
                "type": "auto"
            }            
        if node.id in JAVASCRIPT_GLOBALS:
            return {
                "name": node.id,
                "implementation": JAVASCRIPT_GLOBALS[node.id],
                "n_args": JAVASCRIPT_N_ARGS[node.id],
                "type": "js_lambda"
            }
        if node.id in self.funcDict:
            self.isSource = False
            return {
                "name": node.id,
                "implementation": node.id,
                "n_args": len(self.funcDict[node.id]["parameters"]),
                "type": "js_lambda"
            }
        if node.id in self.paramDict:
            self.isSource = False
            if "options" in self.paramDict[node.id]:
                for iOption in self.paramDict[node.id]["options"]:
                    self.dependencies.add((self.context, iOption))
            self.paramDependencies.add(node.id)
            # Detect if parameter is a lambda here? 
            return {
                "name": node.id,
                "implementation": node.id,
                "type": "paramTensor" if isinstance(self.paramDict[node.id]["value"], list) else "parameter"
            }
        if node.id in [self.context, "self"]:
            return {
                "name": node.id,
                "type": "table",
                "attrChain": [node.id]
            }
        if self.cdsDict[self.context]["type"] == "join":
            if node.id in [self.cdsDict[self.context]["left"], self.cdsDict[self.context]["right"]]:
                return {
                    "name": node.id,
                    "type": "table",
                    "attrChain": [node.id]
                }
        attrNode = ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=node.id)
        return self.visit(attrNode)
    
    def visit_Name_histogram(self, id: str):
        self.isSource = False
        histogram = self.cdsDict[self.context]
        histoSource = histogram.get("source", None)
        for i in histogram["variables"]:
            self.dependencies.add((histoSource, i))
        if "weights" in histogram:
            self.dependencies.add((histoSource, histogram["weights"]))
        if "histograms" in histogram and id in histogram["histograms"] and histogram["histograms"][id] is not None and "weights" in histogram["histograms"][id]:
            self.dependencies.add((histoSource, histogram["histograms"][id]["weights"]))
        isOK = (id == "bin_count")
        if self.cdsDict[self.context]["type"] == "histogram":
            if id in ["bin_bottom", "bin_center", "bin_top"]:
                isOK = True
        else:
            if id in ["bin_bottom_{}".format(i) for i in range(len(histogram["variables"]))]:
                isOK = True
            if id in ["bin_center_{}".format(i) for i in range(len(histogram["variables"]))]:
                isOK = True
            if id in ["bin_top_{}".format(i) for i in range(len(histogram["variables"]))]:
                isOK = True         
        if "histograms" in histogram and id in histogram["histograms"]:
            isOK = True
        if not isOK:
            raise KeyError("Column " + id + " not found in histogram " + histogram["name"])
            #return {
            #    "error": KeyError,
            #    "msg": "Column " + id + " not found in histogram " + histogram["name"]
            #}
        self.aliasDependencies.add(id)
        return {
            "name": id,
            "implementation": id,
            "type": "column"
        }        

    def eval_fallback(self, node):
        if "data" not in self.cdsDict[self.context]:
            raise NotImplementedError("Feature not implemented for tables on client: " + self.code + ' ' + ast.dump(node))
        if self.isSource:
            column_id = self.code
        else:
            raise NotImplementedError("Automatically generated JS functions from server derived columns are not supported")
        code = compile(ast.Expression(body=node), self.code, "eval")
        locals = {**self.cdsDict[self.context]["data"], **math_functions}
        return {
            "name": column_id,
            "value": eval(code, {}, locals),
            "type": "server_derived_column"
            }

    def visit_BinOp(self, node):
        op = node.op
        left = self.visit(node.left)
        right = self.visit(node.right)
        is_boolean = False
        if isinstance(op, ast.Add):
            operator_infix = " + "
        elif isinstance(op, ast.Sub):
            operator_infix = " - "
        elif isinstance(op, ast.Mult):
            left_boolean = left.get("is_boolean", False)
            right_boolean = right.get("is_boolean", True)
            is_boolean = left_boolean and right_boolean
            operator_infix = " && " if left_boolean else " * "
        elif isinstance(op, ast.Div) or isinstance(op, ast.FloorDiv):
            operator_infix = " / "
        elif isinstance(op, ast.Mod):
            operator_infix = " % "
        elif isinstance(op, ast.LShift):
            operator_infix = " << "
        elif isinstance(op, ast.RShift):
            operator_infix = " >> "
        elif isinstance(op, ast.RShift):
            operator_infix = " >> "
        elif isinstance(op, ast.BitOr):
            operator_infix = " | "
        elif isinstance(op, ast.BitXor):
            operator_infix = " ^ "
        elif isinstance(op, ast.BitAnd):
            operator_infix = " & "
        elif isinstance(op, ast.Pow):
            operator_infix = "**"
        else:
            raise NotImplementedError(f"Binary operator {ast.dump(op)} not implemented for expressions on the client")
        implementation = f"({left['implementation']}){operator_infix}({right['implementation']})"
        if isinstance(op, ast.FloorDiv):
            implementation = f"(({implementation})|0)"
        return {
            "name": self.code,
            "type": "javascript",
            "implementation": implementation,
            "is_boolean": is_boolean
        }
        
    def visit_UnaryOp(self, node: ast.UnaryOp):
        op = node.op
        if isinstance(op, ast.UAdd):
            operator_prefix = "+"
        elif isinstance(op, ast.USub):
            operator_prefix = "-"
        elif isinstance(op, ast.Not):
            operator_prefix = "!"
        elif isinstance(op, ast.Invert):
            operator_prefix = "~"
        implementation = f"{operator_prefix}({self.visit(node.operand)['implementation']})"
        return {
            "name": self.code,
            "type": "javascript",
            "implementation": implementation
        }

    def visit_Compare(self, node:ast.Compare):
        js_comparisons = []      
        for i, op in enumerate(node.ops): 
            if i==0:
                lhs = self.visit(node.left)["implementation"]
            else:
                lhs = self.visit(node.comparators[i-1])["implementation"]
            rhs = self.visit(node.comparators[i])["implementation"]
            if isinstance(op, ast.Eq):
                op_infix = " === "
            elif isinstance(op, ast.NotEq):
                op_infix = " !== "           
            elif isinstance(op, ast.Lt):
                op_infix = " < "
            elif isinstance(op, ast.LtE):
                op_infix = " <= "
            elif isinstance(op, ast.Gt):
                op_infix = " > "
            elif isinstance(op, ast.GtE):
                op_infix = " >= "
            else:
                raise NotImplementedError(f"Binary operator {ast.dump(op)} not implemented for expressions on the client")
            js_comparisons.append(f"(({lhs}){op_infix}({rhs}))")
        implementation = " && ".join(js_comparisons)
        return {
            "name": self.code,
            "type": "javascript",
            "implementation": implementation,
            "is_boolean": True
        }
    
    def visit_BoolOp(self, node:ast.BoolOp):
        js_values = [f"({self.visit(i)['implementation']})" for i in node.values]
        if isinstance(node.op, ast.And):
            op_infix = " && "
        elif isinstance(node.op, ast.Or):
            op_infix = " || "
        implementation = op_infix.join(js_values)
        return {
            "name": self.code,
            "type": "javascript",
            "implementation": implementation
        }

    def visit_IfExp(self, node:ast.IfExp):
        test = self.visit(node.test)["implementation"]
        body = self.visit(node.body)["implementation"]
        orelse = self.visit(node.orelse)["implementation"]
        implementation = f"({test})?({body}):({orelse})"
        return {
            "name": self.code,
            "type": "javascript",
            "implementation": implementation
        }

    def visit_Lambda(self, node:ast.Lambda):
        args = [i.arg for i in node.args.args]
        impl_args = ', '.join([i.arg for i in node.args.args])
        if self.locals:
            self.locals.append(self.locals[-1] | set(args))
        else:
            self.locals.append(set(args))
        impl_body = self.visit(node.body)["implementation"]
        self.locals.pop()
        return {
            "name": self.code,
            "type": "js_lambda",
            "n_args": len(args),
            "implementation": f"(({impl_args})=>({impl_body}))"
        }

def checkColumn(columnKey, tableKey, cdsDict):
    return False

def getOrMakeColumns(variableNames, context = None, cdsDict: dict = {}, paramDict: dict = {}, funcDict: dict = {},
                     memoizedColumns: dict = None, aliasDict: dict = None, forbiddenColumns: set = set()):
    if variableNames is None or len(variableNames) == 0:
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
    if memoizedColumns is None:
        memoizedColumns = {}
    if aliasDict is None:
        aliasDict = {}
    for i in range(max(len(variableNames), len(context))):
        variables_tuple = variableNames[i % nvars]
        i_context = context[i % n_context] 
        if i_context == "$IGNORE":
            variables.append(None)
            ctx_updated.append(i_context)
            continue
        if not isinstance(variables_tuple, tuple):
            variables_tuple = (variables_tuple,)
        columns = []
        for i_var in variables_tuple:
            if i_var is None:
                columns.append(None)
                continue
            if i_context in memoizedColumns and i_var in memoizedColumns[i_context]:
                columns.append(memoizedColumns[i_context][i_var])
                continue
            if (i_context, i_var) in forbiddenColumns:
                # Unresolvable cyclic dependency in table - stack trace should tell exactly what went wrong
                raise RuntimeError("Cyclic dependency detected")
            queryAST = ast.parse(i_var, mode="eval")
            evaluator = ColumnEvaluator(i_context, cdsDict, paramDict, funcDict, i_var, aliasDict)
            column = evaluator.visit(queryAST.body)
            i_context = evaluator.context
            if column["type"] == "javascript":
                # Make the column on the server if possible both on server and on client
                # Possibly only do this if lossy compression causes numerical instability?
                if evaluator.isSource:
                    column = evaluator.eval_fallback(queryAST.body)
                elif aliasDict is not None:
                    if i_context not in aliasDict:
                        aliasDict[i_context] = {}
                    columnName = column["name"]
                    func = "return "+column["implementation"]
                    variablesAlias = list(evaluator.aliasDependencies)
                    fieldsAlias = list(evaluator.aliasDependencies)
                    parameters = {i:paramDict[i]["value"] for i in evaluator.paramDependencies if "options" not in paramDict[i]}
                    variablesParam = [i for i in evaluator.paramDependencies if "options" in paramDict[i]]
                    nvars_local = len(variablesAlias)
                    for j in variablesParam:
                        if "subscribed_events" not in paramDict[j]:
                            paramDict[j]["subscribed_events"] = []
                        paramDict[j]["subscribed_events"].append(["value", CustomJS(args={"idx":nvars_local, "column_name":columnName, "table":cdsDict[i_context]["cdsFull"]}, code="""
                                            table.mapping[column_name].fields[idx] = this.value
                                            table.invalidate_column(column_name)
                                                    """)])
                        variablesAlias.append(paramDict[j]["value"])
                        fieldsAlias.append(j)
                        nvars_local = nvars_local+1
                    transform = CustomJSNAryFunction(parameters=parameters, fields=fieldsAlias, func=func)
                    for j in parameters:
                        if "subscribed_events" not in paramDict[j]:
                            paramDict[j]["subscribed_events"] = []
                        paramDict[j]["subscribed_events"].append(["value", CustomJS(args={"mapper":transform, "param":j}, code="""
                                            mapper.parameters[param] = this.value
                                            mapper.update_args()
                                                    """)])
                    aliasDict[i_context][columnName] = {"transform": transform, "fields": variablesAlias}
                    newColumn = {
                        "type": "expr",
                        "name": columnName,
                        "is_boolean": column.get("is_boolean", False)
                    }
                    evaluator.dependencies.update({(evaluator.context, i) for i in evaluator.aliasDependencies})
                    column = newColumn
            if evaluator.isSource:
                used_names.update({(i_context, column["name"])})
            else:
                direct_dependencies = list(evaluator.dependencies)
                dependency_columns = [i[1] for i in direct_dependencies]
                dependency_tables = [i[0] for i in direct_dependencies]
                _, _, memoizedColumns, sources_local = getOrMakeColumns(dependency_columns, dependency_tables, cdsDict, paramDict, funcDict, 
                                                                        memoizedColumns, aliasDict, forbiddenColumns | {(i_context, i_var)})
                used_names.update(sources_local)
            if i_context in memoizedColumns:
                memoizedColumns[i_context][i_var] = column
            else:
                memoizedColumns[i_context] = {i_var: column}
            columns.append(column)
        if len(columns) == 1:
            variables.append(columns[0])
        else:
            variables.append(tuple(columns))
        ctx_updated.append(i_context)
        
    return variables, ctx_updated, memoizedColumns, used_names
