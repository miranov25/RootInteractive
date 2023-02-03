import ast
import ROOT

def getGlobalFunction(name="cos", verbose=0):
    info={"fun":0, "returnType":"", "nArgs":0}
    fun = ROOT.gROOT.GetListOfGlobalFunctions().FindObject(name)
    if fun:
        info["fun"]=fun
        info["returnType"]=fun.GetReturnTypeName()
        info["nArgs"]=fun.GetNargs()
    # fun.GetReturnTypeName()   -> 'long double'
    # fun.GetNargs()            ->  1
    if verbose:
        print("GetGlobalFunction",name, info)
    return info

def getClass(name="TParticle", verbose=0):
    info={"rClass":0}
    rClass = ROOT.gROOT.GetListOfClasses().FindObject(name)
    if rClass:
        info["rClass"]=rClass
        info["publicMethods"]=rClass.GetListOfAllPublicMethods()
        info["publicData"]=rClass.GetListOfAllPublicDataMembers()
    if verbose:
        print("GetGlobalFunction",name, info)
    return info

def getClassMethod(className, methodName, arguments=[]):
    """
    get return type and check consistency
    TODO:  this is a hack - we should get return method description
    return class Mmthod information
    :param className:
    :param methodName:
    :return:  type of the method if exist
    className = "AliExternalTrackParam" ; methodName="GetX"
    """
    import re
    try:
        docString= eval(f"ROOT.{className}.{methodName}.func_doc")
        returnType = re.sub(f"{className}.*","",docString)
        return (returnType,docString)
    except:
        pass
    return ("","")


class RDataFrame_Visit:
    # This class walks the Python abstract syntax tree of the expressions to detect its dependencies
    def __init__(self, code, df, name):
        self.n_iter = []
        self.code = code
        self.df = df
        self.name = name
        self.dependencies = {}

    def visit(self, node):
        if isinstance(node, ast.Call):
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
        elif isinstance(node, ast.Expression):
            return self.visit_Expression(node)
        elif isinstance(node, ast.Subscript):
            return self.visit_Subscript(node)
        elif isinstance(node, ast.Slice):
            return self.visit_Slice(node)
        elif isinstance(node, ast.Tuple):
            return self.visit_Tuple(node)
        elif isinstance(node, ast.Constant):
            return self.visit_Constant(node)
        raise NotImplementedError(node)

    def visit_Call(self, node: ast.Call):
        args = [self.visit(iArg) for iArg in node.args]
        left = self.visit_func(node.func, args)
        implementation = left['implementation'] + '('
        implementation += ", ".join([i['implementation'] for i in args])
        implementation += ')'
        return {
            "implementation": implementation,
            "type": left["type"]
        }

    def visit_Num(self, node: ast.Num):
        # Kept for compatibility with old Python
        return {
            "implementation": str(node.n),
            "value": node.n,
        }

    def visit_Constant(self, node: ast.Num):
        return {
            "implementation": str(node.value),
            "value": node.value,
        }

    def visit_Name(self, node: ast.Name):
        # Replaced with a mock
        if self.df is not None:
            columnType = self.df.GetColumnType(node.id)
            self.dependencies[node.id] = {"type":columnType}
            return {"implementation": node.id, "type":columnType}
        self.dependencies[node.id] = {"type":"RVec<double>"}
        return {"implementation": node.id, "type":"RVec<double>"}

    def visit_BinOp(self, node):
        op = node.op
        if isinstance(op, ast.Add):
            operator_infix = " + "
        elif isinstance(op, ast.Sub):
            operator_infix = " - "
        elif isinstance(op, ast.Mult):
            operator_infix = " * "
        elif isinstance(op, ast.Div):
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
        left = self.visit(node.left)
        right = self.visit(node.right)
        implementation = f"({left['implementation']}){operator_infix}({right['implementation']})"
        return {
            "name": self.code,
            "implementation": implementation,
            "type": "double"
        }

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        op = node.op
        if isinstance(op, ast.UAdd):
            operator_prefix = "+"
            # Constant folding hack
            if "value" in operand:
                return operand
        elif isinstance(op, ast.USub):
            operator_prefix = "-"
            if "value" in operand:
                new_value = -operand["value"]
                return {
                    "value": new_value,
                    "implementation":f"{new_value}"
                }         
        else:
            operator_prefix = "!"
        implementation = f"{operator_prefix}({operand['implementation']})"
        return {
            "name": self.code,
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
                op_infix = " == "
            elif isinstance(op, ast.NotEq):
                op_infix = " != "
            elif isinstance(op, ast.Lt):
                op_infix = " < "
            elif isinstance(op, ast.LtE):
                op_infix = " <= "
            elif isinstance(op, ast.Gt):
                op_infix = " > "
            elif isinstance(op, ast.GtE):
                op_infix = " >= "
            else:
                raise NotImplementedError(f"Binary operator {ast.dump(op)} not implemented")
            js_comparisons.append(f"(({lhs}){op_infix}({rhs}))")
        implementation = " && ".join(js_comparisons)
        return {
            "name": self.code,
            "type": "char",
            "implementation": implementation
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
            "type": "char",
            "implementation": implementation
        }

    def visit_Slice(self, node:ast.Slice, idx=0):
        lower_value = 0
        if node.lower is not None:
            lower = self.visit(node.lower)
            lower_value = lower.get("value", 0)
        upper_value = 0
        if node.upper is not None:
            upper = self.visit(node.upper)
            upper_value = upper.get("value", 0)
        infix="+"
        step = 1
        if node.step is not None:
            if isinstance(node.upper, ast.Constant):
                step = node.step.value
            else:
                raise NotImplementedError(f"Slices are only implemented for constant boundaries, got {ast.dump(node.step)}")
        if step == 0:
            raise ValueError("Slice step cannot be zero")
        n_iter = (upper_value-lower_value)//step
        dim_idx = f"_{idx}" if idx>0 else ""
        return {
            "implementation":f"{lower_value}{infix}i{dim_idx}*{step}",
            "type":"slice",
            "n_iter":n_iter,
            "high_water":max(lower_value,upper_value)
        }

    def visit_IfExp(self, node:ast.IfExp):
        test = self.visit(node.test)["implementation"]
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        body_implementation = body["implementation"]
        orelse_implementation = orelse["implementation"]
        if body["type"] != orelse["type"]:
            raise TypeError(f"Incompatible types: {body['type']}, {orelse['type']}")
        implementation = f"({test})?({body_implementation}):({orelse_implementation})"
        return {
            "implementation": implementation,
            "type":body['type']
        }

    def visit_Subscript(self, node:ast.Subscript):
        value = self.visit(node.value)
        sliceValue = self.visit(node.slice)
        n_iter_arr = sliceValue["n_iter"]
        if not isinstance(n_iter_arr, list):
            n_iter_arr = [n_iter_arr]
        for idx, n_iter in enumerate(n_iter_arr):
            if len(self.n_iter) <= idx:
                self.n_iter.append(n_iter)
            # Detect if length needs to be used here
            if n_iter <= 0:
                self.n_iter[idx] = f"{value['implementation']}.size() - {-n_iter}"
            else:
                self.n_iter[idx] = n_iter
        dtype = unpackScalarType(value["type"])
        return {
            "implementation":f"{value['implementation']}[{sliceValue['implementation']}]",
            "type":dtype
        }

    def visit_Expression(self, node:ast.Expression):
        body = self.visit(node.body)
        loop, array_type = self.makeOuterLoop(0, body["implementation"], body["type"])
        dependencies_list = [(key, value) for key, value in self.dependencies.items()]
        input_args = ', '.join([f"{value['type']} &{key}" for key, value in dependencies_list])
        signature = f"{array_type} {self.name}({input_args})"
        return {
            "implementation":f"""{signature}){{
    {loop}
    return result;
}} """,
"type": array_type,
"dependencies": [i[0] for i in dependencies_list]
        }

    def makeOuterLoop(self, depth:int, innerLoop:str, dtype:str):
        depth_f = f"_{depth}" if depth>0 else ""
        depth_f_lower = f"_{depth-1}" if depth>1 else ""
        if depth>=len(self.n_iter):
            depth_f = f"_{depth-1}" if depth>1 else ""
            return f"result{depth_f}[i{depth_f}] = {innerLoop};", dtype
        next_level, array_type = self.makeOuterLoop(depth+1, innerLoop, dtype)
        array_type = f"ROOT::VecOps::RVec<{array_type}>"
        expr_f = f"result{depth_f_lower}[i{depth_f_lower}] = result{depth_f};" if depth>0 else ""
        return f"""{array_type} result{depth_f}({self.n_iter[depth]});
    for(size_t i{depth_f}=0; i{depth_f}<{self.n_iter[depth]}; i{depth_f}++){{
        {next_level}
    }}
    {expr_f}""", array_type

    def visit_func(self, node, args):
        # Detect global function from class method
        if isinstance(node, ast.Name):
            return self.visit_func_Name(node, args)
        if isinstance(node, ast.Attribute):
            return self.visit_func_Attribute(node, args)        
        raise NotImplementedError(f"{ast.dump(node)} is not supported as a function")

    def visit_func_Name(self, node:ast.Name, args):
        if self.df:
            func = getGlobalFunction(node.id)
            return {"type":"function", "implementation":node.id, "type":func["returnType"]}
        return {"type":"function", "implementation":node.id, "type":"double"}

    def visit_func_Attribute(self, node:ast.Attribute, args):
        left = self.visit(node.value)
        return {"type":"function", "implementation":f"{left['implementation']}.{node.attr}"}

    def visit_Tuple(self, node:ast.Tuple):
        # So far, the only tuple supported is a slice tuple
        x = [self.visit_Slice(iSlice, i) for i, iSlice in enumerate(node.elts)]
        return {"type":"int*", "implementation":']['.join([i["implementation"] for i in x]), "n_iter": [i["n_iter"] for i in x], "high_water": [i["high_water"] for i in x]}

def unpackScalarType(vecType:str, level:int=0):
    if level <= 0:
        return vecType
    vecTypeNew = vecType.split('<',1)[1][:-1]
    return unpackScalarType(vecTypeNew, level-1)

def makeDefine(name, code, df, verbose=3, isTest=False):
    t = ast.parse(code, "<string>", "eval")
    evaluator = RDataFrame_Visit(code, df, name)
    parsed = evaluator.visit(t)
    if verbose>0:
        print("====================================\n")
        print(f"{name}\n", f"{code}")
        print("====================================\n")

    if verbose & 0x1:
        print("Implementation:\n", parsed["implementation"])

    if verbose & 0x2:
        print("Dependencies\n", parsed["dependencies"])
    if df is not None and not isTest:
        # TODO - if function not yet exist declare it
        ROOT.gInterpreter.Declare( parsed["implementation"])
        defineLine=f"""
            auto rdfOut=rdf.Define(name,{name},{parsed["dependencies"]})
        """
        # print(defineLine)
        # ROOT.gInterpreter.ProcessLine(defineLine)
        # rdfOut=df.Define(name,"ROOT.{name}",parsed["dependencies"])
        # return ROOT.rdfOut
        return parsed
    else:
        return parsed


# makeDefine("C","cos(A[1:10])-B[:20:2]", None,3, True)
# makeDefine("C","cos(A[1:10])-B[:20:2,1:3]", None,3, True)
# makeDefine("B","A[1:]-A[:-1]", None,3, True)