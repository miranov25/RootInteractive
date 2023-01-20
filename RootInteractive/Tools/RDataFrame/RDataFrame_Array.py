import ast
import numpy as np

class RDataFrame_Visit:
    # This class walks the Python abstract syntax tree of the expressions to detect its dependencies
    def __init__(self, code, df, name):
        self.n_iter = None
        self.code = code
        self.df = df
        self.name = name
        self.dependencies = set()

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
        raise NotImplementedError(node)

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

    def visit_Name(self, node: ast.Name):
        # Replaced with a mock
        if self.df is not None:
            columnType = self.df.GetColumnType(node.id)
            self.dependencies.add(node.id)
            return {"implementation": node.id, "type":columnType}
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
        op = node.op
        if isinstance(op, ast.UAdd):
            operator_prefix = "+"
        elif isinstance(op, ast.USub):
            operator_prefix = "-"
        else:
            operator_prefix = "!"
        implementation = f"{operator_prefix}({self.visit(node.operand)['implementation']})"
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

    def visit_Slice(self, node:ast.Slice):
        lower = 0
        if node.lower is not None:
            if isinstance(node.lower, ast.Constant):
                lower = node.lower.value
            else:
                raise NotImplementedError(f"Slices are only implemented for constant boundaries, got {ast.dump(node.lower)}")
        upper = 0
        if node.upper is not None:
            if isinstance(node.upper, ast.Constant):
                upper = node.upper.value
            else:
                raise NotImplementedError(f"Slices are only implemented for constant boundaries, got {ast.dump(node.upper)}")
        infix="+"
        step = 1
        if node.step is not None:
            if isinstance(node.upper, ast.Constant):
                step = node.step.value
            else:
                raise NotImplementedError(f"Slices are only implemented for constant boundaries, got {ast.dump(node.step)}")
        if step == 0:
            raise ValueError("Slice step cannot be zero")
        n_iter = (upper-lower)//step
        if self.n_iter is None:
            self.n_iter = n_iter
        return {
            "implementation":f"{lower}{infix}i*{step}",
            "type":"slice"
        }

    def visit_IfExp(self, node:ast.IfExp):
        test = self.visit(node.test)["implementation"]
        body = self.visit(node.body)["implementation"]
        orelse = self.visit(node.orelse)["implementation"]
        implementation = f"({test})?({body}):({orelse})"
        return {
            "implementation": implementation
        }

    def visit_Subscript(self, node:ast.Subscript):
        value = self.visit(node.value)
        sliceValue = self.visit(node.slice)
        dtype = value["type"]
        return {
            "implementation":f"{value['implementation']}[{sliceValue['implementation']}]",
            "type":dtype
        }

    def visit_Expression(self, node:ast.Expression):
        body = self.visit(node.body)
        return {
            "implementation":f"""
auto {self.name}(){{
    RVec<{body["type"]}> result({self.n_iter});
    for(size_t i=0; i<{self.n_iter}; i++){{
        result[i] = {body["implementation"]};
    }}
    return result;
}}
            """,
        }

def makeDefine(name, code, df):
    t = ast.parse(code, "<string>", "eval")
    evaluator = RDataFrame_Visit(code, df, name)
    parsed = evaluator.visit(t)
    if df is not None:
        df.Define(name, parsed["implementation"], list(evaluator.dependencies))
    print(parsed["implementation"])

# makeDefine("C","A[1:10]-B[:20:2]", None)