import ast
import ROOT
import logging
import cppyy.ll

def getGlobalFunction(name="cos", verbose=1):
    info={"fun":0, "returnType":"", "nArgs":0}
    fun = ROOT.gROOT.GetListOfGlobalFunctions().FindObject(name)
    if fun:
        info["fun"]=name
        info["returnType"]=fun.GetReturnTypeName()
        info["nArgs"]=fun.GetNargs()
    if info["fun"]==0:
        logging.error(f"GetGlobalFunction {name} {info} does not exist")
        fun2=name.replace("::",".")
        docString = eval(f"ROOT.{fun2}.func_doc")
        returnType = docString.split(" ", 1)[0]
        #
        if returnType == "":
            logging.error(f"Non supported function {name}")
            raise NotImplementedError(f"Non supported function {name}")
        nArgs=docString.split(" ", 1)[1].count(",") + 1
        funImp=docString.split(" ", 1)[1].split("(", 1)[0]
        info["fun"]=funImp
        info["returnType"]=returnType
        info["nArgs"]=nArgs

    logging.info(f"GetGlobalFunction {name} {info}")
    return info

def getClass(name="TParticle", verbose=0):
    info={"rClass":0}
    rClass = ROOT.gROOT.GetListOfClasses().FindObject(name)
    if rClass:
        info["rClass"]=rClass
        info["publicMethods"]=rClass.GetListOfAllPublicMethods()
        info["publicData"]=rClass.GetListOfAllPublicDataMembers()
    if verbose:
        logging.info(f"GetGlobalFunction {name} {info}")
    return info

def getClassMethod(className, methodName, arguments=[]):
    """
    get return type and check consistency
    TODO:  this is a hack - we should get return method description
    return class Mmthod information
    :param className:
    :param methodName:
    :return:  type of the method if exist
    in some classes more than on function exist - the function with proper arguments to be used
    className = "AliExternalTrackParam" ; methodName="GetX"
    """
    import re
    className2=className.replace("::",".")
    className2 =className2.replace("<", '("')
    className2 =className2.replace(">", '")')

    docString= eval(f"ROOT.{className2}.{methodName}.func_doc")
    returnType=docString.split(" ", 1)[0]
    if returnType=="":
        logging.error(f"Non supported function {className2}.{methodName}")
        raise NotImplementedError(f"Non supported function {className2}.{methodName}")
    return (returnType,docString)


def getClassProperty(className, propertyName):
    """
    return type of the property (or empty if not exist) and the data offset - using TClass interface
    Error handling - https://cppyy.readthedocs.io/en/latest/lowlevel.html
    :param className:  ROOT classname
    :param propertyName:
    :return:
    Example:
        className="TParticle"
        propertyName="fPdgCode"
        In [11]: getClassProperty("TParticle","fPdgCode")
        className="o2::tpc::TrackTPC"
        methodName="mAlpha"
        getClassProperty("o2::tpc::TrackTPC","mAlpha")
Out[11]: ('int', 40)
    """
    tclassNull   = cppyy.bind_object(cppyy.nullptr, 'TClass')
    tobjectNull  = cppyy.bind_object(cppyy.nullptr, 'TObject')
    realDataNull = cppyy.bind_object(cppyy.nullptr, 'TRealData')  # https://cppyy.readthedocs.io/en/latest/lowlevel.html

    clT=  ROOT.TClass.GetClass(className)
    if clT==tclassNull:
        clT = ROOT.gROOT.FindSTLClass(className, True)

    if clT==tclassNull:
        clT = None
        raise NotImplementedError(f"Non supported {className}")

    if clT.GetListOfAllPublicDataMembers(True).FindObject(propertyName)==tobjectNull:
        clT = None
        raise NotImplementedError(f"Non supported property {className}.{propertyName}")

    realData=clT.GetRealData(propertyName)
    if (realData==realDataNull):
        clT=None
        raise NotImplementedError(f"Non supported property {className}.{propertyName}")
    if (realData.GetDataMember()==realDataNull):
        clT=None
        raise NotImplementedError(f"Non supported property {className}.{propertyName}")
    type=realData.GetDataMember().GetTypeName()
    offset=realData.GetDataMember().GetOffset()
    if type=='':
        clT=None
        logging.error(f"Non supported property {className}.{propertyName}")
        raise NotImplementedError(f"Non supported property {className}.{propertyName}")
    clT=None
    realData=None
    return type,offset


def scalar_type(name):
    dtypes = {
        "float": ('f', 32),
        "double": ('f', 64),
        "long double": ('f', 64),
        "size_t": ('u', 64),
        "long long": ('i', 64),
        "unsigned int": ('u', 32),
        "int": ('i', 32),
        "char": ('i', 8),
        "unsigned char": ('u', 8),
        "bool":('u', 8)
    }
    return dtypes.get(name, name)

def scalar_type_str(dtype):
    dtypes = {
        ('f', 32): "float",
        ('f', 64): "double",
        ('u', 64): "size_t",
        ('i', 64): "long long",
        ('u', 32): "unsigned int",
        ('i', 32): "int",
        ('u', 8): "unsigned char",
        ('i', 8): "char"
    }
    return dtypes[dtype]

def add_dtypes(left, right):
    # This is a hack - use minimum for kind and maximum for depth
    return (min(left[0],right[0]), max(left[1], right[1]))

def truediv_dtype(left, right):
    x = add_dtypes(left, right)
    if x[0] == 'f':
        return x
    elif x[1] <= 16:
        return ('f', 32)
    return ('f', 64)

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
        elif isinstance(node, ast.Index):
            return self.visit_Index(node)
        elif isinstance(node, ast.Slice):
            return self.visit_Slice(node)
        elif isinstance(node, ast.ExtSlice):
            return self.visit_ExtSlice(node)
        elif isinstance(node, ast.Tuple):
            return self.visit_Tuple(node)
        elif isinstance(node, ast.Constant):
            return self.visit_Constant(node)
        elif isinstance(node, ast.Attribute):
            return self.visit_Attribute(node)
        raise NotImplementedError(node)

    def visit_Call(self, node: ast.Call):
        args = [self.visit(iArg) for iArg in node.args]
        left = self.visit_func(node.func, args)
        implementation = left['implementation'] + '('
        implementation += ", ".join([i['implementation'] for i in args])
        implementation += ')'
        return {
            "implementation": implementation,
            "type": scalar_type(left["returnType"])
        }

    def visit_Num(self, node: ast.Num):
        # Kept for compatibility with old Python
        node_type = ('f', 64)
        if isinstance(node.value, int):
            node_type = ('i', 64)
        if isinstance(node.value, str):
            node_type = ("o", 64)
        return {
            "implementation": str(node.n),
            "value": node.n,
            "type": node_type
        }

    def visit_Constant(self, node: ast.Constant):
        node_type = ('f', 64)
        if isinstance(node.value, int):
            node_type = ('i', 64)
        if isinstance(node.value, str):
            node_type = ("o", 64)
        return {
            "implementation": str(node.value),
            "value": node.value,
            "type": node_type
        }

    def visit_Name(self, node: ast.Name):
        # Replaced with a mock
        if self.df is not None:
            if self.df.HasColumn(node.id):
                columnType = self.df.GetColumnType(node.id)
                self.dependencies[node.id] = {"type":columnType}
                return {"implementation": node.id, "type":columnType}
            else:
                return {"implementation": node.id, "type":None}               
        self.dependencies[node.id] = {"type":"RVec<double>"}
        return {"implementation": node.id, "type":"RVec<double>"}

    def visit_BinOp(self, node:ast.BinOp):
        op = node.op
        left = self.visit(node.left)
        right = self.visit(node.right)
        merged_dtype = add_dtypes(left["type"], right["type"])
        left_cast = ""
        right_cast = ""
        if isinstance(op, ast.Add):
            operator_infix = " + "
        elif isinstance(op, ast.Sub):
            operator_infix = " - "
        elif isinstance(op, ast.Mult):
            operator_infix = " * "
        elif isinstance(op, ast.Div):
            operator_infix = " / "
            if left["type"] != merged_dtype:
                left_cast = f"({scalar_type_str(merged_dtype)})"
            if right["type"] != merged_dtype:
                right_cast = f"({scalar_type_str(merged_dtype)})"
        elif isinstance(op, ast.FloorDiv):
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
            raise NotImplementedError(f"Binary operator {ast.dump(op)} not implemented")
        implementation = f"{left_cast}({left['implementation']}){operator_infix}{right_cast}({right['implementation']})"
        if isinstance(op, ast.FloorDiv) and merged_dtype[0] == 'f':
            implementation = f"floor({implementation})"
        return {
            "name": self.code,
            "implementation": implementation,
            "type": merged_dtype
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
                    "implementation":f"{new_value}",
                    "type": operand["type"]
                }         
        else:
            operator_prefix = "!"
        implementation = f"{operator_prefix}({operand['implementation']})"
        return {
            "name": self.code,
            "implementation": implementation,
            "type": operand["type"]
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
                raise NotImplementedError(f"Comparison operator {ast.dump(op)} not implemented")
            js_comparisons.append(f"(({lhs}){op_infix}({rhs}))")
        implementation = " && ".join(js_comparisons)
        return {
            "name": self.code,
            "type": ('i', 8),
            "implementation": implementation
        }

    def visit_BoolOp(self, node:ast.BoolOp):
        values = [f"({self.visit(i)})" for i in node.values]
        if isinstance(node.op, ast.And):
            op_infix = " && "
        elif isinstance(node.op, ast.Or):
            op_infix = " || "
        implementation = op_infix.join([i["implementation"] for i in values])
        return {
            "name": self.code,
            "type": values[-1]["type"],
            "implementation": implementation
        }

    def visit_Index(self, node:ast.Index):
        return self.visit(node.value)

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
        n_iter_arr = sliceValue.get("n_iter", None)
        if not isinstance(n_iter_arr, list):
            n_iter_arr = [n_iter_arr]
        axis_idx = 0
        for idx, n_iter in enumerate(n_iter_arr):
            if len(self.n_iter) <= axis_idx:
                self.n_iter.append(n_iter)
            # Detect if length needs to be used here
            if n_iter is None:
                continue
            axis_idx += 1
            if n_iter < 0:
                self.n_iter[idx] = f"{value['implementation']}.size() - {-n_iter}"
            else:
                self.n_iter[idx] = n_iter
        dtype_str = unpackScalarType(value["type"], len(n_iter_arr))
        dtype = scalar_type(dtype_str)
        logging.info(f"\t Data type: {dtype_str}, {dtype}")
        return {
            "implementation":f"{value['implementation']}[{sliceValue['implementation']}]",
            "type":dtype
        }

    def visit_Expression(self, node:ast.Expression):
        body = self.visit(node.body)
        loop, array_type = self.makeOuterLoop(0, body["implementation"], scalar_type_str(body["type"]))
        dependencies_list = [(key, value) for key, value in self.dependencies.items()]
        input_args = ', '.join([f"{value['type']} &{key}" for key, value in dependencies_list])
        signature = f"{array_type} {self.name}({input_args})"
        return {
            "implementation":f"""{signature}{{
    {loop}
    return result;
}} """,
"type": array_type,
"name": self.name,
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
            return {"type":"function", "implementation":node.id, "returnType":func["returnType"]}
        return {"type":"function", "implementation":node.id, "returnType":"double"}
    
    def visit_Attribute(self, node:ast.Attribute):
        left = self.visit(node.value)
        className = left["type"]
        (fieldType, offset) = getClassProperty(className, node.attr)
        return {"type":scalar_type(fieldType), "implementation": f"{left['implementation']}.{node.attr}"}

    def visit_func_Attribute(self, node:ast.Attribute, args):
        left = self.visit(node.value)
        className = left["type"]
        if className is None:
            func = getGlobalFunction(f"{left['implementation']}::{node.attr}")
            return {"type":"function", "implementation":f"{left['implementation']}::{node.attr}", "returnType":func["returnType"]}
        (returnType, docstring) = getClassMethod(className, node.attr, args)
        return {"type":"function", "implementation":f"{left['implementation']}.{node.attr}", "returnType":returnType}

    def visit_Tuple(self, node:ast.Tuple):
        # So far, the only tuple supported is a slice tuple
        x = [self.visit_Slice(iSlice, i) for i, iSlice in enumerate(node.elts)]
        return {"type":"int*", "implementation":']['.join([i["implementation"] for i in x]), "n_iter": [i["n_iter"] for i in x], "high_water": [i["high_water"] for i in x]}

    def visit_ExtSlice(self, node:ast.ExtSlice):
        # DEPRECATED: will be removed when we stop supporting python 3.8
        x = []
        iDim = 0
        for iSlice in node.dims:
            if isinstance(iSlice, ast.Slice):
                x.append(self.visit_Slice(iSlice, iDim))
                iDim += 1
            else:
                elt = self.visit(iSlice)
                elt["high_water"] = elt["value"]
                elt["n_iter"] = 0
                x.append(elt)
        return {"type":"int*", "implementation":']['.join([i["implementation"] for i in x]), "n_iter": [i["n_iter"] for i in x], "high_water": [i["high_water"] for i in x]}       

def unpackScalarType(vecType:str, level:int=0):
    if level <= 0:
        return vecType
    vecTypeNew = vecType.split('<',1)[1].rsplit('>',1)[-2]
    return unpackScalarType(vecTypeNew, level-1)


def makeDefineRNode(columnName, funName, parsed,  rdf, verbose=1, flag=0x1):
    """
    makeDefinerNode         this is internal function to create columns  column Name for fucnion funName

    :param columnName:  output column name to append
    :param funName:     function to use
    :param parsed:      implementation + dependencies
    :param rdf:         input RDF
    :param verbose:     verbosity
    :param flag            - 0x1-makeDefine / do nothing if column exist, 0x2- force bit to redefine if exist
    :return:            data frame with new column - columns name
    """
    if verbose & 0x1:
        logging.info(f"{columnName}\t{funName}\t{parsed}")
    # 0.) Define function if does not exist yet

    try:
        ROOT.gInterpreter.Declare( parsed["implementation"])
    except:
        logging.error(f'makeDefineRNode compilation of {funName} failed Implementation in {parsed["implemntation"]}')
        return rdf
        pass
    # 1.) set  rdf to ROOT space - the RDataFrame_Array should be owner
    #
    try:
        #funString=f"{funName}"+"("+f'{parsed["dependencies"]}'[1:-1]+")".replace("'","")
        funString=f"{funName}" + "(" + f'{parsed["dependencies"]}'[2:-2].replace("'", "") + ")"
        dfOut=rdf.Define(columnName,funString)
    except:
        logging.error(f'makeRNode compilation of {funName} failed Implementation in {parsed["implementation"]}\n {funString}')
        return rdf
        pass
    return  dfOut

def makeDefine(name, code, df, cppLibDictionary=None, verbose=3, flag=0x1):
    """

    :param name:           - name of the new column
    :param code:           - source code string
    :param df:             - data frame to add new imlementation and to define input varaible list
    :param verbose:        - verbosity bitmask
    :param flag            - 0x1-makeDefine / do nothing if column exist, 0x2- force bit to redefine if exist 0x4 - test only
    :return:
    """
    if (df.HasColumn(name) & ((flag&0x2)==0)):
        logging.error(f'Column {name} already exist, please use redefine bit (0x2) in flag variable')
        return df

    if verbose & 0x4:
        logging.info(f"makeDefine - Begin  {name} {code}")
    t = ast.parse(code, "<string>", "eval")
    if verbose & 0x8:
        logging.info("makeDefine - ast parse", name, code)
    evaluator = RDataFrame_Visit(code, df, name)
    if verbose & 0x8:
        logging.info("makeDefine - evaluator",evaluator)
    parsed = evaluator.visit(t)
    if verbose>0:
        logging.info(f"{name} \n{code}")

    if verbose & 0x1:
        logging.info(f'Implementation:\n {parsed["implementation"]}')

    if verbose & 0x2:
        logging.info(f'Dependencies\n  {parsed["dependencies"]}')
    if cppLibDictionary!=None:
        parsed["code"]=code
        cppLibDictionary[name]=parsed

    if df is not None and (flag&0x4)==0:
        if df is not None:
            rdf = makeDefineRNode(name, name, parsed, df, verbose)
            return rdf
    return parsed

def makeLibrary(cppLib,outFile, includes=""):
    """

    :param cppLib:   dictioanry optionaly filled in the makeDefine as c[[LibDictionary
    :param outFile:  output text file with cpp code
    :return:         output file - ROOT C++ macro which can be loaded or compiled as share library
    outFile="rdf.C"
    """
    with open(outFile, 'w') as f:
        # insert implementnation
        f.write(includes)
        for key in cppLib:
            f.write(cppLib[key]["implementation"])
            f.write("\n")
        #inserte rdf
        f.write("""
        ROOT::RDF::RNode getDFAll(ROOT::RDF::RNode &df){
        """)
        for key in cppLib:
            v=cppLib[key]
            dependency = "{" + f'{v["dependencies"]}'[1:-1] + "}".replace("'", "\"")
            dependency = dependency.replace("'", "\"")
            dfLine=\
            f'''
            df=df.Define("{v['name']}",{v['name']},{dependency});\n 
            '''
            f.write(dfLine)
        f.write("return df;\n")
        f.write("}\n")