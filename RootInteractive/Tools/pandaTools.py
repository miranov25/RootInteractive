import re
import logging
import pandas as pd
from types import SimpleNamespace


def initMetadata(ddf):
    """
    create slot for the metadata   see comment 3 of - https://stackoverflow.com/questions/14688306/adding-meta-information-metadata-to-pandas-dataframe/14688398#14688398
    :param ddf:    data frame
    :return:      None
    """
    if hasattr(ddf, "meta"):
        if hasattr(ddf.meta, "metaData"):
            return
    ddf.meta = SimpleNamespace()
    ddf.meta.metaData = {}
    #
    if "meta" not in pd.DataFrame._metadata:
        pd.DataFrame._metadata += ["meta"]


def pandaGetOrMakeColumn(df, variableName):
    """
    pandaGetOrMakeColumn      - Get column if exist or create and append   - in case of function
    :param df:                - input data frame
    :param variableName:      - variableToCheck/resp.append
    :return:
        * df - data frame
        * nev variable name replacing special not allowed characters
        * original name of variable stored in metaData to the dataFrame
    """
    initMetadata(df)
    varName = variableName
    varName = re.sub(r"""\+""", "_Plus_", varName)
    varName = re.sub(r"""\-""", "_Minus_", varName)
    varName = re.sub(r"""\*""", "_Mult_", varName)
    varName = re.sub(r"""/""", "_Divide_", varName)
    varName = re.sub(r"""\.""", "_Dot_", varName)
    varName = re.sub(r"""@""", "_At_", varName)
    varName = re.sub(r"""(\(|\))""", "_", varName)
    if varName in df.columns:
        return df, varName
    expression = variableName
    df.meta.metaData[varName + ".OrigName"] = variableName

    try:
        df[varName] = df.eval(expression)
    except:
        logging.error("Variable can not be evaluated" + varName)
        logging.error("Var.list=", list(df.columns.values))
    return df, varName


def getStatPanda(df, variable, formula):
    """
        getStatPanda      - interpret formula expression as a float or as a panda expression
        :param df:
        :param variable:
        :param formula:
        :return:   float value
        TODO - NOT yet finished
        """
    value = 0;
    if "@" in formula:
        formula = formula.replace("@", f"{variable}.")
        value = 0
        try:
            value = df.eval(formula)
        except:
            logging.error("getStat", f"Invalid formula {formula}")
        return value
    try:
        value = float(formula)
    except:
        logging.error("getStat", f"Invalid formula {formula}")
    return value
