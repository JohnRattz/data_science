# sys.path.insert(0,'../../' + utilities_dir)
import pandas as pd


def pandas_dt_to_str(date):
    """
    Prints pandas `Datetime` objects as strings.
    """
    return date.strftime("%Y-%m-%d")


def numpy_dt64_to_str(date):
    """
    Prints NumPy `datetime64` objects as strings.
    """
    pd_datetime = pd.to_datetime(str(date))
    return pandas_dt_to_str(pd_datetime)