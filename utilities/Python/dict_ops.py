import itertools as it

def get_combinations(param_grid):
    """
    Returns a list of dictionaries comprising all possible combinations of values,
    with the possible values that a key can map to in the output dictionaries being
    specified in lists - one list for each key in the input dictionary.

    Parameters
    ----------
    param_grid: dict
        A `dict`, with values being lists.

    Example
    -------
    `>>>` my_dict={'param1':['p1_val1','p1_val2'],'param2':['p2_val1','p2_val2'],'param3':['p3_val1', 'p3_val2']} \n
    `>>>` combs = get_combinations(my_dict) \n
    `>>>` for comb in combs: \n
    `...` \t print(comb) \n
    `...` \n
    {'param1': 'p1_val1', 'param2': 'p2_val1', 'param3': 'p3_val1'} \n
    {'param1': 'p1_val1', 'param2': 'p2_val1', 'param3': 'p3_val2'} \n
    {'param1': 'p1_val1', 'param2': 'p2_val2', 'param3': 'p3_val1'} \n
    {'param1': 'p1_val1', 'param2': 'p2_val2', 'param3': 'p3_val2'} \n
    {'param1': 'p1_val2', 'param2': 'p2_val1', 'param3': 'p3_val1'} \n
    {'param1': 'p1_val2', 'param2': 'p2_val1', 'param3': 'p3_val2'} \n
    {'param1': 'p1_val2', 'param2': 'p2_val2', 'param3': 'p3_val1'} \n
    {'param1': 'p1_val2', 'param2': 'p2_val2', 'param3': 'p3_val2'} \n
    """
    if len(param_grid) == 0:
        return dict()
    else:
        keys, values = zip(*param_grid.items())
        return [dict(zip(keys, v)) for v in it.product(*values)]