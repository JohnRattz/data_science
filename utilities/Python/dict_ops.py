import itertools as it

def get_dict_combinations(dictionary):
    """
    Returns a list of dictionaries comprising all possible combinations of values in `dictionary`,
    with the possible values that a key can map to in the output dictionaries being
    specified in lists - one list for each key in the input dictionary.

    Parameters
    ----------
    dictionary: dict
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
    if len(dictionary) == 0:
        return dict()
    else:
        keys, values = zip(*dictionary.items())
        return [dict(zip(keys, v)) for v in it.product(*values)]