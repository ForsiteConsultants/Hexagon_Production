def get_grids(grid_list, suffix:str='') -> list[str]:
    """
    Create of list of 4km grid IDs to process, pulled from .txt file.
    """
    with open(grid_list, 'r') as f:
        grids = [line.strip().removesuffix(suffix) for line in f]
    return grids