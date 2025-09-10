from pathlib import Path

def data_path() -> Path:
    """
    Returns the location of the data cube, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the data directory
    """
    cwd = Path("..")
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_folder = folder / "data"
        if data_folder.exists() and data_folder.is_dir():
            print("\nData (main) directory found in ", data_folder)
            return data_folder
        else:
            raise Exception("\nData not found")
        
def plot_data_path() -> Path:
    """
    Returns the location of the plot directory, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the plot directory
    """
    cwd = Path("..")
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_folder = folder / "plots"
        if data_folder.exists() and data_folder.is_dir():
            print("\nPlot directory found in ", data_folder)
            return data_folder
        else:
            raise Exception("\nPlots directory not found")
            raise Exception("Fits files directory not found")