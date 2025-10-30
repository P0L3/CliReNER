# Collection of helper functions for sentence tokenized paper processing
from os import listdir, path


def get_paper_id(paper_name):
    """Extract the paper ID from the paper name."""
    return int(paper_name.split("_")[0])

def load_paper_names(location, file_type=".txt"):
    """
    Loads a list of paper names from a specified directory, filtering based on file type 
    and excluding files with "DUP" in their name.
    
    Args:
        location (str): The path to the directory containing the files.
        file_type (str, optional): The file extension to filter by. Defaults to ".txt".
    Returns:
        list: A list of filenames (strings) in the specified directory that match the 
              given file type and do not contain "DUP" in their names.
    """
    return [p for p in listdir(location) if "DUP" not in p and file_type in p]

def load_paper_sentences(paper_name, DIR="."):
    """
    Loads and splits a paper's content into sentences.

    Parameters:
    -----------
    paper_name : str
        The name of the file containing the paper's text.
    DIR : str, optional
        The directory where the paper file is located (default is the current directory).
        
    Returns:
    -----------
    list of str
    
    A list of sentences from the paper, split by double newlines (\"\\n\\n\").
    """
    PAPER_DIR = DIR + "/" + paper_name
    try:
        with open(PAPER_DIR, "r") as f:
            paper_string = f.read()
    except:
        windows_path = PAPER_DIR.replace("/", "\\")
        
        # Ensure the path is absolute
        absolute_path = path.abspath(windows_path)

        # Prepend \\? to handle long paths
        unc_path = "\\?\\" + absolute_path
        print(unc_path)
        # Try opening the file using the UNC path format
        with open(unc_path, "r") as f:
            paper_string = f.read()



    sentences = paper_string.split("\n\n")
    
    return sentences