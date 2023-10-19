def read_model(model_path, file_type=None):
    """
    Read a model from a file

    :param model_path: Path to the model file
    :type model_path: str
    :param file_type: Type of the file
    :type file_type: str
    :return: The model
    """
    if file_type is None:
        file_type = model_path.split(".")[-1]
    file_type = _parse_file_type(file_type)
    if file_type == "joblib":
        from joblib import load

        model = load(model_path)
    elif file_type == "pickle":
        import pickle

        with open(model_path, "rb") as f:
            model = pickle.load(f)
    elif file_type == "sbml":
        from cobra.io import read_sbml_model

        model = read_sbml_model(model_path)
    elif file_type == "yaml":
        from cobra.io import load_yaml_model

        model = load_yaml_model(model_path)
    elif file_type == "json":
        from cobra.io import load_json_model

        model = load_json_model(model_path)
    elif file_type == "mat":
        from cobra.io import load_matlab_model

        model = load_matlab_model(model_path)
    else:
        raise ValueError("File type not supported")
    return model


def write_model(model, model_path, file_type=None):
    """
    Write a model to a file

    :param model: Model to write
    :type model: cobra.Model
    :param model_path: Path to the model file
    :type model_path: str
    :param file_type: Type of the file
    :type file_type: str
    :return: Nothing
    """
    if file_type is None:
        file_type = model_path.split(".")[-1]
    file_type = _parse_file_type(file_type)
    if file_type == "joblib":
        from joblib import dump

        dump(model, model_path)
    elif file_type == "pickle":
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    elif file_type == "sbml":
        from cobra.io import write_sbml_model

        write_sbml_model(model, model_path)
    elif file_type == "yaml":
        from cobra.io import save_yaml_model

        save_yaml_model(model, model_path)
    elif file_type == "json":
        from cobra.io import save_json_model

        save_json_model(model, model_path)
    elif file_type == "mat":
        from cobra.io import save_matlab_model

        save_matlab_model(model, model_path)
    else:
        raise ValueError("File type not supported")


def _parse_file_type(file_type):
    """
    Parse the file type
    :param file_type: File type to parse
    :type file_type: str
    :return: Parsed file type
    :rtype: str
    """
    if file_type.lower() in ["json", "jsn"]:
        return "json"
    elif file_type.lower() in ["yaml", "yml"]:
        return "yaml"
    elif file_type.lower() in ["sbml", "xml"]:
        return "sbml"
    elif file_type.lower() in ["mat", "m", "matlab"]:
        return "mat"
    elif file_type.lower() in ["joblib", "jl", "jlb"]:
        return "joblib"
    elif file_type.lower() in ["pickle", "pkl"]:
        return "pickle"
    else:
        raise ValueError("File type not supported")
