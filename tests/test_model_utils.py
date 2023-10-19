# Core Library modules
import os.path
import pathlib
import unittest

# Local imports
from imatpy.model_utils import read_model, write_model, _parse_file_type


class TestParseFileType(unittest.TestCase):
    """
    Class for testing if _parse_file_type function works correctly
    """

    def test_parse_file_type(self):
        self.assertEqual(_parse_file_type("joblib"), "joblib")
        self.assertEqual(_parse_file_type("pkl"), "pickle")
        self.assertEqual(_parse_file_type("pickle"), "pickle")
        self.assertEqual(_parse_file_type("yml"), "yaml")
        self.assertEqual(_parse_file_type("xml"), "sbml")
        self.assertEqual(_parse_file_type("jsn"), "json")
        self.assertEqual(_parse_file_type("m"), "mat")
        self.assertEqual(_parse_file_type("mat"), "mat")
        self.assertEqual(_parse_file_type("sbml"), "sbml")


class TestModelIO(unittest.TestCase):
    """
    Class for testing if read_model function works correctly
    """

    def test_read_model(self):
        data_path = str(pathlib.Path(__file__).parent.joinpath("data"))
        model_json = read_model(os.path.join(data_path, "textbook_model.json"))
        model_xml = read_model(os.path.join(data_path, "textbook_model.xml"))
        model_mat = read_model(os.path.join(data_path, "textbook_model.mat"))
        model_yaml = read_model(os.path.join(data_path, "textbook_model.yaml"))
        for rxn in model_json.reactions:
            self.assertTrue(rxn in model_xml.reactions)
            self.assertTrue(rxn in model_mat.reactions)
            self.assertTrue(rxn in model_yaml.reactions)
        for met in model_json.metabolites:
            self.assertTrue(met in model_xml.metabolites)
            self.assertTrue(met in model_mat.metabolites)
            self.assertTrue(met in model_yaml.metabolites)
        for gene in model_json.genes:
            self.assertTrue(gene in model_xml.genes)
            self.assertTrue(gene in model_mat.genes)
            self.assertTrue(gene in model_yaml.genes)

    def test_write_model(self):
        data_path = str(pathlib.Path(__file__).parent.joinpath("data"))
        out_dir = str(pathlib.Path(__file__).parent.joinpath("data").joinpath("temp"))
        try:
            os.mkdir(out_dir)
            model = read_model(os.path.join(data_path, "textbook_model.json"))
            write_model(model, os.path.join(out_dir, "textbook_model.json"))
            write_model(model, os.path.join(out_dir, "textbook_model.xml"))
            write_model(model, os.path.join(out_dir, "textbook_model.yaml"))
            write_model(model, os.path.join(out_dir, "textbook_model.mat"))
            self.assertTrue(
                os.path.exists(os.path.join(out_dir, "textbook_model.json"))
            )
            self.assertTrue(os.path.exists(os.path.join(out_dir, "textbook_model.xml")))
            self.assertTrue(
                os.path.exists(os.path.join(out_dir, "textbook_model.yaml"))
            )
            self.assertTrue(os.path.exists(os.path.join(out_dir, "textbook_model.mat")))
            model_json = read_model(os.path.join(out_dir, "textbook_model.json"))
            model_mat = read_model(os.path.join(out_dir, "textbook_model.mat"))
            model_xml = read_model(os.path.join(out_dir, "textbook_model.xml"))
            model_yaml = read_model(os.path.join(out_dir, "textbook_model.yaml"))
            for rxn in model_json.reactions:
                self.assertTrue(rxn in model_xml.reactions)
                self.assertTrue(rxn in model_mat.reactions)
                self.assertTrue(rxn in model_yaml.reactions)
            for met in model_json.metabolites:
                self.assertTrue(met in model_xml.metabolites)
                self.assertTrue(met in model_mat.metabolites)
                self.assertTrue(met in model_yaml.metabolites)
            for gene in model_json.genes:
                self.assertTrue(gene in model_xml.genes)
                self.assertTrue(gene in model_mat.genes)
                self.assertTrue(gene in model_yaml.genes)
        finally:
            if os.path.exists(out_dir):
                if os.path.exists(os.path.join(out_dir, "textbook_model.json")):
                    os.remove(os.path.join(out_dir, "textbook_model.json"))
                if os.path.exists(os.path.join(out_dir, "textbook_model.xml")):
                    os.remove(os.path.join(out_dir, "textbook_model.xml"))
                if os.path.exists(os.path.join(out_dir, "textbook_model.yaml")):
                    os.remove(os.path.join(out_dir, "textbook_model.yaml"))
                if os.path.exists(os.path.join(out_dir, "textbook_model.mat")):
                    os.remove(os.path.join(out_dir, "textbook_model.mat"))
                os.rmdir(out_dir)


if __name__ == "__main__":
    unittest.main()
