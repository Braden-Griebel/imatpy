# Standard Library Imports
import unittest
# External imports
import numpy as np
import pandas as pd
# Local imports
from imatpy.expr_utils import (count_to_rpkm, count_to_fpkm, count_to_tpm,
                               rpkm_to_tpm, fpkm_to_tpm, count_to_cpm,
                               expr_to_weights)


class TestConversionFunctions(unittest.TestCase):
    feature_length = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up values for use in the tests, mainly fake expression data
        """
        gene_list = ["g_%i" % i for i in range(1, 101)]
        sample_list = ["g_%i" % i for i in range(1, 6)]
        # create rng generator
        rng = np.random.default_rng(42)
        # Generate random count values of the right size
        count_values = rng.integers(low=0, high=500_000, size=(5, 100))
        cls.count_data = pd.DataFrame(count_values, index=sample_list,
                                      columns=gene_list)
        # create feature length data
        feature_length_data = rng.integers(low=20, high=500, size=100)
        cls.feature_length = pd.Series(feature_length_data, index=gene_list)
        # Create feature_length with missing genes
        cls.feature_length_missing = cls.feature_length.iloc[0:90]
        # Also create small examples with known values
        cls.small_counts = pd.DataFrame({
            "A": [83, 50, 85, 17, 91],
            "B": [20, 6, 23, 45, 75],
            "C": [54, 65, 53, 97, 38],
            "D": [58, 79, 77, 32, 64],
        })
        cls.small_feature_length = pd.Series([10, 50, 25, 60],
                                             index=["A", "B", "C", "D"])

    def test_count_to_rpkm(self):
        """
        Test the count_to_rpkm function
        """
        # Calculate the rpkm from count data
        rpkm = count_to_rpkm(self.count_data, self.feature_length)
        # Should return a dataframe
        self.assertIsInstance(rpkm, pd.DataFrame)
        # Should be the same size as before
        self.assertEqual(rpkm.shape, self.count_data.shape)
        # Should have the same column names as before
        self.assertTrue(rpkm.columns.equals(self.count_data.columns))
        # Should have the same index as count data
        self.assertTrue(rpkm.index.equals(self.count_data.index))
        # Use known values for the small examples
        small_rpkm = count_to_rpkm(self.small_counts,
                                   self.small_feature_length)
        known_rpkm = pd.DataFrame({
            "A": [3.860465e+07, 2.500000e+07, 3.571429e+07, 8.900524e+06,
                  3.395522e+07],
            "B": [1.860465e+06, 6.000000e+05, 1.932773e+06, 4.712042e+06,
                  5.597015e+06],
            "C": [1.004651e+07, 1.300000e+07, 8.907563e+06, 2.031414e+07,
                  5.671642e+06],
            "D": [4.496124e+06, 6.583333e+06, 5.392157e+06, 2.792321e+06,
                  3.980100e+06],
        })
        self.assertTrue(np.all(np.isclose(known_rpkm, small_rpkm)))
        # Check that it throws a warning when given incomplete gene data
        with self.assertWarns(Warning):
            count_to_rpkm(self.count_data, self.feature_length_missing)

    def test_count_to_fpkm(self):
        """
        Test the count_to_fpkm function
        """
        # Since this is just a wrapper, just test that it returns the same
        # as count_to_rpkm
        fpkm = count_to_fpkm(self.count_data, self.feature_length)
        rpkm = count_to_rpkm(self.count_data, self.feature_length)
        self.assertTrue(np.all(np.isclose(fpkm, rpkm)))
        # Check that it throws a warning when given incomplete gene data
        with self.assertWarns(Warning):
            count_to_fpkm(self.count_data, self.feature_length_missing)

    def test_count_to_tpm(self):
        """
        Test the count_to_tpm function
        """
        tpm = count_to_tpm(self.count_data, self.feature_length)
        self.assertTrue(tpm.index.equals(self.count_data.index))
        self.assertTrue(tpm.columns.equals(self.count_data.columns))
        self.assertIsInstance(tpm, pd.DataFrame)
        self.assertEqual(tpm.shape, self.count_data.shape)
        # Check that it throws a warning when given incomplete gene data
        with self.assertWarns(Warning):
            count_to_tpm(self.count_data, self.feature_length_missing)
        # Test that the sum of each row is 1e6
        self.assertTrue(np.all(np.isclose(tpm.sum(axis=1), 1e6)))
        known_tpm = pd.DataFrame({
            "A": [701803.833145, 553301.364810, 687516.850903, 242395.437262,
                  690091.001011],
            "B": [33821.871477, 13279.232755, 37206.794284, 128326.996198,
                  113751.263903],
            "C": [182638.105975, 287716.709701, 171474.791049, 553231.939163,
                  115267.947422],
            "D": [81736.189402, 145702.692733, 103801.563764, 76045.627376,
                  80889.787664],
        })
        self.assertTrue(np.all(
            np.isclose(known_tpm, count_to_tpm(self.small_counts,
                                               self.small_feature_length))))

    def test_count_to_cpm(self):
        """
        Test the count_to_cpm function
        """
        cpm = count_to_cpm(self.count_data)
        self.assertIsInstance(cpm, pd.DataFrame)
        self.assertTrue(cpm.index.equals(self.count_data.index))
        self.assertTrue(cpm.columns.equals(self.count_data.columns))
        self.assertTrue(np.all(np.isclose(cpm.sum(axis=1), 1e6)))

    def test_rpkm_to_tpm(self):
        """
        Test the rpkm_to_tpm function
        """
        tpm_known = count_to_tpm(self.count_data, self.feature_length)
        rpkm = count_to_rpkm(self.count_data, self.feature_length)
        tpm_test = rpkm_to_tpm(rpkm)
        self.assertIsInstance(tpm_test, pd.DataFrame)
        self.assertTrue(tpm_test.index.equals(rpkm.index))
        self.assertTrue(tpm_test.columns.equals(rpkm.columns))
        self.assertTrue(np.all(np.isclose(tpm_test, tpm_known)))

    def test_fpkm_to_tpm(self):
        """
        Test the fpkm_to_tpm function
        """
        rpkm = count_to_rpkm(self.count_data, self.feature_length)
        fpkm = count_to_fpkm(self.count_data, self.feature_length)
        tpm_rpkm = rpkm_to_tpm(rpkm)
        tpm_fpkm = fpkm_to_tpm(fpkm)
        self.assertTrue(np.all(np.isclose(tpm_rpkm, tpm_fpkm)))


class TestConversionToWeights(unittest.TestCase):
    def test_series_conversion(self):
        test_series = pd.Series([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        quantile = 0.1, 0.9
        expected = pd.Series([-1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        actual = expr_to_weights(test_series, quantile)
        self.assertTrue(np.all(actual == expected))


if __name__ == '__main__':
    unittest.main()
