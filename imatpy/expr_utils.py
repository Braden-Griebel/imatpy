"""
Module containing utility functions for working with gene expression
data, and converting it into qualitative weights
"""
# Standard library imports
from typing import Callable
from warnings import warn

# External imports
import cobra
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

# Local imports
from imatpy.parse_gpr import eval_gpr, gene_to_rxn_weights


def expr_to_weights(expression: pd.Series | pd.DataFrame,
                    quantile: float | tuple[float, float] = 0.15,
                    aggregator: Callable[[ArrayLike], float] = np.median,
                    sample_axis: int | str = 1,
                    ) -> pd.Series:
    """
    Convert gene expression data to qualitative weights

    :param expression: Normalized gene expression data. If it is a DataFrame
        representing multiple samples, those samples will be aggregated
        using the aggregator function (default median).
    :type expression: pd.Series | pd.DataFrame
    :param quantile: Quantile or quantiles to use for binning expression data.
        Should be between 0 and 1. If single value the bottom quantile
        will be converted to -1, the top quantile converted to 1, and
        all expression values between to 0. If a tuple is provided,
        the first is treated as the low quantile cutoff, and the second
        is treated as the high quantile cutoff.
    :type quantile: float | tuple[float, float]
    :param aggregator: Function used to aggregated gene expression data
        across samples, only used if expression is a DataFrame (default
        median).
    :type aggregator: Callable[[np.ArrayLike], float]
    :param sample_axis: Which axis represents samples in the expression
        data (only used if expression is DataFrame). "index" or 0 if rows
        represent different samples, "column" or 1 if columns represent
        different samples (default is columns).
    :type sample_axis: int | str
    :return: Series of qualitative weights, -1 for low expression, 1 for
        high expression, and 0 otherwise.
    :rtype: pd.Series

    .. note::
        The expression data should only represent biological replicates
        as it will be aggregated. If multiple different conditions are
        represented in your expression data, they should be seperated
        before this function is used.
    """
    # Convert float to tuple if necessary
    if isinstance(quantile, float):
        quantile = (quantile, 1 - quantile)
    if isinstance(expression, pd.DataFrame):
        expression = expression.apply(aggregator, axis=sample_axis)
    low, high = np.quantile(expression, quantile)
    return expression.map(
        lambda x: -1 if x <= low else (1 if x >= high else 0))


# region Conversion functions
def count_to_rpkm(count: pd.DataFrame,
                  feature_length: pd.Series) -> pd.DataFrame:
    """
    Normalize raw count data using RPKM

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: RPKM normalized counts
    :rtype: pd.DataFrame
    """
    # Ensure that the count data frame and feature length series have the same genes
    count_genes = set(count.columns)
    fl_genes = set(feature_length.index)
    if not (count_genes == fl_genes):
        warn(
            "Different genes in count dataframe and feature length series, dropping any not in common")
        genes = count_genes.intersection(fl_genes)
        count = count[genes]
        feature_length = feature_length[genes]
    sum_counts = count.sum(axis=1)
    return count.divide(feature_length, axis=1).divide(sum_counts,
                                                       axis=0) * 1.e9


def count_to_fpkm(count: pd.DataFrame,
                  feature_length: pd.Series) -> pd.DataFrame:
    """
    Converts count data to FPKM normalized expression

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows. Specifically,
        the count data represents the number of fragments, where a fragment corresponds to a single cDNA molecule, which
        can be represented by a pair of reads from each end.
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: FPKM normalized counts
    :rtype: pd.DataFrame
    """
    return count_to_rpkm(count, feature_length)


def count_to_tpm(count: pd.DataFrame,
                 feature_length: pd.Series) -> pd.DataFrame:
    """
    Converts count data to TPM normalized expression

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    # Ensure that the count data frame and feature length series have the same genes
    count_genes = set(count.columns)
    fl_genes = set(feature_length.index)
    if not (count_genes == fl_genes):
        warn(
            "Different genes in count dataframe and feature length series, dropping any not in common")
        genes = count_genes.intersection(fl_genes)
        count = count[genes]
        feature_length = feature_length[genes]
    length_normalized = count.divide(feature_length, axis=1)
    return length_normalized.divide(length_normalized.sum(axis=1),
                                    axis=0) * 1.e6


def count_to_cpm(count: pd.DataFrame) -> pd.DataFrame:
    """
    Converts count data to counts per million

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :return: CPM normalized counts
    :rtype: pd.DataFrame
    """
    total_reads = count.sum(axis=1)
    per_mil_scale = total_reads / 1e6
    return count.divide(per_mil_scale, axis=0)


def rpkm_to_tpm(rpkm: pd.DataFrame):
    """
    Convert RPKM normalized counts to TPM normalized counts

    :param rpkm: RPKM normalized count data, with genes as columns and samples as rows
    :type rpkm: pd.DataFrame
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    return rpkm.divide(rpkm.sum(axis=1), axis=0) * 1.e6


def fpkm_to_tpm(fpkm: pd.DataFrame):
    """
    Convert FPKM normalized counts to TPM normalized counts

    :param fpkm: RPKM normalized count data, with genes as columns and samples as rows
    :type fpkm: pd.DataFrame
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    return rpkm_to_tpm(fpkm)

# endregion Conversion functions
