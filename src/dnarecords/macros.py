"""
DNARecords available macros.
"""


# pylint: disable=too-few-public-methods
# It is reasonable in this case.
class DosageSparsityMacro:
    """Pre-created methods to increase sparsity of your datasets"""
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from hail import MatrixTable

    @staticmethod
    def supercharge_dosage_sparsity(mt: 'MatrixTable', info_score_threshold: float = 0.8,
                                    p_value_hwe_threshold: float = 1e-10, af_threshold: float = 0.001,
                                    sparse_threshold: float = 0.1) -> 'MatrixTable':
        """ Supercharges dosage sparsity based on info_score and variant_qc.

        Assumes the input MatrixTable to have info_score and variant_qc row fields, and dosage entry field.

        Note
        ----
        It flips the dosage (2 - dosage) for those variants where alt allele is more frequent
        and includes a flag indicating whether the flip was done or not.

        Example
        --------

        .. code-block:: python

            import dnarecords as dr

            hl = dr.helper.DNARecordsUtils.init_hail()
            hl.utils.get_1kg('/tmp/1kg')
            mt = hl.read_matrix_table('/tmp/1kg/1kg.mt')
            mt = mt.annotate_rows(info_score=hl.agg.info_score(hl.pl_to_gp(mt.PL)))
            mt = hl.variant_qc(mt)
            mt = mt.annotate_entries(dosage=hl.pl_dosage(mt.PL))
            mt = dr.macros.DosageSparsityMacro.supercharge_dosage_sparsity(mt)
            mt.select_cols().select_rows('dosage_flip') \\
               .select_entries('dosage','sparse_dosage').entries().show()

        .. code-block:: text

            +---------------+------------+-------------+-----------+----------+---------------+
            | locus         | alleles    | dosage_flip | s         |   dosage | sparse_dosage |
            +---------------+------------+-------------+-----------+----------+---------------+
            | locus<GRCh37> | array<str> |        bool | str       |  float64 |       float64 |
            +---------------+------------+-------------+-----------+----------+---------------+
            | 1:904165      | ["G","A"]  |       False | "HG00096" | 5,94e-02 |      0,00e+00 |
            | 1:904165      | ["G","A"]  |       False | "HG00099" | 3,97e-03 |      0,00e+00 |
            | 1:904165      | ["G","A"]  |       False | "HG00105" | 4,99e-03 |      0,00e+00 |
            | 1:904165      | ["G","A"]  |       False | "HG00118" | 7,88e-03 |      0,00e+00 |
            | 1:904165      | ["G","A"]  |       False | "HG00129" | 3,07e-02 |      0,00e+00 |
            | 1:904165      | ["G","A"]  |       False | "HG00148" | 7,36e-02 |      0,00e+00 |
            | 1:904165      | ["G","A"]  |       False | "HG00177" | 2,01e-01 |      2,01e-01 |
            | 1:904165      | ["G","A"]  |       False | "HG00182" | 3,83e-02 |      0,00e+00 |
            | 1:904165      | ["G","A"]  |       False | "HG00242" | 3,07e-02 |      0,00e+00 |
            | 1:904165      | ["G","A"]  |       False | "HG00254" | 1,26e-04 |      0,00e+00 |
            +---------------+------------+-------------+-----------+----------+---------------+

        :param mt: a MatrixTable with **info_score** and **variant_qc** row fields, and **dosage** entry field.
        :param info_score_threshold: rows with info_score.score below the threshold are filtered out.
        :param p_value_hwe_threshold: rows with variant_qc.p_value_hwe below the threshold are filtered out.
        :param af_threshold: rows with AF < af_threshold or AF > (1 - af_threshold) are filtered out.
        :param sparse_threshold: those entries with dosage below the threshold are set dosage = 0.
        :return: a MatrixTable with above transformations done.
        :rtype: MatrixTable
        """
        from dnarecords.helper import DNARecordsUtils
        hl = DNARecordsUtils.init_hail()
        mt = mt.filter_rows(info_score_threshold < mt.info_score.score)
        mt = mt.filter_rows(p_value_hwe_threshold < mt.variant_qc.p_value_hwe)
        mt = mt.filter_rows((af_threshold < mt.variant_qc.AF[1]) & (mt.variant_qc.AF[1] < (1 - af_threshold)))
        mt = mt.annotate_entries(sparse_dosage=hl.if_else(0.5 < mt.variant_qc.AF[1], 2 - mt.dosage, mt.dosage))
        mt = mt.annotate_rows(dosage_flip=(0.5 < mt.variant_qc.AF[1]))
        mt = mt.annotate_entries(sparse_dosage=hl.if_else(mt.sparse_dosage < sparse_threshold, 0.0, mt.sparse_dosage))
        return mt
