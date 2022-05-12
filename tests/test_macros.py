import random

import dnarecords as dr


def test_supercharge_dosage_sparsity(dosage_1kg):
    hl = dr.helper.DNARecordsUtils.init_hail()

    mt = dosage_1kg
    mt = mt.annotate_rows(info_score=hl.agg.info_score(hl.pl_to_gp(mt.PL)))
    mt = hl.variant_qc(mt)

    info_score_threshold = random.uniform(0.75, 0.85)
    p_value_hwe_threshold = random.uniform(1e-11, 1e-9)
    af_threshold = random.uniform(0.001, 0.002)
    sparse_threshold = random.uniform(0.1, 0.2)

    macro = dr.macros.DosageSparsityMacro
    smt = macro.supercharge_dosage_sparsity(mt, info_score_threshold=info_score_threshold,
                                            p_value_hwe_threshold=p_value_hwe_threshold, af_threshold=af_threshold,
                                            sparse_threshold=sparse_threshold)

    mt = mt.filter_rows(info_score_threshold < mt.info_score.score)
    mt = mt.filter_rows(p_value_hwe_threshold < mt.variant_qc.p_value_hwe)
    mt = mt.filter_rows((af_threshold < mt.variant_qc.AF[1]) & (mt.variant_qc.AF[1] < (1 - af_threshold)))
    mt = mt.annotate_entries(sparse_dosage=hl.if_else(0.5 < mt.variant_qc.AF[1], 2 - mt.dosage, mt.dosage))
    mt = mt.annotate_rows(dosage_flip=(0.5 < mt.variant_qc.AF[1]))
    mt = mt.annotate_entries(sparse_dosage=hl.if_else(mt.sparse_dosage < sparse_threshold, 0.0, mt.sparse_dosage))

    assert mt.dosage_flip.collect() == smt.dosage_flip.collect(), 'mismatch in dosage flip flag'
    assert mt.sparse_dosage.collect() == smt.sparse_dosage.collect(), 'mismatch in sparse dosage values'
