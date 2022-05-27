import re
import pandas as pd
import pytest
import dnarecords as dr
import glob
from itertools import groupby
import numpy as np


def check_dosage_sample_wise(skeys, vkeys, mt, dna, approx):
    s, skey = skeys.sample(1).iloc[0, :]
    vkeys_df = vkeys[['locus.contig', 'locus.position', 'alleles', 'chr_key']] \
        .assign(alleles=lambda x: ['-'.join(a) for a in x.alleles])

    src_entries = mt.filter_cols(mt.s == s).select_cols().select_rows().select_entries('dosage') \
        .entries().to_spark().toPandas()[['locus.contig', 'locus.position', 'alleles', 'dosage']] \
        .assign(alleles=lambda x: ['-'.join(a) for a in x.alleles])

    tgt_entries = pd.DataFrame([{'locus.contig': c, 'chr_key': k, 'dosage': v}
                                for c in {re.search(r'chr([A-Za-z\d]+)_', c).group(1)
                                          for c in dna.columns if c != 'key'}
                                for i, r in dna.filter(dna.key == int(skey)).toPandas().iterrows()
                                for k, v in zip(r[f'chr{c}_indices'], r[f'chr{c}_values'])])

    print('\ndense shape check')
    df = dna.filter(dna.key == int(skey)).toPandas()
    df = df[[col for col in df if col.endswith('dense_shape')]]
    assert mt.count_rows() == np.sum(df.values), "dense_shapes error"

    print('\ntarget to source check')
    t2s = tgt_entries.merge(vkeys_df).merge(src_entries, on=['locus.contig', 'locus.position', 'alleles'])
    if approx:
        # tfrecords has float32 precision
        assert t2s[f'dosage_x'].values == pytest.approx(t2s[f'dosage_y'].values), 'target to source mismatch'
    else:
        assert all([x == y for x, y in zip(t2s[f'dosage_x'], t2s[f'dosage_y'])]), 'target to source mismatch'

    print('source to target check')
    s2t = src_entries.merge(vkeys_df).merge(tgt_entries, on=['locus.contig', 'chr_key'])
    if approx:
        # tfrecords has float32 precision
        assert s2t[f'dosage_x'].values == pytest.approx(s2t[f'dosage_y'].values), 'source to target mismatch'
    else:
        assert all([x == y for x, y in zip(s2t[f'dosage_x'], s2t[f'dosage_y'])]), 'source to target mismatch'


def check_dosage_variant_wise(skeys, vkeys, mt, dna, approx):
    cols = ['locus.contig', 'locus.position', 'alleles', 'key']
    contig, position, alleles, vkey = vkeys.sample(1).iloc[0, :][cols]
    skeys_df = skeys[['s', 'key']]

    src_entries = \
        mt.filter_rows((mt.locus.contig == contig) & (mt.locus.position == position) & (mt.alleles == alleles)) \
            .select_cols().select_rows().select_entries('dosage') \
            .entries().to_spark().toPandas()[['s', 'dosage']]

    tgt_entries = pd.DataFrame([{'key': i, 'dosage': v}
                                for i, r in dna.filter(dna.key == int(vkey)).toPandas().iterrows()
                                for i, v in zip(r['indices'], r['values'])])

    print('\ndense shape check')
    df = dna.filter(dna.key == int(vkey)).toPandas()
    assert mt.count_cols() == df.dense_shape[0], "dense_shapes error"

    print('\ntarget to source check')
    t2s = tgt_entries.merge(skeys_df).merge(src_entries, on=['s'])
    if approx:
        # tfrecords has float32 precision
        assert t2s[f'dosage_x'].values == pytest.approx(t2s[f'dosage_y'].values), 'target to source mismatch'
    else:
        assert all([x == y for x, y in zip(t2s[f'dosage_x'], t2s[f'dosage_y'])]), 'target to source mismatch'

    print('source to target check')
    s2t = src_entries.merge(skeys_df).merge(tgt_entries, on=['key'])
    if approx:
        # tfrecords has float32 precision
        assert s2t[f'dosage_x'].values == pytest.approx(s2t[f'dosage_y'].values), 'source to target mismatch'
    else:
        assert all([x == y for x, y in zip(s2t[f'dosage_x'], s2t[f'dosage_y'])]), 'source to target mismatch'


def test_e2e_dosage_sample_wise_records(dosage_dnaspark_reader, dosage_1kg):
    metadata = dosage_dnaspark_reader.metadata()
    skeys = metadata['skeys'].toPandas()
    vkeys = metadata['vkeys'].toPandas()
    dna = dosage_dnaspark_reader.sample_wise_dnarecords()
    check_dosage_sample_wise(skeys, vkeys, dosage_1kg, dna, True)


def test_e2e_dosage_sample_wise_parquet(dosage_dnaspark_reader, dosage_1kg):
    metadata = dosage_dnaspark_reader.metadata()
    skeys = metadata['skeys'].toPandas()
    vkeys = metadata['vkeys'].toPandas()
    dna = dosage_dnaspark_reader.sample_wise_dnaparquet()
    check_dosage_sample_wise(skeys, vkeys, dosage_1kg, dna, False)


def test_e2e_dosage_variant_wise_records(dosage_dnaspark_reader, dosage_1kg):
    metadata = dosage_dnaspark_reader.metadata()
    skeys = metadata['skeys'].toPandas()
    vkeys = metadata['vkeys'].toPandas()
    dna = dosage_dnaspark_reader.variant_wise_dnarecords()
    check_dosage_variant_wise(skeys, vkeys, dosage_1kg, dna, True)


def test_e2e_dosage_variant_wise_parquet(dosage_dnaspark_reader, dosage_1kg):
    metadata = dosage_dnaspark_reader.metadata()
    skeys = metadata['skeys'].toPandas()
    vkeys = metadata['vkeys'].toPandas()
    dna = dosage_dnaspark_reader.variant_wise_dnaparquet()
    check_dosage_variant_wise(skeys, vkeys, dosage_1kg, dna, False)


def test_raises_exception_when_not_numeric_expr(dosage_1kg):
    mt = dosage_1kg.annotate_entries(non_numeric_expr=dosage_1kg.PL)
    with pytest.raises(Exception) as ex_info:
        dr.writer.DNARecordsWriter(mt.non_numeric_expr)
    assert ex_info.match(r"^Only expr_numeric are supported at this time.*")


def test_raises_exception_when_not_sample_wise_and_not_variant_wise(dosage_1kg, dosage_output):
    with pytest.raises(Exception) as ex_info:
        dr.writer.DNARecordsWriter(dosage_1kg.dosage).write(dosage_output, sample_wise=False, variant_wise=False)
    assert ex_info.match(r"^At least one of sample_wise, variant_wise must be True$")


def test_raises_exception_when_not_tfrecord_format_and_not_parquet_format(dosage_1kg, dosage_output):
    with pytest.raises(Exception) as ex_info:
        dr.writer.DNARecordsWriter(dosage_1kg.dosage).write(dosage_output, tfrecord_format=False, parquet_format=False)
    assert ex_info.match(r"^At least one of tfrecord_format, parquet_format must be True$")


def test_generates_single_file_per_block(dosage_output):
    kv_blocks = dosage_output.replace('/output', '/staging/kv-blocks')
    pairs = [f.rsplit('/', 1) for f in glob.glob(f"{kv_blocks}/*/*/*.parquet")]
    for r, g in groupby(pairs, key=lambda t: t[0]):
        files = list(g)
        assert len(list(files)) == 1, "kv-blocks with more than one file exists"
