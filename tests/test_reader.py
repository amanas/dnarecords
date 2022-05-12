import pandas as pd
import pytest
import tensorflow as tf
import dnarecords as dr


def check_dosage_sample_wise(skeys, vkeys, mt, dna):
    s, skey = skeys.sample(1).iloc[0, :]
    vkeys_df = vkeys[['locus.contig', 'locus.position', 'alleles', 'chr_key']] \
        .assign(alleles=lambda x: ['-'.join(a) for a in x.alleles])

    src_entries = mt.filter_cols(mt.s == s).select_cols().select_rows().select_entries('dosage') \
        .entries().to_spark().toPandas()[['locus.contig', 'locus.position', 'alleles', 'dosage']] \
        .assign(alleles=lambda x: ['-'.join(a) for a in x.alleles])

    sparse_tensor = next(iter(dna.filter(lambda t: t['key'] == int(skey))))
    tgt_entries = pd.DataFrame([{'locus.contig': c[3:], 'chr_key': i, 'dosage': v}
                                for c, t in sparse_tensor.items() if c != 'key'
                                for i, v in enumerate(tf.sparse.to_dense(t).numpy())])

    print('\ntarget to source check')
    t2s = tgt_entries.merge(vkeys_df).merge(src_entries, on=['locus.contig', 'locus.position', 'alleles'])
    t2s = t2s[~t2s.dosage_y.isnull()]  # undefined entries are restored as 0
    # tfrecords has float32 precision
    assert t2s[f'dosage_x'].values == pytest.approx(t2s[f'dosage_y'].values), 'target to source mismatch'

    print('source to target check')
    s2t = src_entries.merge(vkeys_df).merge(tgt_entries, on=['locus.contig', 'chr_key'])
    s2t = s2t[~s2t.dosage_x.isnull()]  # undefined entries are restored as 0
    # tfrecords has float32 precision
    assert s2t[f'dosage_x'].values == pytest.approx(s2t[f'dosage_y'].values), 'source to target mismatch'


def check_dosage_variant_wise(skeys, vkeys, mt, dna):
    cols = ['locus.contig', 'locus.position', 'alleles', 'key']
    contig, position, alleles, vkey = vkeys.sample(1).iloc[0, :][cols]
    skeys_df = skeys[['s', 'key']]

    src_entries = mt.filter_rows((mt.locus.contig == contig) & (mt.locus.position == position) &
                                 (mt.alleles == alleles.tolist())) \
        .select_cols().select_rows().select_entries('dosage') \
        .entries().to_spark().toPandas()[['s', 'dosage']]

    sparse_tensor = next(iter(dna.filter(lambda t: t['key'] == int(vkey))))
    tgt_entries = pd.DataFrame([{'key': i, 'dosage': v}
                                for i, v in enumerate(tf.sparse.to_dense(sparse_tensor['tensor']).numpy())])

    print('\ntarget to source check')
    t2s = tgt_entries.merge(skeys_df).merge(src_entries, on=['s'])
    t2s = t2s[~t2s.dosage_y.isnull()]  # undefined entries are restored as 0
    # tfrecords has float32 precision
    assert t2s[f'dosage_x'].values == pytest.approx(t2s[f'dosage_y'].values), 'target to source mismatch'

    print('source to target check')
    s2t = src_entries.merge(skeys_df).merge(tgt_entries, on=['key'])
    s2t = s2t[~s2t.dosage_x.isnull()]  # undefined entries are restored as 0
    # tfrecords has float32 precision
    assert s2t[f'dosage_x'].values == pytest.approx(s2t[f'dosage_y'].values), 'source to target mismatch'


def test_e2e_dosage_sample_wise_records(dosage_dnarecords_reader, dosage_1kg):
    metadata = dosage_dnarecords_reader.metadata()
    skeys = metadata['skeys']
    vkeys = metadata['vkeys']
    dna = dosage_dnarecords_reader.sample_wise_dataset()
    check_dosage_sample_wise(skeys, vkeys, dosage_1kg, dna)


def test_e2e_dosage_sample_wise_parquet(dosage_dnarecords_reader, dosage_1kg):
    metadata = dosage_dnarecords_reader.metadata()
    skeys = metadata['skeys']
    vkeys = metadata['vkeys']
    dna = dosage_dnarecords_reader.sample_wise_dataset()
    check_dosage_sample_wise(skeys, vkeys, dosage_1kg, dna)


def test_e2e_dosage_variant_wise_records(dosage_dnarecords_reader, dosage_1kg):
    metadata = dosage_dnarecords_reader.metadata()
    skeys = metadata['skeys']
    vkeys = metadata['vkeys']
    dna = dosage_dnarecords_reader.variant_wise_dataset()
    check_dosage_variant_wise(skeys, vkeys, dosage_1kg, dna)


def test_e2e_dosage_variant_wise_parquet(dosage_dnarecords_reader, dosage_1kg):
    metadata = dosage_dnarecords_reader.metadata()
    skeys = metadata['skeys']
    vkeys = metadata['vkeys']
    dna = dosage_dnarecords_reader.variant_wise_dataset(batch_size=2).unbatch()
    check_dosage_variant_wise(skeys, vkeys, dosage_1kg, dna)


def test_dnarecordsreader_raises_exception_on_wrong_path():
    reader = dr.reader.DNARecordsReader('some-wrong-path')
    with pytest.raises(Exception) as ex_info:
        reader.sample_wise_dataset()
    assert ex_info.match(r"^No DNARecords found at some-wrong-path/...$")
    with pytest.raises(Exception) as ex_info:
        reader.variant_wise_dataset()
    assert ex_info.match(r"^No DNARecords found at some-wrong-path/...$")


def test_dnasparkreader_raises_exception_on_wrong_path():
    reader = dr.reader.DNASparkReader('some-wrong-path')
    with pytest.raises(Exception) as ex_info:
        reader.sample_wise_dnarecords()
    assert ex_info.match(r"^No DNARecords found at some-wrong-path/data/swrec/...$")
    with pytest.raises(Exception) as ex_info:
        reader.variant_wise_dnarecords()
    assert ex_info.match(r"^No DNARecords found at some-wrong-path/data/vwrec/...$")
    with pytest.raises(Exception) as ex_info:
        reader.sample_wise_dnaparquet()
    assert ex_info.match(r"^No DNARecords found at some-wrong-path/data/swpar/...$")
    with pytest.raises(Exception) as ex_info:
        reader.variant_wise_dnaparquet()
    assert ex_info.match(r"^No DNARecords found at some-wrong-path/data/vwpar/...$")
