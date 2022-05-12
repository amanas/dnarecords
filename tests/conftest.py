import pytest
from hail import MatrixTable, Expression
import dnarecords as dr
from dnarecords.helper import DNARecordsUtils


def utils_fn() -> DNARecordsUtils:
    return dr.helper.DNARecordsUtils()


@pytest.fixture(autouse=True, scope='session')
def utils():
    return utils_fn()


def dosage_1kg_fn() -> MatrixTable:
    hl = dr.helper.DNARecordsUtils.init_hail()
    hl.utils.get_1kg('/tmp/1kg')
    mt = hl.read_matrix_table('/tmp/1kg/1kg.mt')
    mt = mt.annotate_entries(dosage=hl.pl_dosage(mt.PL))
    return mt


@pytest.fixture(autouse=True, scope='session')
def dosage_1kg():
    return dosage_1kg_fn()


def dosage_output_fn() -> str:
    return '/tmp/dnarecords/output'


@pytest.fixture(autouse=True, scope='session')
def dosage_output():
    return dosage_output_fn()


def dosage_write_dnarecords_fn(expr: Expression, output: str) -> None:
    dr.writer.DNARecordsWriter(expr) \
        .write(output, sparse=True, sample_wise=True, variant_wise=True, tfrecord_format=True, parquet_format=True,
               write_mode='overwrite', gzip=True)


@pytest.fixture(autouse=True, scope='session')
def dosage_write_dnarecords(dosage_1kg, dosage_output):
    return dosage_write_dnarecords_fn(dosage_1kg.dosage, dosage_output)


def dosage_dnaspark_reader_fn(output: str) -> dr.reader.DNASparkReader:
    return dr.reader.DNASparkReader(output)


@pytest.fixture(autouse=True, scope='session')
def dosage_dnaspark_reader(dosage_write_dnarecords, dosage_output):
    return dosage_dnaspark_reader_fn(dosage_output)


def dosage_dnarecords_reader_fn(output: str) -> dr.reader.DNARecordsReader:
    return dr.reader.DNARecordsReader(output)


@pytest.fixture(autouse=True, scope='session')
def dosage_dnarecords_reader(dosage_write_dnarecords, dosage_output):
    return dosage_dnarecords_reader_fn(dosage_output)
