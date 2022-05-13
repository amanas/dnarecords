# Local environments

When you are working on your local environment, you expect all dependencies of DNARecords to be properly installed on 
your environment.

**It is highly recommendable to install DNARecords on a new and empty environment**, whatever environment
manager you prefer.

Here you can see a few ways to set up a local/dev environment.

## Test script

To test the installation is correct, just copy this script into a file named `dnarecords-test.py`.


```python
import dnarecords as dr


hl = dr.helper.DNARecordsUtils.init_hail()
hl.utils.get_1kg('/tmp/1kg')
mt = hl.read_matrix_table('/tmp/1kg/1kg.mt').head(100,100)
mt = mt.annotate_entries(dosage=hl.pl_dosage(mt.PL))

path = '/tmp/dnarecords'
writer = dr.writer.DNARecordsWriter(mt.dosage)
writer.write(path, sparse=True, write_mode='overwrite', gzip=True,
             sample_wise=True, variant_wise=True,
             tfrecord_format=True, parquet_format=True)

spark_reader = dr.reader.DNASparkReader(path)
cols = ['key','chr1_indices','chr1_values','chr1_dense_shape']
spark_reader.sample_wise_dnarecords().select(cols).show(1)
spark_reader.sample_wise_dnaparquet().select(cols).show(1)
spark_reader.variant_wise_dnarecords().show(1)
spark_reader.variant_wise_dnaparquet().show(1)

tensor_reader = dr.reader.DNARecordsReader(path)
print(next(iter(tensor_reader.sample_wise_dataset())))
print(next(iter(tensor_reader.variant_wise_dataset())))
```

## Conda

```bash
$ conda create --prefix ./dna-conda pip
$ conda activate ./dna-conda
$ pip install dnarecords
$ python dnarecords-test.py
```

Or, if you prefer, you can test it from a jupyter-lab notebook:

```bash
$ pip install jupyterlab
$ jupyter-lab
```

And now, in any cell:

```python
%run dnarecords-test.py
```


## venv

```bash
$ python3 -m venv dna-venv
$ source dna-venv/bin/activate
$ pip install dnarecords
$ python dnarecords-test.py
```

You can run it with a jupyter-lab notebook as well.

## poetry

```bash
$ git clone https://github.com/amanas/dnarecords.git dna-poetry
$ cd dna-poetry
$ poetry shell
$ pip install dnarecords
$ python ../dnarecords-test.py
```

You can run it with a jupyter-lab notebook as well.
