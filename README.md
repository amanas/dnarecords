# DNARecords

[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
![example workflow](https://github.com/amanas/dnarecords/actions/workflows/ci-cd.yml/badge.svg)
[![codecov](https://codecov.io/gh/amanas/dnarecords/branch/main/graph/badge.svg)](https://codecov.io/gh/amanas/dnarecords)
![pylint Score](https://mperlet.github.io/pybadge/badges/9.97.svg)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

**Genomics data ML ready.**

Transform your vcf, bgen, etc. genomics datasets into a sample wise format so that you can use it 
for Deep Learning models. 

## Installation

DNARecords package has two main dependencies:

* **Hail**, if you are transforming your genomics data into DNARecords
* **Tensorflow**, if you are using a previously DNARecords dataset, for example, to train a DL model

As you may know, Tensorflow and Spark does not play very well together on a cluster with more than one machine.

However, `dnarecords` package needs to be installed **only on the driver machine** of a Hail cluster.

For that reason, we recommend following these installation tips.

### **On a dev environment**

```bash
$ pip install dnarecords
```

For further details (or any trouble), review [Local environments](LOCAL_ENVS.md) section.

### **On a Hail cluster or submitting a job to it**

You will already have Pyspark installed and will not intend to install Tensorflow. 

So, just install dnarecords without dependencies on the driver machine. 

There will be no missing modules as soon as you use the classes and functionality intended for Spark.

```bash
$ /opt/conda/miniconda3/bin/python -m pip install dnarecords --no-deps
```
*Note: assuming Hail python executable is /opt/conda/miniconda3/bin/python* 

### **On a Tensorflow environment or submitting a job to it**

You will already have Tensorflow installed and will not intend to install Pyspark. 

So, just install dnarecords without dependencies. 

There will be no missing modules as soon as you use the classes and functionality intended for Tensorflow.

```bash
$ pip install dnarecords --no-deps
```


## Working on Google Dataproc

Just use and initialization action that installs `dnarecords` without dependencies.

```bash
$ hailctl dataproc start dnarecords --init gs://dnarecords/dataproc-init.sh
```
Iy you need to work with other cloud providers, refer to [Hail docs](https://hail.is/docs/0.2/getting_started.html).

## Usage

It is quite straightforward to understand the functionality of the package.

Given some genomics data, you can transform it into a **DNARecords Dataset** this way:


```python
import dnarecords as dr


hl = dr.helper.DNARecordsUtils.init_hail()
hl.utils.get_1kg('/tmp/1kg')
mt = hl.read_matrix_table('/tmp/1kg/1kg.mt')
mt = mt.annotate_entries(dosage=hl.pl_dosage(mt.PL))

dnarecords_path = '/tmp/dnarecords'
writer = dr.writer.DNARecordsWriter(mt.dosage)
writer.write(dnarecords_path, sparse=True, sample_wise=True, variant_wise=True,
             tfrecord_format=True, parquet_format=True,
             write_mode='overwrite', gzip=True)
```


Given a DNARecords Dataset, you can read it as **Tensorflow Datasets** this way:

```python
import dnarecords as dr


dnarecords_path = '/tmp/dnarecords'
reader = dr.reader.DNARecordsReader(dnarecords_path)
samplewise_ds = reader.sample_wise_dataset()
variantwise_ds = reader.variant_wise_dataset()
```

Or, given a DNARecords Dataset, you can read it as **Pyspark DataFrames** this way:


```python
import dnarecords as dr


dnarecords_path = '/tmp/dnarecords'
reader = dr.reader.DNASparkReader(dnarecords_path)
samplewise_df = reader.sample_wise_dnarecords()
variantwise_df = reader.variant_wise_dnarecords()
```

We will provide more examples and integrations soon.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`dnarecords` was created by Atray Dixit, Andrés Mañas Mañas, Lucas Seninge. It is licensed under the terms of the MIT license.

## Credits

`dnarecords` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
