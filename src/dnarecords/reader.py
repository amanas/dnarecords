"""
DNARecords available readers.
"""


class DNARecordsReader:
    """DNARecords Tensorflow reader. **Sample and variant wise**.

    Genomics data ML ready for frameworks like Tensorflow or Pytorch.

    * Consume the data in a **variant wise fashion (common GWAS analysis)**.
    * Or consume the data in a **sample wise fashion (Deep Learning models)**.
    * **Tested on UKBB**.

    Example
    --------

    .. code-block:: python

        import dnarecords as dr
        import tensorflow as tf


        output = '/tmp/dnarecords/output'
        reader = dr.reader.DNARecordsReader('/tmp/dnarecords/output')

        swds = reader.sample_wise_dataset()
        tf.print(next(iter(swds)))

        vwds = reader.variant_wise_dataset()
        tf.print(next(iter(vwds)))


    .. code-block:: text

        {'key': 191,
         'chr1': 'SparseTensor(indices=[[0] [1] [2] ... [924] [925] [926]],
                               values=[0.200760081 0.200760037 0.200760067 ... 0.0019912892 1.96934652 0.00396528561],
                               shape=[909])',
         'chr10': 'SparseTensor(indices=[[124] [125] [126] ... [665] [666] [667]],
                                values=[1.01560163 0.0306534301 1.99800873 ... 0.999999881 1.01956224 0.111815773],
                                shape=[532])',
          ... }
         ...

        {'key': 3764,
         'tensor': 'SparseTensor(indices=[[0] [1] [2] ... [281] [282] [283]],
                                 values=[0.111815773 0.015601662 0.00788068399 ... 0.0593509413 0.000500936178],
                                 shape=[10880])'}
        ...

    :param dnarecords_path: root path to your DNARecords created with :obj:`.DNARecordsWriter.write`
    :param gzip: whether your tfrecords are gzipped or not. Default: True.

    See Also
    --------
    :obj:`.DNARecordsWriter.write`
    """

    from typing import Dict, List
    from pandas import DataFrame
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from tensorflow.python.data import Dataset

    def __init__(self, dnarecords_path, gzip=True):
        self._dnarecords_path = dnarecords_path
        self._gzip = gzip

    @staticmethod
    def _types_dict():
        import tensorflow as tf
        return {'long': tf.int64,
                'double': tf.float32}

    @staticmethod
    def _pandas_safe_read_parquet(path, columns, taste):
        import pandas as pd
        import tensorflow as tf

        files = tf.io.gfile.glob(f'{path}/*.parquet')
        if files:
            if columns:
                if taste:
                    return pd.read_parquet(files[0], columns=columns)
                else:
                    return pd.concat(pd.read_parquet(f, columns=columns) for f in files)
            if not columns:
                if taste:
                    return pd.read_parquet(files[0])
                else:
                    return pd.concat(pd.read_parquet(f) for f in files)
        return None

    @staticmethod
    def _pandas_safe_read_json(path):
        import pandas as pd
        import tensorflow as tf

        files = tf.io.gfile.glob(f'{path}/*.json')
        if files:
            return pd.concat(pd.read_json(f) for f in files)
        return None

    def metadata(self, vkeys_columns: List[str] = None, skeys_columns: List[str] = None, taste: bool = False) -> Dict[str, DataFrame]:
        """Gets the metadata associated to the DNARecords dataset as a dictionary of names to pandas DataFrames.

        :rtype: Dict[str, DataFrame].
        :return: the metadata associated to the DNARecords as a dictionary of names to pandas DataFrames.
        :param vkeys_columns: columns to return from variant metadata files (potentially big files). Defaults to None (all columns).
        :param skeys_columns: columns to return from sample metadata files (potentially big files). Defaults to None (all columns).
        :param taste: The full metadata DataFrames could be huge, wo you can get a taste of them without going into memory issues. With that, decide wich columns to get metadata for. Defaults to False.

        See Also
        --------
        :obj:`.DNARecordsUtils.dnarecords_tree` to know the available keys
        """
        import dnarecords as dr

        result = {}
        tree = dr.helper.DNARecordsUtils.dnarecords_tree(self._dnarecords_path)
        for k, v in tree.items():
            if k == 'skeys':
                result.update({k: self._pandas_safe_read_parquet(v, skeys_columns, taste)})
            if k == 'vkeys':
                result.update({k: self._pandas_safe_read_parquet(v, vkeys_columns, taste)})
            if k in ['swpfs', 'vwpfs', 'swrfs', 'vwrfs']:
                result.update({k: self._pandas_safe_read_parquet(v, None, False)})
            if k in ['swpsc', 'vwpsc', 'swrsc', 'vwrsc']:
                result.update({k: self._pandas_safe_read_json(v)})
        return result

    def datafiles(self) -> Dict[str, List[str]]:
        """Gets the paths of the DNARecords dataset files as a dictionary of names to List of paths.

        :rtype: Dict[str, List[str]].
        :return: the dnarecord files associated to the DNARecords as a dictionary of names to List of paths.

        See Also
        --------
        :obj:`.DNARecordsUtils.dnarecords_tree` to know the available keys
        """
        import tensorflow as tf
        import dnarecords as dr

        result = {}
        tree = dr.helper.DNARecordsUtils.dnarecords_tree(self._dnarecords_path)
        result.update({'swrec': list(set(tf.io.gfile.glob(f"{tree['swrec']}/*.tfrecord") +
                                         tf.io.gfile.glob(f"{tree['swrec']}/*.tfrecord.gz")))})
        result.update({'vwrec': list(set(tf.io.gfile.glob(f"{tree['vwrec']}/*.tfrecord") +
                                         tf.io.gfile.glob(f"{tree['vwrec']}/*.tfrecord.gz")))})
        result.update({'swpar': list(set(tf.io.gfile.glob(f"{tree['swpar']}/*.parquet") +
                                         tf.io.gfile.glob(f"{tree['swpar']}/*.parquet.gz")))})
        result.update({'vwpar': list(set(tf.io.gfile.glob(f"{tree['vwpar']}/*.parquet") +
                                         tf.io.gfile.glob(f"{tree['vwpar']}/*.parquet.gz")))})
        return result

    @staticmethod
    def _sw_decoder(dnarecords, schema, gzip):
        import json
        import tensorflow as tf
        one_proto = next(iter(tf.data.TFRecordDataset(dnarecords, 'GZIP' if gzip else None)))
        swrsc_dict = {f['fields']['name']: f['fields'] for _, f in schema.iterrows()}
        features = {'key': tf.io.FixedLenFeature([], tf.int64)}
        for indices_field in [field for field in swrsc_dict.keys() if field.endswith('indices')]:
            feature_name = indices_field.replace('_indices', '')
            dense_shape_field = indices_field.replace('indices', 'dense_shape')
            values_field = indices_field.replace('indices', 'values')
            values_type = DNARecordsReader._types_dict()[json.loads(swrsc_dict[values_field]['type'])['elementType']]
            dense_shape = tf.io.parse_example(one_proto, {dense_shape_field: tf.io.FixedLenFeature([], tf.int64)})[
                dense_shape_field]
            features.update({feature_name: tf.io.SparseFeature(indices_field, values_field, values_type, dense_shape)})
        return lambda proto: tf.io.parse_example(proto, features)

    @staticmethod
    def _vw_decoder(dnarecords, schema, gzip):
        import json
        import tensorflow as tf

        one_proto = next(iter(tf.data.TFRecordDataset(dnarecords, 'GZIP' if gzip else None)))
        vwrsc_dict = {f['fields']['name']: f['fields'] for _, f in schema.iterrows()}
        values_type = DNARecordsReader._types_dict()[json.loads(vwrsc_dict['values']['type'])['elementType']]
        dense_shape = tf.io.parse_example(one_proto, {'dense_shape': tf.io.FixedLenFeature([], tf.int64)})[
            'dense_shape']
        features = {'key': tf.io.FixedLenFeature([], tf.int64),
                    'tensor': tf.io.SparseFeature('indices', 'values', values_type, dense_shape)}
        return lambda proto: tf.io.parse_example(proto, features)

    # pylint: disable=too-many-arguments
    # It is reasonable in this case.
    def _dataset(self, dnarecords, decoder, num_parallel_reads, num_parallel_calls, deterministic, drop_remainder,
                 batch_size, buffer_size):
        import tensorflow as tf

        ds = tf.data.TFRecordDataset(dnarecords, 'GZIP' if self._gzip else None, buffer_size, num_parallel_reads)
        if batch_size:
            ds = ds.batch(batch_size, drop_remainder, num_parallel_calls, deterministic)
        return ds.map(decoder, num_parallel_calls, deterministic)

    # pylint: disable=too-many-arguments
    # It is reasonable in this case.
    def sample_wise_dataset(self, num_parallel_reads: int = -1, num_parallel_calls: int = -1,
                            deterministic: bool = False, drop_remainder: bool = False, batch_size: int = None,
                            buffer_size: int = None) -> 'Dataset':
        """DNARecords Tensorflow reader in a sample wise fashion.

        Specially intended for Deep Learning models.

        :return: a Tensorflow dataset with the sample wise DNARecords genomics data.
        :param num_parallel_reads: tf.data.TFRecordDataset equivalent parameter.
        :param num_parallel_calls: tf.data.TFRecordDataset equivalent parameter.
        :param deterministic: tf.data.TFRecordDataset equivalent parameter.
        :param drop_remainder: tf.data.TFRecordDataset equivalent parameter.
        :param batch_size: tf.data.TFRecordDataset equivalent parameter.
        :param buffer_size: tf.data.TFRecordDataset equivalent parameter.
        :rtype: tf.data.Dataset.
        """
        dnarecords = self.datafiles()['swrec']
        schema = self.metadata()['swrsc']
        if schema is None or not dnarecords:
            raise Exception(f"No DNARecords found at {self._dnarecords_path}/...")
        decoder = self._sw_decoder(dnarecords, schema, self._gzip)
        return self._dataset(dnarecords, decoder, num_parallel_reads, num_parallel_calls, deterministic, drop_remainder,
                             batch_size, buffer_size)

    # pylint: disable=too-many-arguments
    # It is reasonable in this case.
    def variant_wise_dataset(self, num_parallel_reads: int = -1, num_parallel_calls: int = -1,
                             deterministic: bool = False, drop_remainder: bool = False, batch_size: int = None,
                             buffer_size: int = None) -> 'Dataset':
        """DNARecords Tensorflow reader in a variant wise fashion.

        Specially intended for GWAS analysis.

        :param num_parallel_reads: tf.data.TFRecordDataset equivalent parameter.
        :param num_parallel_calls: tf.data.TFRecordDataset equivalent parameter.
        :param deterministic: tf.data.TFRecordDataset equivalent parameter.
        :param drop_remainder: tf.data.TFRecordDataset equivalent parameter.
        :param batch_size: tf.data.TFRecordDataset equivalent parameter.
        :param buffer_size: tf.data.TFRecordDataset equivalent parameter.
        :return: a Tensorflow dataset with the variant wise DNARecords genomics data.
        :rtype: tf.data.Dataset.
        """
        dnarecords = self.datafiles()['vwrec']
        schema = self.metadata()['vwrsc']
        if schema is None or not dnarecords:
            raise Exception(f"No DNARecords found at {self._dnarecords_path}/...")
        decoder = self._vw_decoder(dnarecords, schema, self._gzip)
        return self._dataset(dnarecords, decoder, num_parallel_reads, num_parallel_calls, deterministic, drop_remainder,
                             batch_size, buffer_size)


class DNASparkReader:
    """DNARecords Spark reader. **Sample and variant wise**.

    Provides methods to read a previously created DNARecords dataset as pyspark DataFrame objects.

    Review :obj:`.DNARecordsUtils.dnarecords_tree` to know the structure of a previously created DNARecords dataset.

    For each of these directories (when they exist, depending on the configuration used at
    :obj:`.DNARecordsWriter.write`), a spark DataFrame can be returned.


    Example
    --------

    .. code-block:: python

        import dnarecords as dr


        output = '/tmp/dnarecords/output'
        reader = dr.reader.DNASparkReader(output)

        reader.sample_wise_dnarecords().show(2)

        reader.variant_wise_dnarecords().show(2)

    .. code-block:: text

        +---+--------------------+--------------------+----------------+
        |key|        chr1_indices|         chr1_values|chr1_dense_shape| ...
        +---+--------------------+--------------------+----------------+
        | 26|[0, 2, 4, 5, 6, 7...|[0.33607214002352...|             909| ...
        | 29|[0, 1, 2, 3, 4, 5...|[0.20076008098505...|             909| ...
        +---+--------------------+--------------------+----------------+
        only showing top 1 row

        +--------------------+--------------------+----+-----------+
        |             indices|              values| key|dense_shape|
        +--------------------+--------------------+----+-----------+
        |[0, 1, 2, 3, 4, 5...|[0.9984177, 0.007...|3506|      10880|
        |[0, 1, 2, 3, 4, 5...|[0.11181577, 0.01...|3764|      10880|
        +--------------------+--------------------+----+-----------+
        only showing top 2 rows

        ...

    :rtype: Dict[str, DataFrame]
    :param dnarecords_path: path to the DNARecords root directory.
    :return: A dictionary with DataFrame values corresponding to the generated outputs.

    See Also
    --------
    :obj:`.DNARecordsUtils.dnarecords_tree`
    """

    from typing import TYPE_CHECKING
    from typing import Dict

    if TYPE_CHECKING:
        from pyspark.sql import DataFrame

    def __init__(self, dnarecords_path):
        self._dnarecords_path = dnarecords_path

    @staticmethod
    def _spark_safe_load(reader, path):
        try:
            return reader.load(path)
        except Exception as ex:
            raise Exception(f"No DNARecords found at {path}/...") from ex

    def metadata(self) -> Dict[str, 'DataFrame']:
        """Gets the metadata associated to the DNARecords dataset as a dictionary of names to pyspark.sql DataFrames.

        :rtype: Dict[str, DataFrame].
        :return: the metadata associated to the DNARecords as a dictionary of names to pyspark.sql DataFrames.

        See Also
        --------
        :obj:`.DNARecordsUtils.dnarecords_tree` to know the available keys
        """
        import dnarecords as dr

        result = {}
        tree = dr.helper.DNARecordsUtils.dnarecords_tree(self._dnarecords_path)
        spark = dr.helper.DNARecordsUtils.spark_session()
        for k, v in tree.items():
            if k in ['skeys', 'vkeys', 'swpfs', 'vwpfs', 'swrfs', 'vwrfs']:
                result.update({k: self._spark_safe_load(spark.read.format("parquet"), v)})
            if k in ['swpsc', 'vwpsc', 'swrsc', 'vwrsc']:
                result.update({k: self._spark_safe_load(spark.read.format("json"), v)})
        return result

    def sample_wise_dnarecords(self) -> 'DataFrame':
        """Gets a pyspark Dataframe from sample wise DNARecords (when created as tfrecords).

        :rtype: DataFrame.
        :return: a pyspark Dataframe from sample wise DNARecords.
        """
        import dnarecords as dr

        spark = dr.helper.DNARecordsUtils.spark_session()
        path = dr.helper.DNARecordsUtils.dnarecords_tree(self._dnarecords_path)['swrec']
        return self._spark_safe_load(spark.read.format("tfrecord").option("recordType", "Example"), path)

    def variant_wise_dnarecords(self) -> 'DataFrame':
        """Gets a pyspark Dataframe from variant wise DNARecords (when created as tfrecords).

        :rtype: DataFrame.
        :return: a pyspark Dataframe from variant wise DNARecords.
        """
        import dnarecords as dr

        spark = dr.helper.DNARecordsUtils.spark_session()
        path = dr.helper.DNARecordsUtils.dnarecords_tree(self._dnarecords_path)['vwrec']
        return self._spark_safe_load(spark.read.format("tfrecord").option("recordType", "Example"), path)

    def sample_wise_dnaparquet(self) -> 'DataFrame':
        """Gets a pyspark Dataframe from sample wise DNARecords (when created as parquet).

        :rtype: DataFrame.
        :return: a pyspark Dataframe from sample wise DNARecords.
        """
        import dnarecords as dr

        spark = dr.helper.DNARecordsUtils.spark_session()
        path = dr.helper.DNARecordsUtils.dnarecords_tree(self._dnarecords_path)['swpar']
        return self._spark_safe_load(spark.read.format("parquet"), path)

    def variant_wise_dnaparquet(self) -> 'DataFrame':
        """Gets a pyspark Dataframe from variant wise DNARecords (when created as parquet).

        :rtype: DataFrame.
        :return: a pyspark Dataframe from variant wise DNARecords.
        """
        import dnarecords as dr

        spark = dr.helper.DNARecordsUtils.spark_session()
        path = dr.helper.DNARecordsUtils.dnarecords_tree(self._dnarecords_path)['vwpar']
        return self._spark_safe_load(spark.read.format("parquet"), path)
