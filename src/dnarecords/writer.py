"""
DNARecords available writers.
"""


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
# It is reasonable in this case.
class DNARecordsWriter:
    """Genomics data (vcf, bgen, etc.) to tfrecords or parquet. **Sample and variant wise**.

    Core class to go from genomics data to tfrecords or parquet files ready to use with
    Deep Learning frameworks like Tensorflow or Pytorch.

    * Able to generate **dna records variant wise or sample wise (i.e. transposing the matrix)**.
    * **Takes advantage of sparsity** (very convenient to save space and computation, specially with Deep Learning).
    * **Scales automatically to any sized dataset. Tested on UKBB**.

    Example
    --------

    .. code-block:: python

        import dnarecords as dr


        hl = dr.helper.DNARecordsUtils.init_hail()
        hl.utils.get_1kg('/tmp/1kg')
        mt = hl.read_matrix_table('/tmp/1kg/1kg.mt')
        mt = mt.annotate_entries(dosage=hl.pl_dosage(mt.PL))

        output = '/tmp/dnarecords/output'
        writer = dr.writer.DNARecordsWriter(mt.dosage)
        writer.write(output, sparse=True, sample_wise=True, variant_wise=True,
                     tfrecord_format=True, parquet_format=True,
                     write_mode='overwrite', gzip=True)

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

    :param expr: a Hail expression. Currently, ony expressions coercible to numeric are supported
    :param block_size: entries per block in the internal operations
    :param staging: path to staging directory to use for intermediate data. Default: /tmp/dnarecords/staging.
    """
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from hail import MatrixTable, Expression
        from pyspark.sql import DataFrame

    _i_blocks: set
    _j_blocks: set
    _nrows: int
    _ncols: int
    _sparsity: float
    _chrom_ranges: dict
    _mt: 'MatrixTable'
    _skeys: 'DataFrame'
    _vkeys: 'DataFrame'

    def __init__(self, expr: 'Expression', block_size: int = int(1e6), staging: str = '/tmp/dnarecords/staging'):
        self._assert_expr_type(expr)
        self._expr = expr
        self._block_size = block_size
        self._kv_blocks_path = f'{staging}/kv-blocks'
        self._vw_dna_staging = f'{staging}/vw-dnaparquet'
        self._sw_dna_staging = f'{staging}/sw-dnaparquet'

    @staticmethod
    def _assert_expr_type(expr):
        import hail.expr.expressions.expression_typecheck as tc
        msg = 'Only expr_numeric are supported at this time.'
        msg += '\nConsider redefining the entry_expr as an expr_numeric.'
        msg += '\nFeel free to use the helper functions on this module.'
        if not tc.expr_numeric.can_coerce(expr.dtype):
            raise Exception(msg)

    def _set_mt(self):
        from hail.expr.expressions.expression_utils import matrix_table_source
        self._mt = matrix_table_source('dnarecords', self._expr).annotate_entries(v=self._expr)

    def _index_mt(self):
        self._mt = self._mt.add_row_index('i')
        self._mt = self._mt.key_rows_by('i')
        self._mt = self._mt.add_col_index('j')
        self._mt = self._mt.key_cols_by('j')

    def _set_vkeys_skeys(self):
        self._vkeys = self._mt.key_rows_by().rows().to_spark().withColumnRenamed('i', 'key').cache()
        self._skeys = self._mt.key_cols_by().cols().to_spark().withColumnRenamed('j', 'key').cache()

    def _set_chrom_ranges(self):
        from pyspark.sql import functions as F
        gdf = self._vkeys.select('`locus.contig`', 'key').groupby('`locus.contig`')
        gdf = gdf.agg(F.min('key').alias('start'), F.max('key').alias('end'))
        self._chrom_ranges = {r['locus.contig'].replace('chr', ''): [r['start'], r['end']] for i, r in
                              gdf.toPandas().iterrows()}

    def _update_vkeys_by_chrom_ranges(self):
        from dnarecords.helper import DNARecordsUtils
        import pyspark.sql.functions as F
        chr_start_keys = [[k, start] for k, [start, _] in self._chrom_ranges.items()]
        spark = DNARecordsUtils.spark_session()
        chr_start_keys_df = spark.sparkContext.parallelize(chr_start_keys).toDF(['locus.contig', 'chr_start_key'])
        self._vkeys = self._vkeys \
            .join(chr_start_keys_df, on='locus.contig', how='left') \
            .withColumn('chr_key', F.col('key') - F.col('chr_start_key'))

    def _select_ijv(self):
        self._mt = self._mt.select_globals().select_rows().select_cols().select_entries('v')

    def _filter_out_undefined_entries(self):
        from dnarecords.helper import DNARecordsUtils
        hl = DNARecordsUtils.init_hail()
        self._mt = self._mt.filter_entries(hl.is_defined(self._mt.v))

    def _filter_out_zeroes(self):
        from dnarecords.helper import DNARecordsUtils
        hl = DNARecordsUtils.init_hail()
        self._mt = self._mt.filter_entries(0 != hl.coalesce(self._mt.v, 0))

    def _set_max_nrows_ncols(self):
        self._nrows = self._mt.count_rows()
        self._ncols = self._mt.count_cols()

    def _set_sparsity(self):
        mts = self._mt.head(10000, None)
        entries = mts.key_cols_by().key_rows_by().entries().to_spark().filter('v is not null').count()
        self._sparsity = entries / (mts.count_rows() * mts.count_cols())

    def _get_block_size(self):
        import math
        M, N, S = self._nrows + 1, self._ncols + 1, self._sparsity + 1e-6
        B = self._block_size / S
        m = math.ceil(math.sqrt(B * M / N))
        n = math.ceil(math.sqrt(B * N / M))
        return m, n

    def _build_ij_blocks(self):
        import pyspark.sql.functions as F
        m, n = self._get_block_size()
        df = self._mt.key_cols_by().key_rows_by().entries().to_spark().filter('v is not null')
        df = df.withColumn('ib', F.expr(f"i div {m}"))
        df = df.withColumn('jb', F.expr(f"j div {n}"))
        df.repartition('ib', 'jb').write.partitionBy('ib', 'jb').mode('overwrite').parquet(self._kv_blocks_path)

    def _set_ij_blocks(self):
        import re
        from dnarecords.helper import DNARecordsUtils
        hl = DNARecordsUtils.init_hail()
        all_blocks = [p for p in hl.hadoop_ls(f'{self._kv_blocks_path}/*') if p['is_dir']]
        self._i_blocks = {re.search(r'ib=(\d+)', p['path']).group(1) for p in all_blocks}
        self._j_blocks = {re.search(r'jb=(\d+)', p['path']).group(1) for p in all_blocks}

    @staticmethod
    def _dnarecord_variant_agg(rows, mkey):
        for row in rows:
            sd = dict(sorted(row['data'].items()))
            yield {'key': row['i'], 'indices': list(sd.keys()), 'values': list(sd.values()),
                   'dense_shape': mkey + 1}

    @staticmethod
    def _dnarecord_sample_agg(rows, chrom_ranges):
        import numpy as np
        for row in rows:
            sd = dict(sorted(row['data'].items()))
            keys = np.array(list(sd.keys()))
            vals = np.array(list(sd.values()))
            result = {'key': row['j']}
            for locus, [start, end] in chrom_ranges.items():
                mask = (start <= keys) & (keys <= end)
                result.update({f'chr{locus}_indices': (keys[mask] - start).tolist(),
                               f'chr{locus}_values': vals[mask].tolist(),
                               f'chr{locus}_dense_shape': end - start + 1})
            yield result

    @staticmethod
    def _to_dnarecord_variant_wise(mkey):
        return lambda rows: DNARecordsWriter._dnarecord_variant_agg(rows, mkey)

    @staticmethod
    def _to_dnarecord_sample_wise(chrom_ranges):
        return lambda rows: DNARecordsWriter._dnarecord_sample_agg(rows, chrom_ranges)

    def _build_dna_block_variant_wise(self, blocks_path, output):
        from dnarecords.helper import DNARecordsUtils
        import pyspark.sql.functions as F

        def get_dna_schema(gdf):
            import pyspark.sql.types as pytypes

            data_type = [s for s in gdf.schema if s.name == 'data'][0].dataType
            values_type = data_type.valueType
            return pytypes.StructType([pytypes.StructField("key", pytypes.LongType(), False),
                                       pytypes.StructField("indices", pytypes.ArrayType(pytypes.LongType(), False),
                                                           False),
                                       pytypes.StructField("values", pytypes.ArrayType(values_type, False), False),
                                       pytypes.StructField("dense_shape", pytypes.LongType(), False)])

        spark = DNARecordsUtils.spark_session()
        df = spark.read.parquet(blocks_path).select('i', 'j', 'v')
        df = df.groupBy('i').agg(F.map_from_entries(F.collect_list(F.struct('j', 'v'))).alias('data'))
        schema = get_dna_schema(df)
        mapper = self._to_dnarecord_variant_wise(self._nrows)
        df.rdd.mapPartitions(mapper).toDF(schema).repartition(1).write.mode('overwrite').parquet(output)

    def _build_dna_block_sample_wise(self, blocks_path, output, chrom_ranges):
        from dnarecords.helper import DNARecordsUtils
        import pyspark.sql.functions as F

        def get_dna_schema(gdf):
            import pyspark.sql.types as pytypes

            data_type = [s for s in gdf.schema if s.name == 'data'][0].dataType
            values_type = data_type.valueType
            gdf_schema = pytypes.StructType([pytypes.StructField("key", pytypes.LongType(), False)])
            for k in chrom_ranges.keys():
                gdf_schema.add(f"chr{k}_indices", pytypes.ArrayType(pytypes.LongType(), False), False)
                gdf_schema.add(f"chr{k}_values", pytypes.ArrayType(values_type, False), False)
                gdf_schema.add(f"chr{k}_dense_shape", pytypes.LongType(), False)
            return gdf_schema

        spark = DNARecordsUtils.spark_session()
        df = spark.read.parquet(blocks_path).select('i', 'j', 'v')
        df = df.groupBy('j').agg(F.map_from_entries(F.collect_list(F.struct('i', 'v'))).alias('data'))
        schema = get_dna_schema(df)
        mapper = self._to_dnarecord_sample_wise(chrom_ranges)
        df.rdd.mapPartitions(mapper).toDF(schema).repartition(1).write.mode('overwrite').parquet(output)

    def _build_dna_blocks(self, by):
        from multiprocessing.pool import ThreadPool
        import multiprocessing as ms

        pool = ThreadPool(ms.cpu_count() - 1)
        if by == 'i':
            params = [
                [f'{self._kv_blocks_path}/ib={ib}/jb=*', f'{self._vw_dna_staging}/{ib}-of-{len(self._i_blocks):04}']
                for ib in self._i_blocks]
            pool.starmap(self._build_dna_block_variant_wise, params)
        if by == 'j':
            params = [
                [f'{self._kv_blocks_path}/ib=*/jb={jb}', f'{self._sw_dna_staging}/{jb}-of-{len(self._j_blocks):04}',
                 self._chrom_ranges]
                for jb in self._j_blocks]
            pool.starmap(self._build_dna_block_sample_wise, params)
        pool.close()
        pool.join()

    # pylint: disable=too-many-arguments
    # It is reasonable in this case.
    @staticmethod
    def _write_dnarecords(output, output_schema, dna_blocks, write_mode, gzip, tfrecord_format):
        from dnarecords.helper import DNARecordsUtils

        spark = DNARecordsUtils.spark_session()
        df = spark.read.parquet(dna_blocks)
        df_writer = df.write.mode(write_mode)
        if tfrecord_format:
            df_writer = df_writer.format("tfrecord").option("recordType", "Example")
            if gzip:
                # Needs huge overhead memory
                df_writer = df_writer.option("codec", "org.apache.hadoop.io.compress.GzipCodec")
        else:
            df_writer = df_writer.format('parquet')
            if gzip:
                df_writer = df_writer.option("compression", "gzip")
        df_writer.save(output)
        sc_writer = spark.read.json(spark.sparkContext.parallelize([df.schema.json()])).coalesce(1).write
        sc_writer.mode(write_mode).format('json').save(output_schema)

    @staticmethod
    def _write_key_files(source, output, tfrecord_format, write_mode):
        from dnarecords.helper import DNARecordsUtils
        import pyspark.sql.functions as F

        spark = DNARecordsUtils.spark_session()
        if tfrecord_format:
            reader = spark.read.format("tfrecord").option("recordType", "Example")
        else:
            reader = spark.read.format("parquet")
        df = reader.load(source).withColumn("path", F.regexp_extract(F.input_file_name(), f"(.*){source}/(.*)", 2))
        df.select('key', 'path').write.mode(write_mode).parquet(output)

    # pylint: disable=too-many-arguments
    # It is reasonable in this case.
    def write(self, output: str, sparse: bool = True, sample_wise: bool = True, variant_wise: bool = False,
              tfrecord_format: bool = True, parquet_format: bool = False, write_mode: str = 'error',
              gzip: bool = True) -> None:
        """DNARecords spark writer.

        Writes a DNARecords dataset based on the Hail `expr` provided in the class constructor.

        :rtype: Dict[str, DataFrame]
        :param output: path to the output location of the DNARecords.
        :param sparse: generate sparse data (filtering out any zero values). Default: True.
        :param sample_wise: generate DNARecords in a sample wise fashion (i.e. transposing the matrix, one column -> one record). Default: True.
        :param variant_wise: generate DNARecords in a variant wise fashion (i.e. one row -> one record) Default: False.
        :param tfrecord_format: generate tfrecords output files. Default: True.
        :param parquet_format: generate parquet output files. Default: False.
        :param write_mode: spark write mode parameter ('error', 'overwrite', etc.). Default: 'error'.
        :param gzip: gzip the output files. Default: True.
        :return: A dictionary with DataFrames for each generated output.

        See Also
        --------
        :obj:`.DNARecordsUtils.dnarecords_tree`, :obj:`.DNASparkReader`
        """
        from dnarecords.helper import DNARecordsUtils

        if not sample_wise and not variant_wise:
            raise Exception('At least one of sample_wise, variant_wise must be True')
        if not tfrecord_format and not parquet_format:
            raise Exception('At least one of tfrecord_format, parquet_format must be True')

        otree = DNARecordsUtils.dnarecords_tree(output)

        self._set_mt()
        self._index_mt()
        self._set_vkeys_skeys()
        self._set_chrom_ranges()
        self._update_vkeys_by_chrom_ranges()

        self._vkeys.write.mode(write_mode).parquet(otree['vkeys'])
        self._skeys.write.mode(write_mode).parquet(otree['skeys'])

        self._select_ijv()
        self._filter_out_undefined_entries()
        if sparse:
            self._filter_out_zeroes()
        self._set_max_nrows_ncols()
        self._set_sparsity()
        self._build_ij_blocks()
        self._set_ij_blocks()

        if variant_wise:
            self._build_dna_blocks('i')

        if sample_wise:
            self._build_dna_blocks('j')

        if variant_wise:
            if tfrecord_format:
                self._write_dnarecords(otree['vwrec'], otree['vwrsc'], f'{self._vw_dna_staging}/*', write_mode, gzip,
                                       True)
                self._write_key_files(otree['vwrec'], otree['vwrfs'], True, write_mode)
            if parquet_format:
                self._write_dnarecords(otree['vwpar'], otree['vwpsc'], f'{self._vw_dna_staging}/*', write_mode, gzip,
                                       False)
                self._write_key_files(otree['vwpar'], otree['vwpfs'], False, write_mode)

        if sample_wise:
            if tfrecord_format:
                self._write_dnarecords(otree['swrec'], otree['swrsc'], f'{self._sw_dna_staging}/*', write_mode,
                                       gzip, True)
                self._write_key_files(otree['swrec'], otree['swrfs'], True, write_mode)
            if parquet_format:
                self._write_dnarecords(otree['swpar'], otree['swpsc'], f'{self._sw_dna_staging}/*', write_mode,
                                       gzip, False)
                self._write_key_files(otree['swpar'], otree['swpfs'], False, write_mode)
