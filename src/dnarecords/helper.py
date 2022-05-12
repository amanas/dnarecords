"""
DNARecords helper utilities.
"""


class DNARecordsUtils:
    """Utility class to provide common functionalities used in other modules.
    """
    from typing import Dict
    from types import ModuleType
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from pyspark.sql import SparkSession

    @staticmethod
    def spark_session() -> 'SparkSession':
        """Gets the current spark session or builds a new one if none.

        Ensures sparktfrecord libraries are available in the session.

        :return: a spark session with sparktfrecord libraries available.
        :rtype: SparkSession
        """
        from pyspark.sql import SparkSession
        return SparkSession.builder \
            .config('spark.jars.packages', 'com.linkedin.sparktfrecord:spark-tfrecord_2.12:0.3.4') \
            .getOrCreate()

    @staticmethod
    def init_hail() -> ModuleType:
        """Initializes Hail ensuring sparktfrecord libraries are available in the session.
        :return: the hail module (with Hail initialized)
        :rtype: ModuleType
        """
        import hail as hl

        conf = {'spark.jars.packages': 'com.linkedin.sparktfrecord:spark-tfrecord_2.12:0.3.4'}
        hl.init(idempotent=True, log='/tmp/hail.log', spark_conf=conf)
        return hl

    @staticmethod
    def dnarecords_tree(dnarecords_path) -> Dict[str, str]:
        """DNARecords directory structure.

        Gets a dictionary with the full structure of a DNARecords dataset given a root path.

        .. code-block:: text

            swrec -> <dnarecords_path>/data/swrec (sample wise dna tfrecords)
            vwrec -> <dnarecords_path>/data/vwrec (variant wise dna tfrecords)
            swpar -> <dnarecords_path>/data/swpar (sample wise dna parquet files)
            vwpar -> <dnarecords_path>/data/vwpar (variant wise dna parquet files)
            skeys -> <dnarecords_path>/meta/skeys (sample wise key mapping)
            vkeys -> <dnarecords_path>/meta/vkeys (variant wise key mapping)
            swpfs -> <dnarecords_path>/meta/swpfs (sample wise parquet files index)
            vwpfs -> <dnarecords_path>/meta/vwpfs (variant wise parquet files index)
            swrfs -> <dnarecords_path>/meta/swrfs (sample wise tfrecords index)
            vwrfs -> <dnarecords_path>/meta/vwrfs (variant wise tfrecords index)
            swpsc -> <dnarecords_path>/meta/swpsc (sample wise parquet schema)
            vwpsc -> <dnarecords_path>/meta/vwpsc (variant wise parquet schema)
            swrsc -> <dnarecords_path>/meta/swrsc (sample wise tfrecord schema)
            vwrsc -> <dnarecords_path>/meta/vwrsc (variant wise tfrecord schema)

        :return: a dictionary with the structure of the DNARecords dataset.
        :rtype: Dict[str,str]
        """
        return {'swrec': f'{dnarecords_path}/data/swrec',
                'vwrec': f'{dnarecords_path}/data/vwrec',
                'swpar': f'{dnarecords_path}/data/swpar',
                'vwpar': f'{dnarecords_path}/data/vwpar',
                'skeys': f'{dnarecords_path}/meta/skeys',
                'vkeys': f'{dnarecords_path}/meta/vkeys',
                'swpfs': f'{dnarecords_path}/meta/swpfs',
                'vwpfs': f'{dnarecords_path}/meta/vwpfs',
                'swrfs': f'{dnarecords_path}/meta/swrfs',
                'vwrfs': f'{dnarecords_path}/meta/vwrfs',
                'swpsc': f'{dnarecords_path}/meta/swpsc',
                'vwpsc': f'{dnarecords_path}/meta/vwpsc',
                'swrsc': f'{dnarecords_path}/meta/swrsc',
                'vwrsc': f'{dnarecords_path}/meta/vwrsc'}
