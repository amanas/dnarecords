# Changelog

<!--next-version-placeholder-->

## v0.2.7 (2022-06-27)
### Performance
* Metadata reader implemented with parallel read ([`da4754f`](https://github.com/amanas/dnarecords/commit/da4754f61719269e1f84f27836029b5d7f3154a2))

## v0.2.6 (2022-05-27)
### Fix
* Fixed dense_shape error on variant_wise writer ([`523a8e6`](https://github.com/amanas/dnarecords/commit/523a8e6cce8779f9363e09533b1d3281481e1599))

## v0.2.5 (2022-05-22)
### Performance
* Use metadata taste when building DNARecordsReader decoder ([`94785a9`](https://github.com/amanas/dnarecords/commit/94785a92cc014b68952dfbea8754e52236163fbe))

## v0.2.4 (2022-05-21)
### Fix
* Added integer to spark<->tf types_dict ([`98e8a2f`](https://github.com/amanas/dnarecords/commit/98e8a2f1adee8aebe47f847db29f093b5115607f))

## v0.2.3 (2022-05-21)
### Fix
* Stop generating property names with chrchr ([`31a9172`](https://github.com/amanas/dnarecords/commit/31a917204422c54a9219f040f9a4f8d7546de66b))

## v0.2.2 (2022-05-20)
### Fix
* Fixed scalability issues related with too many small files generated ([`b4ba095`](https://github.com/amanas/dnarecords/commit/b4ba09581e803f3117f5dba4ec5751ee939c03eb))

### Performance
* Added metadata taste functionality to the reader. Improved performace of block transposition ([`26b2353`](https://github.com/amanas/dnarecords/commit/26b2353c28fc5642c84002002c51d0363ec84c85))

## v0.2.1 (2022-05-19)
### Fix
* Static block size rather than dynamic ([`6b81af0`](https://github.com/amanas/dnarecords/commit/6b81af01443d62417f9894a418b2562f89859068))

## v0.2.0 (2022-05-19)
### Feature
* First release of dnarecords! ([`b12fa05`](https://github.com/amanas/dnarecords/commit/b12fa055859b1f92bddd2cf6707c50bbcf1d8593))
* First release of dnarecords! ([`688e028`](https://github.com/amanas/dnarecords/commit/688e028d62706cb32ab6563531bc36bfe791a381))

### Fix
* Get rid of pypi's 'Filename has been previously used' ([`fe83729`](https://github.com/amanas/dnarecords/commit/fe8372949a8181e8cb75f89536e09696f312c9f9))
* Fixed bug related to save variant keys as pandas dataframe ([`5ce97bb`](https://github.com/amanas/dnarecords/commit/5ce97bb05939b8ca8f48c208e277412544c12d0e))

### Documentation
* Improved documentation for dev/local environments ([`28679f0`](https://github.com/amanas/dnarecords/commit/28679f05940a79687bc01431019e3dbb8c8e4240))

## v0.1.2 (2022-05-19)
### Fix
* Fixed bug related to save variant keys as pandas dataframe ([`5ce97bb`](https://github.com/amanas/dnarecords/commit/5ce97bb05939b8ca8f48c208e277412544c12d0e))

## v0.1.1 (2022-05-13)
### Documentation
* Improved documentation for dev/local environments ([`28679f0`](https://github.com/amanas/dnarecords/commit/28679f05940a79687bc01431019e3dbb8c8e4240))

## v0.1.0 (2022-05-12)
### Feature
* First release of dnarecords! ([`b12fa05`](https://github.com/amanas/dnarecords/commit/b12fa055859b1f92bddd2cf6707c50bbcf1d8593))
