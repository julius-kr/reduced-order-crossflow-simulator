# Frozen legacy solver

`python/` is a byte-for-byte copy of the original top-level `python/` tree at commit
`d69ab5fbcc71103fc770d4ad8818476fc7cf35e2`, written from git blobs by `tools/create_legacy_snapshot.py`.
`manifest.json` records the SHA-256 of every file and the excluded build artefacts (bytecode caches
and egg-info).

**Do not edit anything here.** Known defects are preserved on purpose, and fixes belong in `src/`.
