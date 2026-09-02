"""Trace bodies on the artifact store, as Parquet.

Postgres holds one pointer row per trace - id, split, content hash - so slicing
and joining stay SQL. The bodies live here because traces are wide, deeply
nested, and repetitive, which is exactly what columnar compression is for.

The nested structure is stored as a JSON string in a `payload` column rather
than as a Parquet schema. A trace has free-form `ground_truth` and `metadata`
whose keys differ per customer, so a real schema would either have to be
inferred per snapshot - making two snapshots of the same product incompatible -
or flattened, which loses the shape. A JSON column keeps `trace_id`, `split`,
and `content_hash` genuinely columnar for filtering while the body round-trips
exactly.

Two fields are excluded from the payload.

`ingested_at`, because every trace in a snapshot is ingested at the same moment
- the authoritative time is `snapshot.created_at`. Keeping a per-trace copy made
the serialized bytes differ on every run, which defeated content addressing
entirely: re-ingesting unchanged data wrote a second Parquet file identical but
for a timestamp.

`content_hash`, because it is derived. Storing it in its own column and
recomputing it on read turns the round trip into an integrity check - a payload
that no longer hashes to its recorded value is corrupt, and saying so beats
handing back a trace that quietly is not the one that was ingested.
"""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from evalloop.contracts.trace import Trace
from evalloop.store.artifacts import LocalArtifactStore

__all__ = ["SCHEMA", "read_traces", "write_traces"]

_DERIVED = {"ingested_at", "content_hash"}

SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("content_hash", pa.string(), nullable=False),
        pa.field("payload", pa.string(), nullable=False),
    ]
)


def write_traces(
    store: LocalArtifactStore,
    traces: Sequence[Trace],
    *,
    split: str = "train",
) -> str:
    """Write traces to the artifact store, returning their `cas://` URI.

    Content-addressed, so the same traces written twice occupy one file - which
    is what makes a re-ingest cheap as well as idempotent.
    """
    table = pa.Table.from_pydict(
        {
            "trace_id": [t.trace_id for t in traces],
            "split": [split] * len(traces),
            "content_hash": [t.content_hash for t in traces],
            "payload": [t.model_dump_json(exclude=_DERIVED) for t in traces],
        },
        schema=SCHEMA,
    )

    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression="zstd")
    return store.put_bytes(buffer.getvalue().to_pybytes())


def read_traces(store: LocalArtifactStore, uri: str, *, verify: bool = True) -> list[Trace]:
    """Read traces back, checking each against its recorded content hash.

    Lossless except for `ingested_at`, which is not stored per trace - callers
    that need it read `snapshot.created_at`.
    """
    table = pq.read_table(store.path_of(uri))
    payloads = table.column("payload").to_pylist()
    recorded = table.column("content_hash").to_pylist()

    traces = [Trace.model_validate_json(payload) for payload in payloads]
    if verify:
        for trace, expected in zip(traces, recorded, strict=True):
            if trace.content_hash != expected:
                raise ValueError(
                    f"trace {trace.trace_id!r} in {uri} does not match its recorded "
                    f"content hash: stored {expected[:12]}…, recomputed "
                    f"{trace.content_hash[:12]}…"
                )
    return traces
