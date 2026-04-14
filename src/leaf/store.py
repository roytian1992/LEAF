from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .records import (
    MemoryAtomRecord,
    MemoryEventRecord,
    MemoryEvidenceLinkRecord,
    MemoryObjectRecord,
    MemoryObjectVersionRecord,
    MemorySnapshotRecord,
)


def _json_loads(payload: str | None, default):
    if payload in (None, ""):
        return default
    return json.loads(payload)


class SQLiteMemoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            create table if not exists leaf_events (
              event_id text primary key,
              corpus_id text not null,
              session_id text not null,
              speaker text not null,
              text text not null,
              turn_index integer not null,
              timestamp text,
              raw_span_id text,
              entity_refs_json text not null default '[]',
              canonical_entity_refs_json text not null default '[]',
              atom_ids_json text not null default '[]',
              metadata_json text not null default '{}',
              embedding_json text
            );
            create table if not exists leaf_atoms (
              atom_id text primary key,
              event_id text not null,
              corpus_id text not null,
              span_id text not null,
              atom_type text not null,
              content text not null,
              entities_json text not null default '[]',
              canonical_entities_json text not null default '[]',
              support_span_ids_json text not null default '[]',
              derived_from_atom_ids_json text not null default '[]',
              memory_kind text not null default 'event',
              status text not null default 'active',
              time_range text,
              confidence real not null,
              metadata_json text not null default '{}'
            );
            create table if not exists leaf_objects (
              object_id text primary key,
              corpus_id text not null,
              subject text not null,
              slot text not null,
              memory_kind text not null,
              policy text not null,
              latest_version_id text,
              status text not null default 'active',
              aliases_json text not null default '[]',
              canonical_entities_json text not null default '[]',
              created_at_event_id text,
              updated_at_event_id text,
              metadata_json text not null default '{}'
            );
            create table if not exists leaf_object_versions (
              version_id text primary key,
              object_id text not null,
              corpus_id text not null,
              value text not null,
              normalized_value text not null,
              summary text not null,
              operation text not null,
              status text not null,
              confidence real not null,
              valid_from text,
              valid_to text,
              event_id text,
              atom_id text,
              metadata_json text not null default '{}'
            );
            create table if not exists leaf_evidence_links (
              link_id text primary key,
              corpus_id text not null,
              object_id text not null,
              version_id text,
              event_id text not null,
              span_id text not null,
              atom_id text,
              role text not null default 'support',
              metadata_json text not null default '{}'
            );
            create table if not exists leaf_snapshots (
              snapshot_id text primary key,
              corpus_id text not null,
              parent_id text,
              snapshot_kind text not null,
              scope_id text not null,
              title text not null,
              synopsis text not null,
              summary text not null,
              object_ids_json text not null default '[]',
              event_ids_json text not null default '[]',
              raw_refs_json text not null default '[]',
              child_ids_json text not null default '[]',
              entity_refs_json text not null default '[]',
              canonical_entity_refs_json text not null default '[]',
              time_range text,
              metadata_json text not null default '{}',
              embedding_json text
            );
            create index if not exists idx_leaf_events_corpus_session on leaf_events(corpus_id, session_id, turn_index);
            create index if not exists idx_leaf_events_corpus_entity on leaf_events(corpus_id, session_id);
            create index if not exists idx_leaf_atoms_event on leaf_atoms(event_id);
            create index if not exists idx_leaf_objects_subject_slot on leaf_objects(corpus_id, subject, slot);
            create index if not exists idx_leaf_versions_object on leaf_object_versions(object_id, status);
            create index if not exists idx_leaf_links_object on leaf_evidence_links(object_id, event_id);
            create index if not exists idx_leaf_links_event on leaf_evidence_links(corpus_id, event_id);
            create index if not exists idx_leaf_snapshots_kind on leaf_snapshots(corpus_id, snapshot_kind, scope_id);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def add_event(self, event: MemoryEventRecord) -> None:
        self.conn.execute(
            """
            insert or replace into leaf_events
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.corpus_id,
                event.session_id,
                event.speaker,
                event.text,
                event.turn_index,
                event.timestamp,
                event.raw_span_id,
                json.dumps(event.entity_refs, ensure_ascii=False),
                json.dumps(event.canonical_entity_refs, ensure_ascii=False),
                json.dumps(event.atom_ids, ensure_ascii=False),
                json.dumps(event.metadata, ensure_ascii=False),
                json.dumps(event.embedding, ensure_ascii=False) if event.embedding is not None else None,
            ),
        )

    def add_atom(self, atom: MemoryAtomRecord) -> None:
        self.conn.execute(
            """
            insert or replace into leaf_atoms
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                atom.atom_id,
                atom.event_id,
                atom.corpus_id,
                atom.span_id,
                atom.atom_type,
                atom.content,
                json.dumps(atom.entities, ensure_ascii=False),
                json.dumps(atom.canonical_entities, ensure_ascii=False),
                json.dumps(atom.support_span_ids, ensure_ascii=False),
                json.dumps(atom.derived_from_atom_ids, ensure_ascii=False),
                atom.memory_kind,
                atom.status,
                atom.time_range,
                atom.confidence,
                json.dumps(atom.metadata, ensure_ascii=False),
            ),
        )

    def upsert_object(self, obj: MemoryObjectRecord) -> None:
        self.conn.execute(
            """
            insert or replace into leaf_objects
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                obj.object_id,
                obj.corpus_id,
                obj.subject,
                obj.slot,
                obj.memory_kind,
                obj.policy,
                obj.latest_version_id,
                obj.status,
                json.dumps(obj.aliases, ensure_ascii=False),
                json.dumps(obj.canonical_entities, ensure_ascii=False),
                obj.created_at_event_id,
                obj.updated_at_event_id,
                json.dumps(obj.metadata, ensure_ascii=False),
            ),
        )

    def add_version(self, version: MemoryObjectVersionRecord) -> None:
        self.conn.execute(
            """
            insert or replace into leaf_object_versions
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version.version_id,
                version.object_id,
                version.corpus_id,
                version.value,
                version.normalized_value,
                version.summary,
                version.operation,
                version.status,
                version.confidence,
                version.valid_from,
                version.valid_to,
                version.event_id,
                version.atom_id,
                json.dumps(version.metadata, ensure_ascii=False),
            ),
        )

    def update_version_window(self, version_id: str, status: str, valid_to: str | None) -> None:
        self.conn.execute(
            """
            update leaf_object_versions
            set status = ?, valid_to = ?
            where version_id = ?
            """,
            (status, valid_to, version_id),
        )

    def add_evidence_link(self, link: MemoryEvidenceLinkRecord) -> None:
        self.conn.execute(
            """
            insert or replace into leaf_evidence_links
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                link.link_id,
                link.corpus_id,
                link.object_id,
                link.version_id,
                link.event_id,
                link.span_id,
                link.atom_id,
                link.role,
                json.dumps(link.metadata, ensure_ascii=False),
            ),
        )

    def upsert_snapshot(self, snapshot: MemorySnapshotRecord) -> None:
        self.conn.execute(
            """
            insert or replace into leaf_snapshots
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.snapshot_id,
                snapshot.corpus_id,
                snapshot.parent_id,
                snapshot.snapshot_kind,
                snapshot.scope_id,
                snapshot.title,
                snapshot.synopsis,
                snapshot.summary,
                json.dumps(snapshot.object_ids, ensure_ascii=False),
                json.dumps(snapshot.event_ids, ensure_ascii=False),
                json.dumps(snapshot.raw_refs, ensure_ascii=False),
                json.dumps(snapshot.child_ids, ensure_ascii=False),
                json.dumps(snapshot.entity_refs, ensure_ascii=False),
                json.dumps(snapshot.canonical_entity_refs, ensure_ascii=False),
                snapshot.time_range,
                json.dumps(snapshot.metadata, ensure_ascii=False),
                json.dumps(snapshot.embedding, ensure_ascii=False) if snapshot.embedding is not None else None,
            ),
        )

    def get_next_turn_index(self, corpus_id: str, session_id: str) -> int:
        row = self.conn.execute(
            """
            select coalesce(max(turn_index), -1) as max_turn_index
            from leaf_events
            where corpus_id = ? and session_id = ?
            """,
            (corpus_id, session_id),
        ).fetchone()
        return int(row["max_turn_index"]) + 1 if row else 0

    def get_event(self, event_id: str) -> MemoryEventRecord | None:
        row = self.conn.execute(
            "select * from leaf_events where event_id = ?",
            (event_id,),
        ).fetchone()
        return self._row_to_event(row) if row else None

    def get_events(
        self,
        corpus_id: str,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEventRecord]:
        query = "select * from leaf_events where corpus_id = ?"
        params: list[object] = [corpus_id]
        if session_id is not None:
            query += " and session_id = ?"
            params.append(session_id)
        query += " order by session_id, turn_index"
        if limit is not None:
            query += " limit ?"
            params.append(limit)
        rows = self.conn.execute(query, tuple(params)).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_events_for_entity(self, corpus_id: str, entity: str, limit: int | None = None) -> list[MemoryEventRecord]:
        pattern = json.dumps(entity, ensure_ascii=False)
        query = """
            select distinct e.*
            from leaf_events e
            left join leaf_evidence_links l on l.event_id = e.event_id and l.corpus_id = e.corpus_id
            left join leaf_objects o on o.object_id = l.object_id
            where e.corpus_id = ?
              and (
                e.canonical_entity_refs_json like '%' || ? || '%'
                or o.subject = ?
              )
            order by e.session_id, e.turn_index desc
        """
        params: list[object] = [corpus_id, pattern, entity]
        if limit is not None:
            query += " limit ?"
            params.append(limit)
        rows = self.conn.execute(query, tuple(params)).fetchall()
        events = [self._row_to_event(row) for row in rows]
        events.reverse()
        return events

    def get_atoms_for_event(self, event_id: str) -> list[MemoryAtomRecord]:
        rows = self.conn.execute(
            "select * from leaf_atoms where event_id = ? order by atom_id",
            (event_id,),
        ).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def get_atoms_for_events(self, event_ids: list[str]) -> list[MemoryAtomRecord]:
        normalized_ids = [str(event_id).strip() for event_id in event_ids if str(event_id).strip()]
        if not normalized_ids:
            return []
        placeholders = ", ".join("?" for _ in normalized_ids)
        rows = self.conn.execute(
            f"""
            select *
            from leaf_atoms
            where event_id in ({placeholders})
            order by event_id, atom_id
            """,
            tuple(normalized_ids),
        ).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def get_snapshot(self, corpus_id: str, snapshot_kind: str, scope_id: str) -> MemorySnapshotRecord | None:
        row = self.conn.execute(
            """
            select * from leaf_snapshots
            where corpus_id = ? and snapshot_kind = ? and scope_id = ?
            """,
            (corpus_id, snapshot_kind, scope_id),
        ).fetchone()
        return self._row_to_snapshot(row) if row else None

    def list_snapshots(self, corpus_id: str, snapshot_kind: str) -> list[MemorySnapshotRecord]:
        rows = self.conn.execute(
            """
            select * from leaf_snapshots
            where corpus_id = ? and snapshot_kind = ?
            order by scope_id
            """,
            (corpus_id, snapshot_kind),
        ).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def get_object(self, object_id: str) -> MemoryObjectRecord | None:
        row = self.conn.execute(
            "select * from leaf_objects where object_id = ?",
            (object_id,),
        ).fetchone()
        return self._row_to_object(row) if row else None

    def get_object_versions(self, object_id: str) -> list[MemoryObjectVersionRecord]:
        rows = self.conn.execute(
            """
            select * from leaf_object_versions
            where object_id = ?
            order by coalesce(valid_from, ''), version_id
            """,
            (object_id,),
        ).fetchall()
        return [self._row_to_version(row) for row in rows]

    def get_latest_version(self, object_id: str) -> MemoryObjectVersionRecord | None:
        row = self.conn.execute(
            """
            select v.*
            from leaf_object_versions v
            join leaf_objects o on o.latest_version_id = v.version_id
            where o.object_id = ?
            """,
            (object_id,),
        ).fetchone()
        return self._row_to_version(row) if row else None

    def get_active_versions_for_subject(self, corpus_id: str, subject: str) -> list[MemoryObjectVersionRecord]:
        rows = self.conn.execute(
            """
            select v.*
            from leaf_object_versions v
            join leaf_objects o on o.object_id = v.object_id
            where o.corpus_id = ? and o.subject = ? and v.status = 'active'
            order by o.slot, v.valid_from, v.version_id
            """,
            (corpus_id, subject),
        ).fetchall()
        return [self._row_to_version(row) for row in rows]

    def get_objects_for_subject(self, corpus_id: str, subject: str) -> list[MemoryObjectRecord]:
        rows = self.conn.execute(
            """
            select * from leaf_objects
            where corpus_id = ? and subject = ?
            order by slot, object_id
            """,
            (corpus_id, subject),
        ).fetchall()
        return [self._row_to_object(row) for row in rows]

    def get_objects_for_session(self, corpus_id: str, session_id: str) -> list[MemoryObjectRecord]:
        rows = self.conn.execute(
            """
            select distinct o.*
            from leaf_objects o
            join leaf_evidence_links l on l.object_id = o.object_id and l.corpus_id = o.corpus_id
            join leaf_events e on e.event_id = l.event_id and e.corpus_id = l.corpus_id
            where o.corpus_id = ? and e.session_id = ?
            order by o.subject, o.slot, o.object_id
            """,
            (corpus_id, session_id),
        ).fetchall()
        return [self._row_to_object(row) for row in rows]

    def list_subjects(self, corpus_id: str) -> list[str]:
        rows = self.conn.execute(
            """
            select distinct subject
            from leaf_objects
            where corpus_id = ?
            order by subject
            """,
            (corpus_id,),
        ).fetchall()
        return [str(row["subject"]) for row in rows]

    def list_session_ids(self, corpus_id: str) -> list[str]:
        rows = self.conn.execute(
            """
            select distinct session_id
            from leaf_events
            where corpus_id = ?
            order by session_id
            """,
            (corpus_id,),
        ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def list_corpora(self) -> list[str]:
        rows = self.conn.execute(
            """
            select corpus_id from leaf_snapshots
            union
            select corpus_id from leaf_events
            order by corpus_id
            """
        ).fetchall()
        return [str(row["corpus_id"]) for row in rows if row["corpus_id"] is not None]

    def get_corpus_stats(self, corpus_id: str) -> dict[str, Any]:
        def scalar(query: str, *params: object) -> int:
            row = self.conn.execute(query, params).fetchone()
            return int(row[0]) if row and row[0] is not None else 0

        snapshot_rows = self.conn.execute(
            """
            select snapshot_kind, count(*) as count
            from leaf_snapshots
            where corpus_id = ?
            group by snapshot_kind
            order by snapshot_kind
            """,
            (corpus_id,),
        ).fetchall()
        snapshot_counts_by_kind = {str(row["snapshot_kind"]): int(row["count"]) for row in snapshot_rows}

        object_rows = self.conn.execute(
            """
            select memory_kind, count(*) as count
            from leaf_objects
            where corpus_id = ?
            group by memory_kind
            order by memory_kind
            """,
            (corpus_id,),
        ).fetchall()
        object_counts_by_memory_kind = {str(row["memory_kind"]): int(row["count"]) for row in object_rows}

        version_rows = self.conn.execute(
            """
            select status, count(*) as count
            from leaf_object_versions
            where corpus_id = ?
            group by status
            order by status
            """,
            (corpus_id,),
        ).fetchall()
        version_counts_by_status = {str(row["status"]): int(row["count"]) for row in version_rows}

        db_path = Path(self.db_path)
        return {
            "events": scalar("select count(*) from leaf_events where corpus_id = ?", corpus_id),
            "atoms": scalar("select count(*) from leaf_atoms where corpus_id = ?", corpus_id),
            "objects": scalar("select count(*) from leaf_objects where corpus_id = ?", corpus_id),
            "versions": scalar("select count(*) from leaf_object_versions where corpus_id = ?", corpus_id),
            "evidence_links": scalar("select count(*) from leaf_evidence_links where corpus_id = ?", corpus_id),
            "snapshots": scalar("select count(*) from leaf_snapshots where corpus_id = ?", corpus_id),
            "sessions": scalar("select count(distinct session_id) from leaf_events where corpus_id = ?", corpus_id),
            "subjects": scalar("select count(distinct subject) from leaf_objects where corpus_id = ?", corpus_id),
            "snapshot_counts_by_kind": snapshot_counts_by_kind,
            "object_counts_by_memory_kind": object_counts_by_memory_kind,
            "version_counts_by_status": version_counts_by_status,
            "db_file_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
        }

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> MemoryEventRecord:
        return MemoryEventRecord(
            event_id=str(row["event_id"]),
            corpus_id=str(row["corpus_id"]),
            session_id=str(row["session_id"]),
            speaker=str(row["speaker"]),
            text=str(row["text"]),
            turn_index=int(row["turn_index"]),
            timestamp=row["timestamp"],
            raw_span_id=row["raw_span_id"],
            entity_refs=list(_json_loads(row["entity_refs_json"], [])),
            canonical_entity_refs=list(_json_loads(row["canonical_entity_refs_json"], [])),
            atom_ids=list(_json_loads(row["atom_ids_json"], [])),
            metadata=dict(_json_loads(row["metadata_json"], {})),
            embedding=list(_json_loads(row["embedding_json"], [])) if row["embedding_json"] else None,
        )

    @staticmethod
    def _row_to_atom(row: sqlite3.Row) -> MemoryAtomRecord:
        return MemoryAtomRecord(
            atom_id=str(row["atom_id"]),
            event_id=str(row["event_id"]),
            corpus_id=str(row["corpus_id"]),
            span_id=str(row["span_id"]),
            atom_type=str(row["atom_type"]),
            content=str(row["content"]),
            entities=list(_json_loads(row["entities_json"], [])),
            canonical_entities=list(_json_loads(row["canonical_entities_json"], [])),
            support_span_ids=list(_json_loads(row["support_span_ids_json"], [])),
            derived_from_atom_ids=list(_json_loads(row["derived_from_atom_ids_json"], [])),
            memory_kind=str(row["memory_kind"]),
            status=str(row["status"]),
            time_range=row["time_range"],
            confidence=float(row["confidence"]),
            metadata=dict(_json_loads(row["metadata_json"], {})),
        )

    @staticmethod
    def _row_to_object(row: sqlite3.Row) -> MemoryObjectRecord:
        return MemoryObjectRecord(
            object_id=str(row["object_id"]),
            corpus_id=str(row["corpus_id"]),
            subject=str(row["subject"]),
            slot=str(row["slot"]),
            memory_kind=str(row["memory_kind"]),
            policy=str(row["policy"]),
            latest_version_id=row["latest_version_id"],
            status=str(row["status"]),
            aliases=list(_json_loads(row["aliases_json"], [])),
            canonical_entities=list(_json_loads(row["canonical_entities_json"], [])),
            created_at_event_id=row["created_at_event_id"],
            updated_at_event_id=row["updated_at_event_id"],
            metadata=dict(_json_loads(row["metadata_json"], {})),
        )

    @staticmethod
    def _row_to_version(row: sqlite3.Row) -> MemoryObjectVersionRecord:
        return MemoryObjectVersionRecord(
            version_id=str(row["version_id"]),
            object_id=str(row["object_id"]),
            corpus_id=str(row["corpus_id"]),
            value=str(row["value"]),
            normalized_value=str(row["normalized_value"]),
            summary=str(row["summary"]),
            operation=str(row["operation"]),
            status=str(row["status"]),
            confidence=float(row["confidence"]),
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            event_id=row["event_id"],
            atom_id=row["atom_id"],
            metadata=dict(_json_loads(row["metadata_json"], {})),
        )

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> MemorySnapshotRecord:
        return MemorySnapshotRecord(
            snapshot_id=str(row["snapshot_id"]),
            corpus_id=str(row["corpus_id"]),
            parent_id=row["parent_id"],
            snapshot_kind=str(row["snapshot_kind"]),
            scope_id=str(row["scope_id"]),
            title=str(row["title"]),
            synopsis=str(row["synopsis"]),
            summary=str(row["summary"]),
            object_ids=list(_json_loads(row["object_ids_json"], [])),
            event_ids=list(_json_loads(row["event_ids_json"], [])),
            raw_refs=list(_json_loads(row["raw_refs_json"], [])),
            child_ids=list(_json_loads(row["child_ids_json"], [])),
            entity_refs=list(_json_loads(row["entity_refs_json"], [])),
            canonical_entity_refs=list(_json_loads(row["canonical_entity_refs_json"], [])),
            time_range=row["time_range"],
            metadata=dict(_json_loads(row["metadata_json"], {})),
            embedding=list(_json_loads(row["embedding_json"], [])) if row["embedding_json"] else None,
        )
