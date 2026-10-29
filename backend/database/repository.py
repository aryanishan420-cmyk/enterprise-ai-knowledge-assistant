"""Data access layer. SQLite by default; any SQLAlchemy URL (e.g. PostgreSQL) works."""

import hashlib
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, DocumentRecord, QueryLog


class Repository:
    def __init__(self, database_url: str):
        url = make_url(database_url)
        if url.get_backend_name() == "sqlite" and url.database not in (None, ":memory:"):
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"connect_args": {"check_same_thread": False}} if url.get_backend_name() == "sqlite" else {}
        self.engine = create_engine(database_url, **kwargs)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(self.engine, expire_on_commit=False)

    def add_document(self, **fields) -> None:
        with self._Session.begin() as s:
            s.add(DocumentRecord(**fields))

    def delete_document(self, document_id: str) -> bool:
        with self._Session.begin() as s:
            rec = s.get(DocumentRecord, document_id)
            if rec is None:
                return False
            s.delete(rec)
            return True

    def list_documents(self, owner: str | None = None) -> list[DocumentRecord]:
        with self._Session() as s:
            q = select(DocumentRecord).order_by(DocumentRecord.created_at)
            if owner:
                q = q.where(DocumentRecord.owner == owner)
            return list(s.scalars(q))

    def log_query(self, question: str, mode: str, hits: int, insufficient: bool, latency_ms: float) -> None:
        with self._Session.begin() as s:
            s.add(
                QueryLog(
                    question_sha256=hashlib.sha256(question.encode()).hexdigest(),
                    question_chars=len(question),
                    mode=mode,
                    hits=hits,
                    insufficient=int(insufficient),
                    latency_ms=latency_ms,
                )
            )

    def query_count(self) -> int:
        with self._Session() as s:
            return s.scalar(select(func.count()).select_from(QueryLog)) or 0
