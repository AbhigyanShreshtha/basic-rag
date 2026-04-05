from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from app.core.models import ChatTurn


class SessionService:
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, deque[ChatTurn]] = defaultdict(
            lambda: deque(maxlen=max_turns * 2)
        )
        self._lock = Lock()

    def resolve_session_id(self, session_id: str | None) -> str:
        return session_id or str(uuid4())

    def get_history(self, session_id: str) -> list[ChatTurn]:
        with self._lock:
            return list(self._sessions.get(session_id, deque()))

    def append_exchange(self, session_id: str, question: str, answer: str) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            session = self._sessions[session_id]
            session.append(ChatTurn(role="user", content=question, timestamp=now))
            session.append(ChatTurn(role="assistant", content=answer, timestamp=now))
