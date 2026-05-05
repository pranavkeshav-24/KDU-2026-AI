from __future__ import annotations

from dataclasses import dataclass, field

from ..agents import BillingAgent, ConsensusAgent, DBAgent, TriageAgent, VectorAgent
from ..audio import AudioCoordinator, AudioEvent, LocalWhisperTranscriptionService, StreamingTTSRenderer
from ..core import (
    EventType,
    HandoffPayload,
    Message,
    SessionState,
    Transcript,
    utc_now,
)
from ..core import SystemConfig
from ..infrastructure import ConcurrencyQueue, EventLog, monitored_agent_call, prune_history
from .coordinator import Coordinator


@dataclass
class Orchestrator:
    config: SystemConfig = field(default_factory=SystemConfig)
    event_log: EventLog | None = None
    audio: AudioCoordinator = field(default_factory=AudioCoordinator)
    transcription: LocalWhisperTranscriptionService = field(default_factory=LocalWhisperTranscriptionService)
    triage_agent: TriageAgent = field(default_factory=TriageAgent)
    billing_agent: BillingAgent = field(default_factory=BillingAgent)
    vector_agent: VectorAgent = field(default_factory=VectorAgent)
    consensus_agent: ConsensusAgent = field(default_factory=ConsensusAgent)

    def __post_init__(self) -> None:
        if self.event_log is None:
            self.event_log = EventLog(self.config.event_log_path)
        queue = ConcurrencyQueue(self.config.max_db_concurrent, self.config.max_db_queue_depth)
        self.db_agent = DBAgent(queue=queue)
        self.coordinator = Coordinator(
            db_agent=self.db_agent,
            vector_agent=self.vector_agent,
            consensus_agent=self.consensus_agent,
            event_log=self.event_log,
            per_task_timeout_seconds=self.config.agent_timeout_seconds,
        )
        self.tts = StreamingTTSRenderer(self.audio)
        self.sessions: dict[str, SessionState] = {}

    def session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id)
        return self.sessions[session_id]

    async def handle_audio_event(self, session_id: str, event: AudioEvent) -> str | None:
        if event.event_type is EventType.INTERRUPTION_DETECTED:
            self.handle_interruption(session_id)
            return None
        if event.event_type is EventType.UTTERANCE_READY:
            return await self.process_utterance(session_id, await self.transcription.transcribe(event.frames))
        return None

    def handle_interruption(self, session_id: str) -> None:
        state = self.session(session_id)
        self.audio.signal_interruption()
        self.tts.stop()
        self.event_log.append(
            EventType.INTERRUPTION_DETECTED,
            session_id,
            {
                "active_response_text": state.active_response_text,
                "spoken_response_text": state.spoken_response_text,
            },
        )

    async def process_utterance(self, session_id: str, transcript: Transcript | str) -> str:
        state = self.session(session_id)
        if isinstance(transcript, str):
            transcript = Transcript(text=transcript, confidence=1.0)

        self.event_log.append(
            EventType.UTTERANCE_READY,
            session_id,
            {"transcript": transcript.text, "confidence": transcript.confidence},
        )
        state.append("user", transcript.text)

        triage_history = prune_history(
            state.conversation_history,
            self.config.triage_history_token_budget,
            "Older conversation turns were pruned deterministically by sliding window.",
        )
        triage_result = await monitored_agent_call(
            agent_name="Triage Agent",
            session_id=session_id,
            payload={"transcript": transcript.text, "conversation_history": triage_history},
            event_log=self.event_log,
            call=lambda: self.triage_agent.classify(transcript.text, session_id),
        )

        billing_history = prune_history(
            state.conversation_history,
            self.config.billing_history_token_budget,
            "Older conversation turns were pruned deterministically by sliding window.",
        )
        handoff = HandoffPayload(
            session_id=session_id,
            conversation_history=billing_history,
            classified_intent=triage_result.intent,
            entity_context=triage_result.entity_context,
            triage_confidence=triage_result.confidence,
            timestamp_utc=utc_now(),
            call_reason_summary=triage_result.call_reason_summary,
        )
        self.event_log.append(EventType.HANDOFF_COMPLETED, session_id, {"handoff": handoff})

        consensus = await self.coordinator.run(handoff)
        response_text = await monitored_agent_call(
            agent_name="Billing Agent",
            session_id=session_id,
            payload={"handoff": handoff, "consensus": consensus},
            event_log=self.event_log,
            call=lambda: self.billing_agent.answer(handoff, consensus),
        )
        state.active_response_text = response_text
        state.spoken_response_text = ""
        self.audio.clear_interruption()
        playback = await self.tts.speak(response_text)
        state.spoken_response_text = playback.spoken_text

        if playback.completed:
            state.append("assistant", response_text)
            self.event_log.append(
                EventType.RESPONSE_SPOKEN,
                session_id,
                {"response_text": response_text, "spoken_text": playback.spoken_text},
            )
        else:
            self.event_log.append(
                EventType.PLAYBACK_TRUNCATED,
                session_id,
                {
                    "response_text": response_text,
                    "spoken_text": playback.spoken_text,
                    "unspoken_text": playback.unspoken_text,
                },
            )
            self.event_log.append(
                EventType.RESPONSE_TRUNCATED,
                session_id,
                {
                    "spoken_text": playback.spoken_text,
                    "unspoken_text": playback.unspoken_text,
                },
            )
        state.active_response_text = None
        return playback.spoken_text

    async def process_text_turn(self, session_id: str, text: str) -> str:
        return await self.process_utterance(session_id, Transcript(text=text, confidence=1.0))

    def get_history(self, session_id: str) -> list[Message]:
        return list(self.session(session_id).conversation_history)
