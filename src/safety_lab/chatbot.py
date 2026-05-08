from __future__ import annotations

from time import perf_counter

from safety_lab.cloud_safety import CloudSafetyClient
from safety_lab.guardrails import LocalGuardrailEngine, asks_for_ssn_last4
from safety_lab.llm import CustomerServiceLLM
from safety_lab.mock_backend import get_customer_record
from safety_lab.models import ChatTurnResult, CloudSafetyResult, GuardrailAction, GuardrailDecision
from safety_lab.observability import TraceRecorder


class CustomerServiceBot:
    def __init__(
        self,
        *,
        llm: CustomerServiceLLM,
        guardrails: LocalGuardrailEngine | None,
        cloud_safety: CloudSafetyClient | None,
        tracer: TraceRecorder,
        user_id: str = "cust_001",
    ) -> None:
        self.llm = llm
        self.guardrails = guardrails
        self.cloud_safety = cloud_safety
        self.tracer = tracer
        self.user_id = user_id

    def chat(self, user_input: str) -> ChatTurnResult:
        self.tracer.spans = []
        total_start = perf_counter()
        guardrail_events: list[GuardrailDecision] = []
        cloud_events: list[CloudSafetyResult] = []
        raw_output = ""
        final_output = ""
        blocked = False

        with self.tracer.span(
            "chat_turn",
            run_type="chain",
            inputs={"user_input": user_input},
            metadata={"user_id": self.user_id},
        ) as root_outputs:
            if self.guardrails:
                with self.tracer.span("input_prompt_injection_guardrail", inputs={"text": user_input}) as outputs:
                    decision = self.guardrails.inspect_input(user_input)
                    guardrail_events.append(decision)
                    outputs.update({"action": decision.action.value, "rules": decision.triggered_rules})
                if decision.action == GuardrailAction.BLOCK:
                    blocked = True
                    final_output = decision.text
                    root_outputs["final_output"] = final_output
                    return self._finish(user_input, raw_output, final_output, blocked, guardrail_events, cloud_events, total_start)

            if self.cloud_safety:
                with self.tracer.span("cloud_safety_input", inputs={"text": user_input}) as outputs:
                    cloud_result = self.cloud_safety.analyze(user_input, source="input")
                    cloud_events.append(cloud_result)
                    outputs.update({"blocked": cloud_result.blocked, "categories": cloud_result.categories})
                if cloud_result.blocked:
                    blocked = True
                    final_output = "I cannot process that request because it violates the configured content policy."
                    root_outputs["final_output"] = final_output
                    return self._finish(user_input, raw_output, final_output, blocked, guardrail_events, cloud_events, total_start)

            with self.tracer.span("mock_backend_customer_lookup", inputs={"user_id": self.user_id}) as outputs:
                customer = get_customer_record(self.user_id)
                outputs["record_fields"] = ["name", "email", "ssn"]

            if self.guardrails and asks_for_ssn_last4(user_input):
                raw_output = f"The last 4 digits of your SSN are {customer.ssn_last4}."
                final_output = raw_output
                root_outputs["raw_model_output"] = raw_output
                root_outputs["final_output"] = final_output
                return self._finish(user_input, raw_output, final_output, blocked, guardrail_events, cloud_events, total_start)

            with self.tracer.span("llm_generation", run_type="llm", inputs={"user_input": user_input}) as outputs:
                raw_output = self.llm.generate(user_input=user_input, customer=customer)
                outputs["raw_model_output"] = raw_output

            final_output = raw_output
            if self.guardrails:
                with self.tracer.span("output_pii_guardrail", inputs={"raw_model_output": raw_output}) as outputs:
                    output_decision = self.guardrails.inspect_output(raw_output)
                    guardrail_events.append(output_decision)
                    final_output = output_decision.text
                    outputs.update(
                        {
                            "action": output_decision.action.value,
                            "rules": output_decision.triggered_rules,
                            "modified_output": final_output,
                        }
                    )

            if self.cloud_safety:
                with self.tracer.span("cloud_safety_output", inputs={"text": final_output}) as outputs:
                    cloud_result = self.cloud_safety.analyze(final_output, source="output")
                    cloud_events.append(cloud_result)
                    outputs.update({"blocked": cloud_result.blocked, "categories": cloud_result.categories})
                if cloud_result.blocked:
                    blocked = True
                    final_output = "The response was blocked by the configured Bedrock Guardrails policy."

            root_outputs["raw_model_output"] = raw_output
            root_outputs["final_output"] = final_output
            return self._finish(user_input, raw_output, final_output, blocked, guardrail_events, cloud_events, total_start)

    def _finish(
        self,
        user_input: str,
        raw_output: str,
        final_output: str,
        blocked: bool,
        guardrail_events: list[GuardrailDecision],
        cloud_events: list[CloudSafetyResult],
        total_start: float,
    ) -> ChatTurnResult:
        total_latency_ms = (perf_counter() - total_start) * 1000
        guardrail_triggered = any(event.triggered for event in guardrail_events)
        failed = blocked
        self.tracer.flush(
            guardrail_triggered=guardrail_triggered,
            failed=failed,
            total_latency_ms=total_latency_ms,
        )
        return ChatTurnResult(
            user_input=user_input,
            raw_model_output=raw_output,
            final_output=final_output,
            blocked=blocked,
            guardrail_events=guardrail_events,
            cloud_events=cloud_events,
            spans=self.tracer.spans,
            total_latency_ms=total_latency_ms,
        )
