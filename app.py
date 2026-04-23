import streamlit as st
import os
import json
from openai import OpenAI

from src.config.env_setup import init_environment, render_sidebar, check_keys, compute_cost
from src.tools.registry import TOOL_SCHEMAS, execute_tool
from src.ui.stream import process_stream

# Setup env variables via .env
init_environment()

st.set_page_config(page_title="Multi-Function AI Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Function AI Assistant")
st.markdown("Features integrated: **Weather, Calculator, Web Search, Function Calling, Streaming, & Usage Tracking**.")

openai_api_key, serper_api_key = render_sidebar()
check_keys(openai_api_key)

# Initialize environment for the tools
os.environ["SERPER_API_KEY"] = serper_api_key

# We will use OpenAI. Default model is GPT-5 nano for low cost.
model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")

# Pricing defaults are for GPT-5 nano and can be overridden via environment variables.
input_cost_per_1m = float(os.getenv("OPENAI_INPUT_COST_PER_1M", "0.05"))
output_cost_per_1m = float(os.getenv("OPENAI_OUTPUT_COST_PER_1M", "0.40"))

# Pass api_key explicitly, letting the SDK handle its default base_url
client = OpenAI(
    api_key=openai_api_key
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful multi-function assistant. Use tools when needed. If the user asks for weather, calculate something, or search the web, use the available tools."}
    ]

if "token_usage" not in st.session_state:
    st.session_state.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

st.sidebar.subheader("Session Token Usage")
st.sidebar.write(f"**Model:** {model_name}")
st.sidebar.write(f"**Prompt Tokens:** {st.session_state.token_usage['prompt_tokens']}")
st.sidebar.write(f"**Completion Tokens:** {st.session_state.token_usage['completion_tokens']}")
st.sidebar.write(f"**Total Tokens:** {st.session_state.token_usage['total_tokens']}")

# Cost estimation
cost = compute_cost(
    prompt_tokens=st.session_state.token_usage['prompt_tokens'],
    completion_tokens=st.session_state.token_usage['completion_tokens'],
    input_cost_per_1m=input_cost_per_1m,
    output_cost_per_1m=output_cost_per_1m
)
st.sidebar.write(f"**Estimated Cost:** ${cost:.6f}")
st.sidebar.caption(
    f"Pricing: input ${input_cost_per_1m}/1M, output ${output_cost_per_1m}/1M"
)
st.sidebar.caption("Tip: use OPENAI_API_KEY environment variable to avoid typing it.")

for msg in st.session_state.messages:
    if msg["role"] not in ["system", "tool"]:
        with st.chat_message(msg["role"]):
            if msg.get("content"):
                st.markdown(msg["content"])
            elif msg.get("tool_calls"):
                st.status(f"🛠️ Executed {len(msg['tool_calls'])} tools...")

if prompt := st.chat_input("Ask something (e.g. 'What's the weather in Tokyo?' or 'Search for the latest AI news')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # First API call
            placeholder = st.empty()
            stream = client.chat.completions.create(
                model=model_name,
                messages=st.session_state.messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                stream=True,
                stream_options={"include_usage": True}
            )
            
            assistant_msg = process_stream(stream, placeholder)
            st.session_state.messages.append(assistant_msg)
            
            # Check if there are tool calls
            if assistant_msg.get("tool_calls"):
                for tool_call in assistant_msg["tool_calls"]:
                    fn_name = tool_call["function"]["name"]
                    args_str = tool_call["function"]["arguments"]
                    with st.status(f"🛠️ Executing {fn_name}({args_str})...", expanded=True):
                        try:
                            args = json.loads(args_str)
                            result = execute_tool(fn_name, args)
                            st.write(f"Result: {result}")
                        except Exception as e:
                            result = str(e)
                            st.error(result)
                            
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": fn_name,
                            "content": result
                        })
                
                # Second API call to formulate the final answer based on tools
                placeholder_final = st.empty()
                with st.spinner("Generating final response..."):
                    second_stream = client.chat.completions.create(
                        model=model_name,
                        messages=st.session_state.messages,
                        stream=True,
                        stream_options={"include_usage": True}
                    )
                    final_msg = process_stream(second_stream, placeholder_final)
                    st.session_state.messages.append(final_msg)
                    
            st.rerun()

        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "Missing Authentication header" in err_msg:
                st.error(
                    "Authentication failed (401). Set a valid OpenAI API key in OPENAI_API_KEY environment variable or paste it in the sidebar."
                )
            else:
                st.error(f"Error during API call: {err_msg}")
