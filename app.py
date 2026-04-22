import streamlit as st
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tools import TOOL_SCHEMAS, execute_tool

load_dotenv()

st.set_page_config(page_title="Multi-Function AI Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Function AI Assistant")
st.markdown("Features integrated: **Weather, Calculator, Web Search, Function Calling, Streaming, & Usage Tracking**.")

st.sidebar.header("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key (or OpenRouter)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
serper_api_key = st.sidebar.text_input("Serper API Key", type="password", value=os.getenv("SERPER_API_KEY", ""))

if not openai_api_key:
    st.warning("Please provide an OpenAI API Key to start.")
    st.stop()

# Initialize environment for the tools
os.environ["SERPER_API_KEY"] = serper_api_key

# We will use OpenRouter or OpenAI. We'll default to OpenAI, using gpt-4o-mini
base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(
    api_key=openai_api_key,
    base_url=base_url
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful multi-function assistant. Use tools when needed. If the user asks for weather, calculate something, or search the web, use the available tools."}
    ]

if "token_usage" not in st.session_state:
    st.session_state.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

st.sidebar.subheader("Session Token Usage")
st.sidebar.write(f"**Prompt Tokens:** {st.session_state.token_usage['prompt_tokens']}")
st.sidebar.write(f"**Completion Tokens:** {st.session_state.token_usage['completion_tokens']}")
st.sidebar.write(f"**Total Tokens:** {st.session_state.token_usage['total_tokens']}")

# Cost estimation for gpt-4o-mini is ~$0.150 / 1M input tokens and $0.600 / 1M output tokens
cost = (st.session_state.token_usage['prompt_tokens'] / 1000000) * 0.15 + (st.session_state.token_usage['completion_tokens'] / 1000000) * 0.60
st.sidebar.write(f"**Estimated Cost:** ${cost:.6f}")

for msg in st.session_state.messages:
    if msg["role"] not in ["system", "tool"]:
        with st.chat_message(msg["role"]):
            if msg.get("content"):
                st.markdown(msg["content"])
            elif msg.get("tool_calls"):
                st.status(f"🛠️ Executed {len(msg['tool_calls'])} tools...")

def process_stream(response_stream, placeholder):
    """ Handle streaming response from OpenAI """
    full_response = ""
    tool_calls = []
    
    # We iterate over chunks
    for chunk in response_stream:
        # Accumulate usage statistics if available
        if hasattr(chunk, 'usage') and chunk.usage:
            st.session_state.token_usage['prompt_tokens'] += getattr(chunk.usage, 'prompt_tokens', 0)
            st.session_state.token_usage['completion_tokens'] += getattr(chunk.usage, 'completion_tokens', 0)
            st.session_state.token_usage['total_tokens'] += getattr(chunk.usage, 'total_tokens', 0)
            
        if not chunk.choices:
            continue
            
        delta = chunk.choices[0].delta
        if not delta:
            continue
            
        # Accumulate text content
        if delta.content:
            full_response += delta.content
            placeholder.markdown(full_response + "▌")
            
        # Accumulate tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                # Ensure the list is long enough
                while len(tool_calls) <= tc.index:
                    tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                
                # Append the fields
                if tc.id:
                    tool_calls[tc.index]["id"] = tc.id
                if tc.function and tc.function.name:
                    tool_calls[tc.index]["function"]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

    if full_response:
        placeholder.markdown(full_response)
    else:
        placeholder.empty()
    
    # Reconstruct the message
    assistant_msg = {"role": "assistant"}
    if full_response:
        assistant_msg["content"] = full_response
        
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
        
    return assistant_msg

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
            st.error(f"Error during API call: {str(e)}")
