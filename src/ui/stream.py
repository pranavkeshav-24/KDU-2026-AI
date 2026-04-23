import streamlit as st
import json
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

def process_stream(response_stream, placeholder):
    """
    Handles streaming responses from OpenAI, updating the Streamlit placeholder.
    Returns the reconstructed assistant message dict (with content and/or tool_calls).
    """
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
    
    # Reconstruct the message object expected by the conversation history
    assistant_msg = {"role": "assistant"}
    if full_response:
        assistant_msg["content"] = full_response
        
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
        
    return assistant_msg