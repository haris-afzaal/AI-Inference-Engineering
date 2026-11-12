import asyncio
import json

# Mock SSE stream data (Exactly how OpenAI or vLLM sends data over the network)
mock_sse_stream = [
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Artificial "}}]}\n\n',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Intelligence "}}]}\n\n',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Inference "}}]}\n\n',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Engineering."}}]}\n\n',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{}}],"usage":{"prompt_tokens":15,"completion_tokens":4,"total_tokens":19}}\n\n',
    'data: [DONE]\n\n'
]

async def parse_openai_stream():
    print("--- Connecting to OpenAI-Compatible Server ---")
    full_text = ""
    
    print("Streaming Response: ", end="", flush=True)
    
    for chunk in mock_sse_stream:
        # Simulate network latency (delay) between tokens
        await asyncio.sleep(0.4) 
        
        line = chunk.strip()
        if not line:
            continue
            
        # Stop condition
        if line == "data: [DONE]":
            print("\n\n[Connection Closed] Received [DONE] signal.")
            break
            
        # Parse the JSON payload
        if line.startswith("data: "):
            json_str = line[6:] # Remove 'data: ' prefix
            try:
                data = json.loads(json_str)
                
                # 1. Extract text chunks (deltas)
                delta = data["choices"][0].get("delta", {})
                if "content" in delta:
                    content = delta["content"]
                    full_text += content
                    # Print immediately to terminal without newline
                    print(content, end="", flush=True)
                    
                # 2. Extract Token Usage (usually in the last chunk before DONE)
                if "usage" in data and data["usage"] is not None:
                    usage = data["usage"]
                    print(f"\n\n[Billing / Usage Stats] Prompt: {usage['prompt_tokens']} | Completion: {usage['completion_tokens']} | Total: {usage['total_tokens']}")
                    
            except json.JSONDecodeError:
                print(f"\n[Error] Invalid JSON received: {json_str}")

    print(f"\nFinal Assembled Text in Client Memory: '{full_text}'")

if __name__ == "__main__":
    asyncio.run(parse_openai_stream())
