import json
import sys
import os

# --- CONFIGURATION ---
# usage: python validate_batch.py path/to/your_file.jsonl


FILE_PATH = "/nlp/scr/nmeist/EvalDims/output_data/yin/together/batch_inputs/batch_input_seshadri_zollo_meta_llama_3.1_8b_instruct_turbo.jsonl" 

# ---------------------

def validate_line(line_num, line):
    line = line.strip()
    
    # 1. Check for Empty Lines (Critical Cause of "Silent Failure")
    if not line:
        return f"Line {line_num} is EMPTY. JSONL files cannot have blank lines."

    # 2. Check for Valid JSON
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        return f"Line {line_num} is NOT valid JSON: {e}"

    # 3. Check Required Fields
    required = ["custom_id", "method", "url", "body"]
    missing = [field for field in required if field not in data]
    if missing:
        return f"Line {line_num} missing fields: {missing}"

    # 4. Check Method
    if data.get("method") != "POST":
        return f"Line {line_num} has wrong method: {data.get('method')} (Must be 'POST')"

    # 5. Check URL (MOST COMMON MISTAKE)
    # It must be relative path, NOT full URL
    url = data.get("url", "")
    if url.startswith("http") or "together.xyz" in url:
        return f"Line {line_num} has INVALID URL: '{url}'. Must be relative (e.g., '/v1/chat/completions')"

    # 6. Check Body Structure
    body = data.get("body", {})
    if not isinstance(body, dict):
        return f"Line {line_num} 'body' must be a dictionary object."
    
    if "messages" not in body:
        return f"Line {line_num} body missing 'messages' list."
        
    return None # No errors

def run_validation(file_path):
    print(f"🔍 Validating: {file_path} ...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"FATAL: Could not read file. {e}")
        return

    if not lines:
        print("FATAL: File is EMPTY (0 lines). Batch will fail immediately.")
        return

    error_count = 0
    for i, line in enumerate(lines, 1):
        error = validate_line(i, line)
        if error:
            print(f"❌ {error}")
            error_count += 1
            # Stop after 5 errors to avoid spamming
            if error_count >= 5:
                print("... (Stopping after 5 errors)")
                break

    if error_count == 0:
        print(f"✅ SUCCESS: {len(lines)} lines validated. File is safe to upload.")
    else:
        print(f"\n🚫 FAILED: Found errors in {error_count} lines. Fix these before uploading.")

# Run it
if __name__ == "__main__":
    # Allow running from command line or hardcoded path
    target_file = sys.argv[1] if len(sys.argv) > 1 else FILE_PATH
    run_validation(target_file)