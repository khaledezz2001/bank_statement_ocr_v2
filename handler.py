import runpod
import json
import base64
import re
import os
import io
import sys
import time
import subprocess
import urllib.request
import torch
from datetime import datetime
from pdf2image import convert_from_bytes
from PIL import Image
from openai import OpenAI

def log(msg):
    print(f"[LOG] {msg}", flush=True)

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/models/qwen3.6-27b"
VLLM_PORT = 8100
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"
MODEL_NAME = "qwen3.6-27b"
MAX_PAGES_PER_BATCH = 4  # Number of pages to process in a single prompt (multi-image)
MAX_NEW_TOKENS = 65536  # Large enough for 50+ transactions per page

# ===============================
# LAUNCH vLLM SERVER
# ===============================
def start_vllm_server():
    """Launch vLLM OpenAI-compatible server as a subprocess and wait until ready."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--served-model-name", MODEL_NAME,
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.95",
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--limit-mm-per-prompt", "image=8",
        "--disable-log-requests",
    ]
    log(f"Starting vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    # Wait for server to be ready (up to 5 minutes)
    max_wait = 300
    waited = 0
    while waited < max_wait:
        try:
            resp = urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health", timeout=2)
            if resp.status == 200:
                log(f"vLLM server is ready after {waited}s")
                return proc
        except Exception:
            pass
        time.sleep(2)
        waited += 2
        if waited % 20 == 0:
            log(f"Still waiting for vLLM server... ({waited}s)")
        # Check if process died
        if proc.poll() is not None:
            log(f"ERROR: vLLM server process exited with code {proc.returncode}")
            raise RuntimeError(f"vLLM server failed to start (exit code {proc.returncode})")

    proc.kill()
    raise RuntimeError(f"vLLM server did not become ready within {max_wait}s")

log("Launching vLLM server for Qwen3.6-27B...")
vllm_process = start_vllm_server()

log("Initializing OpenAI client...")
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="EMPTY",
)
log("OpenAI client ready. vLLM server running at " + VLLM_BASE_URL)

# ===============================
# PROMPT
# ===============================
SYSTEM_PROMPT = """You are a helpful financial assistant.
Your task is to extract all transaction details from the provided bank statement images.
Return ONLY a valid JSON array of objects. Do not include any markdown formatting (like ```json).

CRITICAL — Number formatting:
- ALL numeric values (debit, credit, balance) MUST be plain decimal numbers using a DOT as the decimal separator.
- NEVER use dots or commas as thousands separators.
- NEVER use a comma as a decimal separator.
- Convert any European-style numbers you see in the statement to standard format:
    18.812,76  →  18812.76
    3.884,81   →  3884.81
    300.000,00 →  300000.00
    276.361,88 →  276361.88
    3,25       →  3.25
    150,00     →  150.00
    8.658,25   →  8658.25
- If the value is a whole number, write it as an integer (e.g. 200, not 200.00).
- Numbers in JSON must NOT be quoted — they are raw numeric values.

Output Format Examples:

Example 1 (separate Debits/Credits columns):
[
  {
    "date": "2014-05-15",
    "description": "DIVIDEND",
    "debit": null,
    "credit": 1495.80,
    "balance": 514894.75,
    "currency": "USD"
  }
]

Example 2 (single Amount column with negative values):
[
  {
    "date": "2025-07-01",
    "description": "IBU-Low Activity Fees For June 2025",
    "debit": 23.46,
    "credit": null,
    "balance": 5105.29,
    "currency": "USD"
  }
]

Rules:
1. Extract every single transaction row.
2. If a value is missing, use null.
3. Ensure numbers are plain decimal floats using DOT as decimal separator. No currency symbols, no thousand separators (no dots or commas for grouping). Use absolute values (always positive).
4. CRITICAL DATE RULES — output dates as YYYY-MM-DD:
   a. First, determine the date format used in the statement by examining:
      - The statement header / period line (e.g. "01.01.2025 to 31.12.2025" or "01/01/2025 - 12/31/2025")
      - The language and country of the bank (European/Middle-Eastern banks use DD.MM.YYYY or DD/MM/YYYY; US banks use MM/DD/YYYY)
      - Whether any date in the column has a day value > 12, which proves it is DD-first format
   b. Common formats and how to convert them:
      - DD.MM.YYYY or DD/MM/YYYY → 18.02.2025 means Day=18, Month=02, output 2025-02-18
      - MM/DD/YYYY → 02/18/2025 means Month=02, Day=18, output 2025-02-18
      - DD MMM YYYY or DD-Mon-YYYY → 18 Feb 2025 or 18-Feb-2025, output 2025-02-18
      - MMM DD, YYYY → Feb 18, 2025, output 2025-02-18
      - YYYY-MM-DD → already in correct format, output as-is
   c. When the format uses dots (.) or slashes (/) and ALL values are ≤ 12, default to DD first (day.month.year) unless the bank is clearly US-based.
   d. Read each date digit by digit from the image very carefully. Do not guess or approximate dates.
   e. Use the statement period header to verify that all transaction dates fall within the expected range.
   f. Examples: 18.02.2025 → 2025-02-18 | 24.02.2025 → 2025-02-24 | 26.06.2025 → 2025-06-26 | 02/18/2025 → 2025-02-18 | Jan 5, 2025 → 2025-01-05
5. CAREFULLY check the column headers to determine whether an amount is a debit or credit:
   - If there are separate "Debits" and "Credits" columns, look at which column the number appears under.
   - If there is a single "Amount" column, negative values (with a minus sign) are DEBITS and positive values are CREDITS.
   - Fees, charges, and withdrawals are always DEBITS.
6. "description" should contain the transaction type/name and any meaningful details (including any reference codes, voucher numbers, or transaction IDs found in the row).
7. "currency" is the currency of the account as shown on the statement header or transaction details (e.g. USD, EUR, GBP, SAR, AED, CHF). Detect it from the statement context.
8. Output ONLY these 6 fields per transaction: date, description, debit, credit, balance, currency. Do NOT include any other fields.
"""



def repair_truncated_json(text):
    """Attempt to repair truncated JSON arrays by finding the last complete object.
    
    Strategy: walk backwards through all '}' positions to find the last
    closing brace that, when followed by ']', yields a valid JSON array.
    This handles cases where the truncation happens mid-object.
    """
    start = text.find('[')
    if start == -1:
        return None
    
    # Find all '}' positions and try each from the end
    body = text[start:]
    positions = [i for i, ch in enumerate(body) if ch == '}']
    
    for pos in reversed(positions):
        candidate = body[:pos + 1].rstrip().rstrip(',') + '\n]'
        try:
            data = json.loads(candidate)
            if isinstance(data, list) and len(data) > 0:
                log(f"Repaired truncated JSON: recovered {len(data)} transactions.")
                return data
        except json.JSONDecodeError:
            continue
    
    return None


def image_to_base64_url(img):
    """Convert a PIL Image to a base64 data URL for the OpenAI Vision API."""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"


def process_pages(images):
    """
    Process multiple pages using Qwen3.6-27B via vLLM's OpenAI-compatible API.
    Sends images as base64-encoded data URLs in the chat completions format.
    """
    # Resize images for consistency
    processed_images = []
    for img in images:
        if max(img.size) > 2000:
            img.thumbnail((2000, 2000))
        processed_images.append(img)

    # Build multi-image content array for OpenAI Vision API format
    content = []
    for img in processed_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_base64_url(img)
            }
        })
    content.append({"type": "text", "text": SYSTEM_PROMPT})

    messages = [
        {"role": "user", "content": content}
    ]

    try:
        log(f"Sending {len(processed_images)} images to vLLM for inference...")
        
        # Call vLLM via OpenAI-compatible chat completions API
        # Using non-thinking (instruct) mode for deterministic JSON extraction
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        raw_response = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        usage = response.usage

        log(f"Finish reason: {finish_reason}")
        log(f"Usage: prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}, total_tokens={usage.total_tokens}")
        log(f"Raw output length: {len(raw_response)} chars")
        log(f"=== FULL RAW MODEL OUTPUT START ===")
        log(raw_response)
        log(f"=== FULL RAW MODEL OUTPUT END ===")
        
        return [raw_response]

    except Exception as e:
        log(f"Inference error: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        return [json.dumps({"error": f"Batch failed: {str(e)}"})]


def parse_raw_output(raw_output, batch_idx):
    """Parse raw model output into transaction list.
    
    Returns:
        tuple: (transactions_list, was_truncated)
    """
    was_truncated = False
    try:
        cleaned = raw_output

        # Strip markdown code fences
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # Strip any <think>...</think> blocks that might leak through
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()

        # Try direct JSON parse first
        batch_data = None
        try:
            batch_data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback 1: extract JSON array using regex
            json_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if json_match:
                try:
                    batch_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Fallback 2: repair truncated JSON (model ran out of tokens)
            if batch_data is None:
                log("Direct parse failed. Attempting truncated JSON repair...")
                batch_data = repair_truncated_json(cleaned)
                if batch_data is not None:
                    was_truncated = True

        if batch_data is not None and isinstance(batch_data, list):
            log(f"Batch {batch_idx} parsed successfully: {len(batch_data)} transactions.")
            return batch_data, was_truncated
        elif batch_data is not None:
            log(f"Warning: Batch {batch_idx} returned non-list JSON: {batch_data}")
        else:
            log(f"Failed to parse JSON for batch {batch_idx}. Skipping.")
            log(f"Raw output (first 500 chars): {raw_output[:500]}")
            log(f"Cleaned text (first 300 chars): {cleaned[:300]}")
    except Exception as e:
        log(f"Failed to parse JSON for batch {batch_idx}: {e}. Skipping.")
        log(f"Raw output (first 500 chars): {raw_output[:500]}")
    
    return [], was_truncated


def process_pdf(pdf_bytes):
    # 1. Convert PDF to Images
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        log(f"Converted PDF to {len(images)} images.")
    except Exception as e:
        log(f"Error converting PDF: {e}")
        return json.dumps({"error": f"Failed to convert PDF: {str(e)}"})

    if not images:
        return json.dumps({"error": "No images extracted from PDF"})

    all_transactions = []
    
    # Process in batches — Qwen3.6-27B handles multi-image natively
    for i in range(0, len(images), MAX_PAGES_PER_BATCH):
        batch = images[i:i + MAX_PAGES_PER_BATCH]
        batch_num = i // MAX_PAGES_PER_BATCH + 1
        total_batches = (len(images) + MAX_PAGES_PER_BATCH - 1) // MAX_PAGES_PER_BATCH
        log(f"Processing batch {batch_num}/{total_batches} ({len(batch)} pages as multi-image prompt)...")
        
        # Process all pages in the batch as a single multi-image prompt
        raw_outputs = process_pages(batch)
        
        # Parse the combined output
        batch_truncated = False
        for j, raw_output in enumerate(raw_outputs):
            batch_transactions, was_truncated = parse_raw_output(raw_output, batch_num)
            if was_truncated:
                batch_truncated = True
            all_transactions.extend(batch_transactions)
        
        # If output was truncated and batch had multiple pages, retry each page individually
        if batch_truncated and len(batch) > 1:
            log(f"Batch {batch_num} was truncated with {len(batch)} pages. Retrying each page individually...")
            all_transactions = all_transactions[:-len(batch_transactions)]  # remove partial results
            for page_idx, single_page in enumerate(batch):
                log(f"  Re-processing page {i + page_idx + 1} individually...")
                page_outputs = process_pages([single_page])
                for raw_output in page_outputs:
                    page_txns, _ = parse_raw_output(raw_output, f"{batch_num}-page{page_idx+1}")
                    all_transactions.extend(page_txns)
            
    # Filter out ghost transactions — only remove truly empty records
    final_transactions = []
    for t in all_transactions:
        balance = t.get("balance")
        credit = t.get("credit")
        debit = t.get("debit")
        description = t.get("description", "").strip()
        
        # Only remove if ALL value fields are empty/zero AND no meaningful description
        is_completely_empty = (
            (balance is None or balance == 0 or balance == 0.0)
            and credit is None
            and debit is None
        )
        # Also remove if both debit and credit are explicitly zero
        both_zero = (credit == 0 and debit == 0)
        
        if (is_completely_empty and not description) or both_zero:
            log(f"Filtered ghost transaction: {t}")
            continue
        
        # Keep only the 6 required fields
        cleaned_t = {
            "date": t.get("date", ""),
            "description": t.get("description", ""),
            "debit": t.get("debit"),
            "credit": t.get("credit"),
            "balance": t.get("balance"),
            "currency": t.get("currency", "")
        }
        final_transactions.append(cleaned_t)
    
    # ---- Post-processing: normalize dates ----
    # Step 1: Fix obviously swapped dates (month > 12)
    for t in final_transactions:
        date_str = t.get("date", "")
        if date_str:
            try:
                parts = date_str.split("-")
                if len(parts) == 3:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    if month > 12 and day <= 12:
                        log(f"Fixing swapped date: {date_str} -> {year}-{day:02d}-{month:02d}")
                        t["date"] = f"{year}-{day:02d}-{month:02d}"
            except (ValueError, IndexError):
                pass
    
    # Step 2: Detect the statement's actual date range from the data itself.
    # Collect all valid (year, month) pairs and find the dominant range.
    # Then check if swapping day/month on each date would place it better
    # within the dominant range — this catches subtle misreads where both
    # day and month are <= 12.
    from collections import Counter
    valid_months = []
    for t in final_transactions:
        date_str = t.get("date", "")
        if date_str:
            try:
                parts = date_str.split("-")
                if len(parts) == 3:
                    y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                    if 1 <= m <= 12 and 1 <= d <= 31:
                        valid_months.append(m)
            except (ValueError, IndexError):
                pass
    
    if valid_months:
        month_counts = Counter(valid_months)
        # The most common months represent the real date range
        common_months = {m for m, cnt in month_counts.items() if cnt >= 2}
        
        # Step 3: For dates where month appears only once (singleton month),
        # check if swapping day↔month would produce a month that's in common_months.
        # This fixes cases like 2025-01-18 that should be 2025-02-18 (misread from 18.02).
        for t in final_transactions:
            date_str = t.get("date", "")
            if not date_str:
                continue
            try:
                parts = date_str.split("-")
                if len(parts) != 3:
                    continue
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                
                # Only attempt swap if current month is a singleton AND
                # the swapped version would be valid and in the common set
                if (month_counts.get(m, 0) <= 1 and   # current month is rare
                    d <= 12 and m <= 31 and             # swap would be valid
                    d in common_months and              # swapped month is common
                    d != m):                            # not the same value
                    
                    new_date = f"{y}-{d:02d}-{m:02d}"
                    log(f"Fixing likely misread date: {date_str} -> {new_date} "
                        f"(month {m} is rare, month {d} is common in this statement)")
                    t["date"] = new_date
            except (ValueError, IndexError):
                pass
    
    # Step 4: Log singleton dates for debugging awareness
    date_counts = Counter(t.get("date", "") for t in final_transactions if t.get("date"))
    for t in final_transactions:
        d = t.get("date", "")
        if d and date_counts[d] == 1:
            log(f"Singleton date detected (possible misread): {d} — {t.get('description', '')[:60]}")
    
    # ---- Post-processing: sort by date (chronological) for balance validation ----
    try:
        final_transactions.sort(key=lambda t: t.get("date", "0000-00-00"))
    except Exception:
        pass
    
    # ---- Post-processing: validate debit/credit using balance changes ----
    for i in range(1, len(final_transactions)):
        prev_balance = final_transactions[i - 1].get("balance")
        curr_balance = final_transactions[i].get("balance")
        credit = final_transactions[i].get("credit")
        debit = final_transactions[i].get("debit")
        
        if prev_balance is None or curr_balance is None:
            continue
        
        balance_diff = curr_balance - prev_balance
        
        if balance_diff < 0:
            if credit is not None and debit is None:
                log(f"Correcting transaction {i}: credit -> debit (balance decreased by {abs(balance_diff):.2f})")
                final_transactions[i]["debit"] = credit
                final_transactions[i]["credit"] = None
        
        elif balance_diff > 0:
            if debit is not None and credit is None:
                log(f"Correcting transaction {i}: debit -> credit (balance increased by {balance_diff:.2f})")
                final_transactions[i]["credit"] = debit
                final_transactions[i]["debit"] = None
            
    return final_transactions

# ===============================
# RUNPOD HANDLER
# ===============================
def handler(event):
    log(f"Received event: {event.keys()}")
    if "input" not in event:
        log("ERROR: No 'input' key in event")
        return {"error": "No input provided"}
        
    job_input = event["input"]
    log(f"Input keys: {job_input.keys() if isinstance(job_input, dict) else type(job_input)}")
    
    # Accept either 'pdf_base64' or 'file' as the input key
    pdf_b64 = job_input.get("pdf_base64") or job_input.get("file")
    
    if not pdf_b64:
        log("ERROR: Missing 'pdf_base64' or 'file' in input")
        return {"error": "Missing pdf_base64 or file field"}

    log(f"Received PDF data of length: {len(pdf_b64)}")
    
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
        log(f"Decoded PDF: {len(pdf_bytes)} bytes")
    except Exception as e:
        log(f"ERROR: Invalid base64: {str(e)}")
        return {"error": f"Invalid base64: {str(e)}"}

    # Run Inference
    try:
        final_data = process_pdf(pdf_bytes)
        log(f"Processing complete. Transactions found: {len(final_data) if isinstance(final_data, list) else 'N/A'}")
        return final_data
    except Exception as e:
        log(f"ERROR during process_pdf: {str(e)}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Processing failed: {str(e)}"}

if __name__ == "__main__":
    # Log GPU status at startup
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        log("WARNING: CUDA is NOT available! Model will run on CPU (very slow).")
    
    # Verify vLLM server is accessible
    try:
        models = client.models.list()
        log(f"vLLM server models: {[m.id for m in models.data]}")
    except Exception as e:
        log(f"WARNING: Could not connect to vLLM server: {e}")
    
    try:
        runpod.serverless.start({"handler": handler})
    finally:
        # Clean up vLLM server when handler exits
        log("Shutting down vLLM server...")
        vllm_process.terminate()
        vllm_process.wait(timeout=10)