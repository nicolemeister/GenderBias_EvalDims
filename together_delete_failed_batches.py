from together import Together

client = Together()

# --- 1) List batches (v1: list_batches; v2: list) ---
if hasattr(client.batches, "list_batches"):
    batches = client.batches.list_batches()
else:
    batches = client.batches.list()

def get(obj, key, default=None):
    return getattr(obj, key, obj.get(key, default) if isinstance(obj, dict) else default)

# Collect file IDs referenced by batches
file_ids = set()

print(f"Found {len(batches)} batches. Checking for FAILED status...")

for b in batches:
    status = get(b, "status")
    bid = get(b, "id")

    # --- CHANGE: Target only FAILED batches ---
    if status == 'FAILED':
        print(f"Processing FAILED batch: {bid}")
        
        # 1. Attempt to cancel (Note: API may reject this as it is already terminal)
        try:
            if hasattr(client.batches, "cancel_batch"):
                client.batches.cancel_batch(bid)
            else:
                client.batches.cancel(bid)
            print(f"  - Cancel request sent for: {bid}")
        except Exception as e:
            # Common to see errors here since Failed is a terminal state
            print(f"  - Could not cancel (batch likely already terminal): {bid} | Error: {e}")

        # 2. Gather file references ONLY for these failed batches
        # (Moved inside the 'if' so we don't delete files from successful batches)
        for k in ("input_file_id", "output_file_id", "error_file_id"):
            fid = get(b, k)
            if fid:
                file_ids.add(fid)

# --- 2) Delete the specific files found above ---
if not file_ids:
    print("No files found associated with FAILED batches.")
else:
    print(f"About to delete {len(file_ids)} files associated with FAILED batches.")

    # SAFETY SWITCH
    DO_DELETE = True
    if DO_DELETE:
        for fid in sorted(file_ids):
            try:
                client.files.delete(id=fid)
                print("Deleted file:", fid)
            except Exception as e:
                print("Failed to delete:", fid, e)
    else:
        print("Dry run only. Set DO_DELETE=True to actually delete.")