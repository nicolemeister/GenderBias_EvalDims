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

for b in batches:
    status = get(b, "status")
    bid = get(b, "id")

    # Cancel if possible:
    # Docs show cancel methods, but FAQ notes you may not be able to cancel once processing begins. :contentReference[oaicite:5]{index=5}
    if status in {"VALIDATING"}:
        try:
            if hasattr(client.batches, "cancel_batch"):
                client.batches.cancel_batch(bid)
            else:
                client.batches.cancel(bid)
            print("Cancelled batch:", bid)
        except Exception as e:
            print("Could not cancel batch:", bid, status, e)

    # Gather known file references
    for k in ("input_file_id", "output_file_id", "error_file_id"):
        fid = get(b, k)
        if fid:
            file_ids.add(fid)

# --- 2) Also delete any uploaded files with purpose=batch-api ---
files_resp = client.files.list()  # :contentReference[oaicite:6]{index=6}
files = get(files_resp, "data", files_resp)

for f in files:
    if get(f, "purpose") == "batch-api":
        file_ids.add(get(f, "id"))

print(f"About to delete {len(file_ids)} files total.")

# SAFETY SWITCH
DO_DELETE = True
if DO_DELETE:
    for fid in sorted(file_ids):
        try:
            client.files.delete(id=fid)  # :contentReference[oaicite:7]{index=7}
            print("Deleted file:", fid)
        except Exception as e:
            print("Failed to delete:", fid, e)
else:
    print("Dry run only. Set DO_DELETE=True to actually delete.")
