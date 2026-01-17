from together import Together

client = Together()

# 1. Get a list of all batches
all_batches = client.batches.list_batches()

print(f"Found {len(all_batches)} batches. Checking for active jobs...")

for batch in all_batches:
    # 2. Check if the batch is in a state that needs cancelling
    # We skip COMPLETED, FAILED, or already CANCELLED batches
    if batch.status in ["VALIDATING", "IN_PROGRESS"]:
        print(f"Attempting to cancel batch {batch.id} (Status: {batch.status})...")
        
        try:
            # 3. Cancel the batch
            # Note: The documentation mentions batches might not be cancellable 
            # once processing (IN_PROGRESS) begins, but we can attempt it.
            cancelled_batch = client.batches.cancel_batch(batch.id)
            print(f"Successfully sent cancel request for {batch.id}")
            
        except Exception as e:
            print(f"Failed to cancel {batch.id}: {e}")
            
    else:
        pass 
        # print(f"Skipping {batch.id} (Status: {batch.status})")