!pip install avro google-cloud-storage pandas
from google.colab import drive

# This will prompt you to authorize Colab to access your Drive
drive.mount('/content/drive')


import os
import csv
import json
import gcsfs
import io
import threading
import traceback
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_h01_synapses_debug(target_neuron_ids, output_dir='./h01_extracted_synapses', max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    
    target_set = set(int(nid) for nid in target_neuron_ids).union(
                 set(str(nid) for nid in target_neuron_ids))
    
    print(f"🔗 Connecting to Google Cloud Storage via GCSFS...")
    fs = gcsfs.GCSFileSystem(token='anon')
    
    blob_path = 'h01-release/data/20210729/c3/synapses/exported/*'
    all_paths = fs.glob(blob_path)
    
    json_files = [b for b in all_paths if b.endswith('.json') and fs.info(b)['size'] > 0]

    if not json_files:
        print("❌ Found 0 .json shards.")
        return

    # FOR DEBUGGING: Let's only run the first 3 files so we don't wait hours to see if it works.
    # If this works, you can remove the [:3] to run the whole dataset.
    test_files = json_files
    
    print(f"📂 Found {len(json_files)} JSONL shards. Running DEBUG mode on first {len(test_files)} files...\n")

    # --- 1. Disk Writer Thread ---
    write_queue = Queue(maxsize=50000) 

    def writer_worker():
        file_handles = {}
        csv_writers = {}
        fieldnames = ['synapse_id', 'partner_neuron_id', 'direction', 'synapse_type', 'location_x', 'location_y', 'location_z']

        while True:
            item = write_queue.get()
            if item is None: break 
            
            nid, record = item

            if nid not in file_handles:
                filepath = os.path.join(output_dir, f"neuron_{nid}_synapses.csv")
                f = open(filepath, 'w', newline='', encoding='utf-8')
                file_handles[nid] = f
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                csv_writers[nid] = writer
                print(f"💾 [DISK] Created new CSV for neuron {nid}")

            csv_writers[nid].writerow(record)
            write_queue.task_done()

        for f in file_handles.values():
            f.close()

    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    writer_thread.start()

    # --- 2. JSON Streamer (Producer) ---
    def process_shard(file_path):
        filename = file_path.split('/')[-1]
        print(f"🟢 [START] Opening {filename}...")
        
        lines_read = 0
        matches_found = 0
        
        try:
            with fs.open(file_path, 'rb') as raw_file:
                with io.TextIOWrapper(raw_file, encoding='utf-8', errors='replace') as text_file:
                    for line in text_file:
                        lines_read += 1
                        line = line.strip()
                        
                        # DEBUG: Print the first 50 characters of the very first line read
                        if lines_read == 1:
                            print(f"📄 [READ {filename}] Line 1 snippet: {line[:50]}...")
                            
                        if not line: continue
                        
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as e:
                            # DEBUG: If a line fails to parse, show us why (only for the first few errors to avoid spam)
                            if lines_read < 5:
                                print(f"⚠️ [JSON ERROR {filename}] Line {lines_read}: {e} | Text: {line[:50]}")
                            continue 
                        
                        pre_id, post_id = None, None
                        
                        if 'pre_synaptic_site' in record and isinstance(record['pre_synaptic_site'], dict):
                            pre_id = record['pre_synaptic_site'].get('neuron_id')
                        if 'post_synaptic_partner' in record and isinstance(record['post_synaptic_partner'], dict):
                            post_id = record['post_synaptic_partner'].get('neuron_id')

                        if pre_id is None and post_id is None: continue

                        # Match Logic
                        if pre_id in target_set or post_id in target_set:
                            matches_found += 1
                            
                            # DEBUG: Print the first match we find in this file
                            if matches_found == 1:
                                print(f"🎯 [FIRST MATCH {filename}] Found synapse for neuron!")

                            direction = 'outgoing' if pre_id in target_set else 'incoming'
                            target = int(pre_id) if pre_id in target_set else int(post_id)
                            partner = post_id if pre_id in target_set else pre_id
                            
                            loc = record.get('location', {})
                            write_queue.put((target, {
                                'synapse_id': record.get('id', record.get('synapse_id')),
                                'partner_neuron_id': partner,
                                'direction': direction,
                                'synapse_type': record.get('type', 'chemical'),
                                'location_x': loc.get('x'),
                                'location_y': loc.get('y'),
                                'location_z': loc.get('z')
                            }))
                            
        except Exception as e:
            print(f"❌ [CRASH {filename}]: {e}")
            traceback.print_exc() # Prints the exact line of code that caused the crash
            
        print(f"🔴 [DONE] {filename} finished. Read {lines_read} lines. Found {matches_found} matches.")

    # --- 3. Run Parallel Threads ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # NOTE: We are only submitting test_files (the first 3) for this debug run
        futures = [executor.submit(process_shard, blob) for blob in test_files]
        for future in as_completed(futures):
            pass # We don't need the progress bar for the debug test

    # Shut down safely
    write_queue.put(None)
    writer_thread.join()
    print(f"\n✅ Debug scan complete! Check the console output above.")
