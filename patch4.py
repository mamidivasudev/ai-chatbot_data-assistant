import json
import os

file_path = r'D:\AI\AI_file_db_reader - Copy\business_rules.json'

if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    new_data = {}
    for key, value in data.items():
        # Keep the original
        new_data[key] = value
        
        # Add the 'dev_' prefixed version if not already there
        if not key.startswith('dev_'):
            new_data[f"dev_{key}"] = value
            
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4)
        
    print("MIGRATION_SUCCESS")
else:
    print("FILE_NOT_FOUND")
