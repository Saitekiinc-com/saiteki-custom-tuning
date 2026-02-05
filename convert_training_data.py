import json

input_file = "training_data.jsonl"
output_file = "training_data_vertex.jsonl"

def convert():
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip(): continue
            
            # Original format: {"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]}
            entry = json.loads(line)
            messages = entry.get("messages", [])
            
            # Target format: 
            # {
            #   "contents": [
            #     {"role": "user", "parts": [{"text": "A"}]},
            #     {"role": "model", "parts": [{"text": "B"}]}
            #   ]
            # }
            
            new_contents = []
            for msg in messages:
                role = msg.get("role")
                # Map 'assistant' to 'model'
                if role == "assistant":
                    role = "model"
                
                content_text = msg.get("content", "")
                
                new_contents.append({
                    "role": role,
                    "parts": [{"text": content_text}]
                })
            
            # Only write if we have a valid conversation pair
            if new_contents:
                new_entry = {"contents": new_contents}
                outfile.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

    print(f"Converted {input_file} to {output_file}")

if __name__ == "__main__":
    convert()
