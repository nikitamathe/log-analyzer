import re
from typing import List, Dict

def parse_log_file(file_path: str) -> List[Dict[str, str]]:
    """
    Reads a log file and structures each line with metadata for indexing.
    """
    parsed_logs = []
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Determine basic severity log level
        level = "INFO"
        if re.search(r"fail|error|invalid|kill|out of memory", line, re.IGNORECASE):
            level = "ERROR"
        elif re.search(r"warn|close|timeout", line, re.IGNORECASE):
            level = "WARNING"
            
        parsed_logs.append({
            "content": line,
            "line_number": idx + 1,
            "level": level,
            "source": file_path
        })
        
    return parsed_logs

if __name__ == "__main__":
    # Test script locally
    logs = parse_log_file("logs/sample_auth.log")
    print(f"Successfully parsed {len(logs)} log lines.")
    print(f"Sample parsed line: {logs[0]}")
