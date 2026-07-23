import time
import requests

jid = "aac40ee9-09a5-47bb-bebd-9abd6c61372f"
base = "http://127.0.0.1:9000"

def fetch():
    return requests.get(f"{base}/jobs/{jid}", timeout=10).json()

for i in range(600):  # up to 10 minutes
    j = fetch()
    st = j.get("status")
    cs = j.get("current_stage") or j.get("stage_index")
    err = j.get("error")
    logs = j.get("logs") or []
    tail = logs[-15:]

    print(i, "status=", st, "stage=", cs, "error=", ("SET" if err else "None"))
    if err or st == "ERROR" or st == "DONE":
        print("FINAL_STATUS", j.get("status"))
        print("FINAL_STAGE", j.get("current_stage") or j.get("stage_index"))
        print("FINAL_ERROR", j.get("error"))
        print("LOGS_TAIL", tail)
        break

    time.sleep(1.0)
