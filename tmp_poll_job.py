import time
import requests

jid = "1842c02e-93ad-4880-8b6c-de9323e87408"
base = "http://127.0.0.1:9000"

def fetch():
    return requests.get(f"{base}/jobs/{jid}", timeout=10).json()

for i in range(180):
    j = fetch()
    st = j.get("status")
    cs = j.get("current_stage") or j.get("stage_index")
    err = j.get("error")
    print(i, "status=", st, "stage=", cs, "error=", ("SET" if err else "None"))
    logs = j.get("logs") or []
    # Break on error/finished
    if st == "ERROR" or err:
        break
    if st == "DONE":
        break
    time.sleep(1.0)

j = fetch()
print("FINAL_STATUS", j.get("status"))
print("FINAL_STAGE", j.get("current_stage") or j.get("stage_index"))
print("FINAL_ERROR", j.get("error"))

logs = j.get("logs") or []
print("LOGS_TAIL", logs[-30:])
