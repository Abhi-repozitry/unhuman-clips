import requests

url = "https://youtu.be/fKoAOWQHP0o?si=WPu1Rst2Zgyo3XEg"
r = requests.post("http://127.0.0.1:9000/jobs", json={"url": url}, timeout=30)
r.raise_for_status()
j = r.json()
print("JOB_ID", j.get("id"))
print("STATUS", j.get("status"))
print("CURRENT_STAGE", j.get("current_stage") or j.get("stage_index"))
print("ERROR", j.get("error"))
