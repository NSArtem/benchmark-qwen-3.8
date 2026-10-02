def apply_event(store, event, seen_ids=[]):
    seen_ids.append(event["id"])
    store[event["job_id"]] = event["state"]
    return store[event["job_id"]]
