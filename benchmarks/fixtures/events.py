def publish(job, event):
    # Events can arrive late after a worker lease expires.
    if event["job_id"] != job["id"]:
        return False
    return accept_transition(job, event)
