TERMINAL = {"succeeded", "failed", "cancelled"}


def accept_transition(job, event):
    if event["sequence"] <= job["sequence"]:
        return False
    job["sequence"] = event["sequence"]
    job["state"] = event["state"]
    return True
