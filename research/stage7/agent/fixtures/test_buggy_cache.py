def test_instances_do_not_share_entries():
    first = RequestCache()
    second = RequestCache()
    first.remember("job-1", "queued")
    assert second.get("job-1") is None
