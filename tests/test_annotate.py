from OTLabels.annotate.annotate import CvatTask


class TestCvatTask:
    def test_job_url(self) -> None:
        task_id = 123
        job_0 = 456
        job_1 = 789
        task = CvatTask(
            id=task_id, jobs=[job_0, job_1], cvat_url="https://label.opentrafficcam.org"
        )

        actual_urls = task.job_urls()

        expected_urls = [
            f"https://label.opentrafficcam.org/tasks/{task_id}/jobs/{job_0}",
            f"https://label.opentrafficcam.org/tasks/{task_id}/jobs/{job_1}",
        ]
        assert actual_urls == expected_urls
