"""Export pre-labelled image data to CVAT"""

import json

import fiftyone as fo


class CVAT:
    def __init__(
        self,
        url: str,
        security_file: str = "OTLabels/config/security.json",
        organization_name: str = "OpenTrafficCam",
        project_name: str = "",
        class_file: str = "",
    ):

        if security_file != "":
            with open(security_file) as json_file:
                self.security = json.load(json_file)

        self.url = url
        self.username = self.security["username"]
        self.password = self.security["password"]
        self.project_name = project_name
        self.organization_name = organization_name

        if class_file != "":
            with open(class_file) as json_file:
                self.classes = json.load(json_file)

    def export_data(
        self,
        anno_key: str,
        label_field: str = "pre_annotation",
        segment_size: int = 25,
        task_assignees: list = ["michael"],
        job_assignees: list = ["martin"],
    ) -> None:
        dataset = fo.load_dataset("OTLabels")

        dataset.annotate(
            anno_key=anno_key,
            label_field=label_field,
            classes=self.classes,
            label_type="detections",
            segment_size=segment_size,
            task_assignees=task_assignees,
            job_assignees=job_assignees,
            username=self.username,
            password=self.password,
            url=self.url,
            project_name=self.project_name,
            headers={"X-Organization": self.organization_name},
        )

    def import_data(self, anno_key: str) -> None:
        dataset = fo.load_dataset("OTLabels")

        dataset.load_annotations(
            anno_key=anno_key,
            dest_field="ground_truth",
            username=self.username,
            password=self.password,
            url=self.url,
        )

        session = fo.launch_app(dataset)
        session.wait()
