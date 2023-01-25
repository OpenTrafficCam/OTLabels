"""Export pre-labelled image data to CVAT"""

import json

import fiftyone as fo


class CVAT:
    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        project_name: str = "",
        class_file: str = "",
    ):

        self.url = url
        self.username = username
        self.password = password
        self.project_name = project_name

        if class_file != "":
            with open(class_file) as json_file:
                self.classes = json.load(json_file)

    def export_data(self) -> None:
        dataset = fo.load_dataset("OTLabels")

        dataset.annotate(
            anno_key="test_fiftyone",
            label_field="pre_annotation",
            classes=self.classes,
            label_type="detections",
            segment_size=25,
            task_assignees=["michael"],
            job_assignees=["martin"],
            username=self.username,
            password=self.password,
            url=self.url,
            project_name=self.project_name,
            headers={"X-Organization": "OpenTrafficCam"},
        )

    def import_data(self) -> None:
        dataset = fo.load_dataset("OTLabels")

        dataset.load_annotations(
            anno_key="test_fiftyone",
            dest_field="ground_truth",
            username=self.username,
            password=self.password,
            url=self.url,
        )

        session = fo.launch_app(dataset)
        session.wait()
