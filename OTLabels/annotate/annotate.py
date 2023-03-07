"""Export pre-labelled image data to CVAT"""

import json

import fiftyone as fo
from fiftyone import ViewField as F


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

    def set_status(self, data, status: str):
        for sample in data.iter_samples(autosave=True, progress=True):
            sample["status"] = status

    def export_data(
        self,
        anno_key: str,
        label_field: str = "pre_annotation",
        samples: int = 0,
        exclude_labels: tuple = (),
        segment_size: int = 25,
        task_assignee: str = "michael",
        job_assignees: list = ["martin"],
        job_reviewers: list = [],
        dataset_name: str = "OTLabels",
        include_classes: tuple = (),
        overwrite_annotation: bool = False,
        keep_samples: bool = True,
    ) -> None:
        dataset = fo.load_dataset(dataset_name)

        runs = dataset.list_annotation_runs()

        if overwrite_annotation and anno_key in runs:
            dataset.delete_annotation_run(anno_key)
            print(f"WARNING: Overwriting existing annotation session {anno_key}!")

        if keep_samples:
            dataset_filtered = dataset.match(
                F("status").is_in(["imported, pre-annotation"])
            )
            print("INFO: Excluding samples from existing annotation sessions.")
        else:
            dataset_filtered = dataset

        if exclude_labels != ():
            dataset_filtered = dataset_filtered.filter_labels(
                "pre_annotation", ~F("label").is_in(exclude_labels)
            )
            print(f"INFO: Excluding classes {str(exclude_labels)} from images.")

        if include_classes != ():
            match = F("label").is_in(include_classes)
            dataset_filtered = dataset_filtered.match_labels(filter=match)
            print(f"INFO: Ensure classes {str(include_classes)} are in images.")

        if samples > 0:
            dataset_filtered = dataset_filtered.take(samples)
            print(f"INFO: Taking {samples} samples.")
            if samples > len(dataset_filtered):
                print("WARNING: Number of samples is larger than number of images!")

        dataset_filtered = self.set_status(dataset_filtered, "in annotation")

        dataset_filtered.annotate(
            anno_key=anno_key,
            label_field=label_field,
            classes=self.classes,
            label_type="detections",
            segment_size=segment_size,
            task_assignee=task_assignee,
            job_assignees=job_assignees,
            job_reviewers=job_reviewers,
            username=self.username,
            password=self.password,
            url=self.url,
            project_name=self.project_name,
            headers={"X-Organization": self.organization_name},
        )

    # TODO: set status when reimported
    def import_data(
        self,
        anno_key: str,
        launch_app: bool = True,
        dataset_name: str = "OTLabels",
    ) -> None:
        dataset = fo.load_dataset(dataset_name)

        dataset.load_annotations(
            anno_key=anno_key,
            dest_field="ground_truth",
            username=self.username,
            password=self.password,
            url=self.url,
        )

        if launch_app:
            session = fo.launch_app(dataset)
            session.wait()
