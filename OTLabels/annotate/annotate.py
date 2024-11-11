"""Export pre-labelled image data to CVAT"""

import json

import fiftyone
from fiftyone import ViewField


class CVAT:
    def __init__(
        self,
        url: str,
        security_file: str = "config/security.json",
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

        return data

    def export_data(
        self,
        annotation_key: str,
        label_field: str = "pre_annotation",
        samples: int = 0,
        exclude_labels: tuple = (),
        segment_size: int = 25,
        task_size: int = 500,
        task_assignee: str = "lars",
        job_assignees: list = ["lars"],
        dataset_name: str = "OTLabels",
        include_classes: tuple = (),
        overwrite_annotation: bool = False,
        keep_samples: bool = True,
        set_status: bool = True,
    ) -> None:
        dataset = fiftyone.load_dataset(dataset_name)

        runs = dataset.list_annotation_runs()

        if overwrite_annotation and annotation_key in runs:
            dataset.delete_annotation_run(annotation_key)
            print(f"WARNING: Overwriting existing annotation session {annotation_key}!")

        if keep_samples:
            dataset_filtered = dataset.match(
                ViewField("status").is_in(["imported", "pre-annotated"])
            )
            print("INFO: Excluding samples from existing annotation sessions.")
        else:
            dataset_filtered = dataset

        if exclude_labels != ():
            dataset_filtered = dataset_filtered.filter_labels(
                "pre_annotation", ~ViewField("label").is_in(exclude_labels)
            )
            print(f"INFO: Excluding classes {str(exclude_labels)} from images.")

        if include_classes != ():
            match = ViewField("label").is_in(include_classes)
            dataset_filtered = dataset_filtered.match_labels(filter=match)
            print(f"INFO: Ensure classes {str(include_classes)} are in images.")

        if samples > 0:
            dataset_filtered = dataset_filtered.take(samples)
            print(f"INFO: Taking {samples} samples.")
            if samples > len(dataset_filtered):
                print("WARNING: Number of samples is larger than number of images!")

        if len(dataset_filtered) > 0:
            dataset_filtered.annotate(
                anno_key=annotation_key,
                label_field=label_field,
                classes=self.classes,
                label_type="detections",
                task_size=task_size,
                segment_size=segment_size,
                task_assignee=task_assignee,
                job_assignees=job_assignees,
                username=self.username,
                password=self.password,
                url=self.url,
                project_name=self.project_name,
                backend="cvat",
                headers={"X-Organization": self.organization_name},
            )
            print("INFO: Set status to 'in annotation' for selected images.")

            if set_status:
                dataset_filtered = self.set_status(dataset_filtered, "in annotation")

        else:
            print("ERROR: No images to annotate! Please set your filters correctly.")

    # TODO: set status when reimported
    def import_data(
        self,
        annotation_key: str,
        launch_app: bool = True,
        dataset_name: str = "OTLabels",
    ) -> None:
        dataset = fiftyone.load_dataset(dataset_name)

        dataset.load_annotations(
            annotation_key=annotation_key,
            dest_field="ground_truth",
            username=self.username,
            password=self.password,
            url=self.url,
        )

        if launch_app:
            session = fiftyone.launch_app(dataset)
            session.wait()
