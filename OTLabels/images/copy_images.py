import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from pyparsing import Any
from tqdm import tqdm


class Site:
    name: str
    input_path: Path
    output_path: Path
    recording_start_date: datetime
    recording_end_date: datetime
    video_files: list
    geolocation: dict
    cam_type: str
    lense_type: str
    height: int
    occasion: str
    comments: str
    files: list[Path]

    def __init__(
        self,
        name: str,
        input_path: Path,
        output_path: Path,
        video_files: list = [],
        geolocation: dict = {"lat": "", "lon": ""},
        cam_type: str = "OpenTrafficCam",
        lense_type: str = "normal",  # "normal", "fisheye"
        height: int = 5,  # in m
        occasion: str = "",  # project or occasion of recording
        comments: str = "",
    ) -> None:
        self.name = name
        self.input_path = input_path
        self.output_path = output_path
        self.video_files = video_files
        self.geolocation = geolocation
        self.cam_type = cam_type
        self.lense_type = lense_type
        self.height = height
        self.occasion = occasion
        self.comments = comments
        self.files = list(input_path.glob(f"*{name}*"))
        file_names = [str(f) for f in self.files]

        def _get_file_dates(file_names) -> list[datetime]:
            return [
                datetime.strptime(
                    value.group(),
                    "%Y-%m-%d_%H-%M-%S",
                )
                for f in file_names
                if (value := re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", f))
            ]

        file_dates = _get_file_dates(file_names)

        def _set_video_files(file_names) -> list[dict[str, Any]]:
            return [
                dict.fromkeys(
                    [
                        value.group(1).split("/")[-1] + ".mp4"
                        for s in file_names
                        if (value := re.search("/(.*).mp4", s))
                    ]
                )
            ]

        if self.video_files == []:
            self.video_files = _set_video_files(file_names)
        self.recording_start_date = min(file_dates)
        self.recording_end_date = max(file_dates)

        self.tags = {
            "site": self.name,
            "cam_type": self.cam_type,
            "recording_start_date": str(self.recording_start_date),
            "recording_end_date": str(self.recording_end_date),
            "lense_type": self.lense_type,
            "geolocation": self.geolocation,
            "height": f"{self.height}m",
            "occasion": self.occasion,
            "video_files": self.video_files,
            "comments": self.comments,
        }


class ImageSet:
    image_set: list[Site]

    def __init__(self, image_set: list[Site]) -> None:
        self.image_set = image_set

    def copy(self) -> None:
        data_json = dict()
        for site in self.image_set:
            print(f"Copy files for Site {site.name}...")

            site_path = f"{site.output_path}/{site.name}"

            data_json[site.name] = {
                "tags": site.tags,
                "image_path": f"{site_path}/images",
                "label_path": f"{site_path}/{site.name}",
            }

            for image in tqdm(site.files):
                from_file = image
                to_dir = Path(f"{site.output_path}/{site.name}/images")
                to_file = Path(to_dir, f"{str(image).split('/')[-1]}")
                if not to_dir.exists():
                    Path(to_dir).mkdir(parents=True, exist_ok=True)

                shutil.copy(from_file, to_file)

            print(f"Copy files for Site {site.name}...Done!")

        with open(f"{site.output_path}/training_data.json", "w") as outfile:
            json.dump(data_json, outfile)
