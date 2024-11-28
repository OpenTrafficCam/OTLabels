from math import ceil
from urllib.error import HTTPError

import requests
from annotate.pre_annotate import User


class OpenProject:
    _api_key: str
    _session: requests.Session
    _user: str = "apikey"
    url: str = "https://openproject.platomo.de"
    _api_route: str = "/api/v3"
    _base_url: str = url + _api_route
    query_id: str = "1041"
    to_be_scheduled_status_href = _api_route + "/statuses/5"
    scheduled_status_href = _api_route + "/statuses/6"
    in_motion_custom_field = "customField13"
    importance_custom_field = "customField8"
    reviewer_custom_field = "customField5"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session = self._open_session()

    def _open_session(self) -> requests.Session:
        session = requests.Session()
        session.auth = (self._user, self._api_key)
        return session

    def get_work_packages_to_schedule(self) -> list:
        params = {}
        work_packages = []
        offset = 1
        max_offset = 1
        while offset <= max_offset:
            params["offset"] = offset
            query = self._session.get(
                f"{self._base_url}/queries/{self.query_id}", params=params
            ).json()
            results = query["_embedded"]["results"]
            max_offset = ceil(results["total"] / results["pageSize"])
            work_packages.extend(results["_embedded"]["elements"])
            offset += 1

        return work_packages

    def get_work_package_link(self, work_package: dict) -> str:
        url = self.url + work_package["_links"]["self"]["href"].removeprefix(
            self._api_route
        )
        return url

    @staticmethod
    def get_work_package_type(work_package: dict) -> str:
        wp_type = work_package["_links"]["type"]["title"]
        return wp_type

    def get_work_package_priority(self, work_package: dict) -> int:
        # 7: Low
        # 8: Normal
        # 9: High
        # 10: Immediate
        wp_priority = work_package["_links"]["priority"]["href"].removeprefix(
            self._api_route + "/priorities/"
        )
        return int(wp_priority)

    def get_work_package_importance(self, work_package: dict) -> int:
        # 8: High
        # 9: Normal
        # 10: Low

        wp_priority = work_package["_links"][self.importance_custom_field][
            "href"
        ].removeprefix(self._api_route + "/custom_options/")
        return int(wp_priority)

    def set_in_motion(self, work_package: dict) -> None:

        wp_id = work_package["id"]
        lock_version = work_package["lockVersion"]
        current_status = work_package["_links"]["status"]["href"]
        body = {
            self.in_motion_custom_field: True,
            "lockVersion": lock_version,
        }
        if current_status == self.to_be_scheduled_status_href:
            body["_links"] = {"status": {"href": self.scheduled_status_href}}
        response = self._session.patch(
            f"{self._base_url}/work_packages/{wp_id}",
            json=body,
            params={"notify": False},
        )
        if response.status_code != 200:
            raise HTTPError(
                response.url,
                response.status_code,
                response.text,
                response.headers,  # type: ignore
                response.content,  # type: ignore
            )
        comment = {"comment": {"raw": "Automatically added to motion."}}
        comment_response = self._session.post(
            f"{self._base_url}/work_packages/{wp_id}/activities",
            json=comment,
            params={"notify": False},
        )
        if comment_response.status_code != 201:
            raise HTTPError(
                response.url,
                response.status_code,
                response.text,
                response.headers,  # type: ignore
                response.content,  # type: ignore
            )

    def create_task(
        self, title: str, assignee: User, reviewer: User, content: str, project_id: int
    ) -> None:
        assignee_id = self._get_id_for(assignee.open_project, project_id)
        reviewer_id = self._get_id_for(reviewer.open_project, project_id)
        new_wp = {
            "subject": title,
            "description": {
                "format": "markdown",
                "raw": content,
            },
            "_links": {
                "assignee": {"href": f"/api/v3/users/{assignee_id}"},
                "responsible": {"href": f"/api/v3/users/{assignee_id}"},
                "customField5": {"href": f"/api/v3/users/{reviewer_id}"},
            },
        }
        response = self._session.post(
            f"{self._base_url}/projects/{project_id}/work_packages/",
            json=new_wp,
            params={"notify": False},
        )
        if response.status_code != 201:
            raise IOError("Could not create Task")

    def _get_id_for(self, user: str, project_id: int) -> str:
        response = self._session.get(
            f"{self._base_url}/projects/{project_id}/available_assignees"
        )
        if response.status_code != 200:
            raise IOError("Could not get Assignees")
        for user_id in response.json()["_embedded"]["elements"]:
            if user_id["_links"]["self"]["title"] == user:
                return user_id["id"]
        raise ValueError("User not found")
