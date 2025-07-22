import requests
from annotate.pre_annotate import User

STATUS_TO_BE_ANNOTATED = 23

TYPE_ANNOTATION: int = 30


class OpenProject:
    _api_key: str
    _session: requests.Session
    _user: str = "apikey"
    url: str = "https://openproject.platomo.de"
    _api_route: str = "/api/v3"
    _base_url: str = url + _api_route
    reviewer_custom_field = "customField5"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session = self._open_session()

    def _open_session(self) -> requests.Session:
        session = requests.Session()
        session.auth = (self._user, self._api_key)
        return session

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
                "type": {"href": f"/api/v3/types/{TYPE_ANNOTATION}"},
                "status": {"href": f"/api/v3/statuses/{STATUS_TO_BE_ANNOTATED}"},
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
        elements = response.json()["_embedded"]["elements"]
        for element in elements:
            if element["_type"] == "User" and element["login"] == user:
                return element["id"]
        raise ValueError(f"User {user} not found")
