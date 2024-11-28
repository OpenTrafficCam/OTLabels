from OTLabels.annotate.annotate import CvatTask
from OTLabels.annotate.pre_annotate import User
from OTLabels.api_keys import OPENPROJECT_API_KEY
from OTLabels.apis.openproject import OpenProject
from OTLabels.logger.logger import logger


class CreateWorkPackages:
    def __init__(self) -> None:
        self._openproject = OpenProject(OPENPROJECT_API_KEY)
        self._project_id = 37

    def create_open_project_tasks(
        self, tasks: list[CvatTask], assignee: User, reviewer: User
    ) -> None:
        logger().info("Creating OpenProject tasks")
        logger().info(tasks)
        self._openproject.create_task(
            title="Test",
            content="Content",
            assignee=assignee,
            reviewer=reviewer,
            project_id=self._project_id,
        )
