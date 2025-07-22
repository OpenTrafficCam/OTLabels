from OTLabels.annotate.annotate import CvatTask
from OTLabels.annotate.pre_annotate import User
from OTLabels.api_keys import OPENPROJECT_API_KEY
from OTLabels.apis.openproject import OpenProject
from OTLabels.logger.logger import logger


class CreateWorkPackages:
    def __init__(self, project_id: int) -> None:
        self._openproject = OpenProject(OPENPROJECT_API_KEY)
        self._project_id = project_id

    def create_open_project_tasks(
        self, tasks: list[CvatTask], assignee: User, reviewer: User
    ) -> None:
        logger().info("Creating OpenProject tasks")
        logger().info(tasks)
        for task in tasks:
            for job in task.jobs:
                self._openproject.create_task(
                    title=f"TEST!!!! --- Annotate job {job} in task {task.id}",
                    content=self._create_content(task),
                    assignee=assignee,
                    reviewer=reviewer,
                    project_id=self._project_id,
                )

    def _create_content(self, task: CvatTask) -> str:
        concatenate_urls = "\n".join(task.job_urls())
        return f"""
# Link

{concatenate_urls}

# Statuswechsel

*   in annotation zu to be reviewed: Zugewiesen auf Reviewer setzen
*   in review zu to be fixed: Zugewiesen auf Verantwortlichen setzen
*   fixing zu close: Nichts weiteres
*   fixing zu to be supervised: Zugewiesen auf Lars setzen

# Kommentare

# Anleitung

1. Durchgehen aller vorbeschrifteten Labels auf der rechten Seite
2. Prüfen und Löschen falsch positiver Labels (wenn es kein Objekt gibt)
3. Prüfen und löschen Sie doppelte Labels für dasselbe Objekt
4. Prüfen und korrigieren der Detektionsklasse für die Objekte
6. Prüfen und korrigieren Sie die Position der 2D-Boxen des Objekts
7. Prüfen Sie, ob es Objekte aus den Detektionsklassen gibt, die noch nicht beschriftet
sind, zeichnen Sie neue 2D-Kästchen und annotieren Sie die Detektionsklasse

# Hinweise

Die aktuelle Übersicht über den Ablauf und die Randfälle ist hier zu finden:
https://hedgedoc.intra.platomo.de/0-fgOfIVRSuDK3VMShBFng

Die allgmeine Anleitung zum Annotieren ist hier zu finden:
/Volumes/platomo/Projekte/001 OpenTrafficCam_live/work/OTLabels/Labelling in CVAT.pdf

Die aktuelle Definition der Klassen ist hier zu finden:
/Volumes/platomo/Produkte/OpenTrafficCam/OTLabels/Klassen_Ground_Truth_V1_2.pdf

"""
