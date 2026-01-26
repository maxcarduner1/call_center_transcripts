from databricks.sdk import WorkspaceClient
from databricks.sdk.service import apps
from databricks.sdk.service.apps import App

def get_or_create_app(wc: WorkspaceClient, name: str) -> App:
    try:
        app = wc.apps.get(name=name)
        return app
    except Exception:
        pass
    app_deployment = App(
        name=name,
        default_source_code_path=f"/Workspace/shared/apps/{name}"
    )
    app = wc.apps.create_and_wait(app=app_deployment)
    return app