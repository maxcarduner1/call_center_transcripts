from databricks.sdk import WorkspaceClient

def get_service_principal_by_name(wc: WorkspaceClient, name: str):
    sp = next(wc.service_principals.list(filter=f'displayName eq "{name}"'))
    if not sp:
        raise Exception(f"Service principal {name} not found")
    return sp


def get_or_create_service_principal(wc: WorkspaceClient, name: str):
    sp = None
    try:
        sp = get_service_principal_by_name(wc, name)
        return sp
    except Exception as e:
        pass
    sp = wc.service_principals.create(display_name=name)
    return sp
