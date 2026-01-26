
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import iam

def add_identity_to_group(wc: WorkspaceClient, group_id: str, identity_id: str):
    wc.groups.patch(
        id=group_id,
        operations=[
            iam.Patch(
                op=iam.PatchOp.ADD,
                value={
                    "members": [
                        {
                            "value": identity_id,
                        }
                    ]
                },
            )
        ],
        schemas=[iam.PatchSchema.URN_IETF_PARAMS_SCIM_API_MESSAGES_2_0_PATCH_OP],
    )

def create_group_if_not_exists(wc: WorkspaceClient, group_name: str):
    group = None
    try:
        group_iterator = wc.groups.list(filter=f'displayName eq "{group_name}"')
        group = next(group_iterator)
        if group:
            return group
    except StopIteration:
        pass
    group = wc.groups.create(display_name=group_name)
    return group


def get_group_by_name(wc: WorkspaceClient, group_name: str):
    group = next(wc.groups.list(filter=f'displayName eq "{group_name}"'))
    if not group:
        raise Exception(f"Group {group_name} not found")
    return group
