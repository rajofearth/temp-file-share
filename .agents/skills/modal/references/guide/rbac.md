# Role-Based Access Control (RBAC)

<Callout variant="gated-feature">
RBAC is available on the <a href="/pricing">Team and Enterprise plans</a>. Visit <a href="/settings/plans">workspace settings</a> to upgrade.
</Callout>

Role-Based Access Control (RBAC) gives Workspace administrators more granular control over who can access and modify resources.

This is especially useful for protecting production while allowing broader access to development and staging.

Modal's RBAC system operates at two levels:

* **Workspace Roles** control overall Workspace permissions.
* **Environment Roles** control access to specific RBAC restricted Environments.

## Workspace Roles

Modal [Workspaces](/docs/guide/workspaces) organize Modal Apps and other resources for a group of users. These roles control access at the level of the entire Workspace.

All Workspace Members have one of three Roles that determine their overall permissions:

* **Owner** — Full read-write access to everything in the Workspace, including billing, Workspace management, and all Environments. Can assign any Role to other members.
* **Manager** — Same as Owner, but cannot modify the Owner Role.
* **Member** — Can deploy and manage Apps, but cannot access billing, Workspace management, or other Workspace settings.

## Environment Roles

Modal [Environments](/docs/guide/environments) isolate Modal Apps and other resources from one another within a Workspace.

Environments can be set as **restricted** to enable RBAC at the Environment level. Restricted Environments introduce two specific Roles for more granular control:

* **Viewer** — Read-only access to resources in the Environment, including dashboards, logs, metrics, app and function configuration.
* **Contributor** — Full read and write access to the Environment. Workspace Owners and Managers automatically have Contributor access to all restricted Environments.

## Setting up restricted Environments

To enable RBAC for specific Environments:

1. **Enable RBAC**: This requires a Team or Enterprise plan. See our [pricing page](/pricing) for more information.

2. **Create or restrict an Environment**: To create a new restricted Environment, use
   [`modal environment create --restricted NAME`](/docs/cli/latest/environment#modal-environment-create). To restrict an existing unrestricted Environment, navigate to your Workspace's
   Environment Management page in [Settings](/settings), select the Environment,
   and click **Make Restricted**.

3. **Configure Environment Roles**: Once an Environment is restricted:
   * All Workspace Members automatically get **Viewer** access.
   * Workspace Owners and Managers automatically get **Contributor** access.
   * You can assign specific users or service users **Contributor** access.

4. **Manage access**: Use the **Manage** button next to any restricted Environment to:
   * Add users or service users with specific Roles
   * Change existing Roles
   * Remove access

### Default access by actor

| Workspace Role               | Unrestricted Environment Default | Restricted Environment Default                           |
| ---------------------------- | -------------------------------- | -------------------------------------------------------- |
| Workspace Owner              | Contributor                      | Contributor                                              |
| Workspace Manager            | Contributor                      | Contributor                                              |
| Workspace Member             | Contributor                      | Viewer by default, or Contributor if explicitly assigned |
| Service user / service token | Contributor                      | Viewer by default, or Contributor if explicitly assigned |

In restricted Environments, you can grant specific users or service users **Contributor** access for deployment and management workflows.

## Service users and service tokens

[Service users](/docs/guide/service-users) are programmatic identities authenticated with API tokens. They are useful for CI/CD pipelines, deployment bots, and other machine-to-machine communication needs.

Unlike human users, service users do not have a Workspace-level role. Their access is controlled through Environment Roles.

| Use case                                     | Recommended identity                                        | How access works                                                                                    |
| -------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Interactive development or manual management | Human user                                                  | Access is based on the user's Workspace Role, plus any Environment Role for restricted Environments |
| Automation in CI/CD or deployment workflows  | Service user authenticated with a service token             | Access is based only on the service user's Environment Role                                         |
| Deploying to an unrestricted Environment     | Human user or service user                                  | Both have Contributor access                                                                        |
| Deploying to a restricted Environment        | Human user or service user with explicit Contributor access | Viewer by default; must be granted Contributor to deploy or manage resources                        |

This makes service users the recommended way to let automation deploy to a restricted Environment without granting broad Workspace permissions.

## Proxy tokens for Web Functions

[Web Functions](/docs/guide/webhooks) can be protected with proxy tokens, which authenticate inbound HTTP requests before they reach your function.

On workspaces with RBAC enabled, proxy tokens are **scoped** — each token is explicitly associated with one or more Environments, and will only be accepted for Web Functions deployed to those Environments. This prevents a token intended for a staging endpoint from being used to call a production one.

### Creating a scoped proxy token

1. Navigate to **Settings → Proxy Tokens** and click **New Token**.
2. Copy the token ID and secret — the secret is only shown once.
3. You will be prompted to select the Environments this token should be valid for.
4. Use the **Manage Environments** button on any existing scoped token to update its Environment associations. Changes take effect immediately, so removing an Environment will instantly revoke access for any clients using that token to call endpoints in that Environment.

### Scoped vs. workspace-wide tokens

| Token type     | Who gets it                  | Valid for                                                  |
| -------------- | ---------------------------- | ---------------------------------------------------------- |
| Scoped         | Workspaces with RBAC enabled | Only the Environments explicitly associated with the token |
| Workspace-wide | Workspaces without RBAC      | Any Web Function in the workspace                          |

Existing workspace-wide tokens continue to work as-is. New tokens created on workspaces with RBAC enabled are scoped by default.

If RBAC is disabled on a workspace, scoped tokens fall back to workspace-wide access.

## Cross-Environment access

Restricted Environments prevent app and task identities in other Environments from accessing resources inside the restricted Environment. For more detail, see [Cross-Environment Lookups](/docs/guide/environments#cross-environment-lookups).

In practice, this means a task can access objects in its own Environment and other unrestricted Environments, but code running in another Environment cannot use APIs such as `modal.App.lookup()`, `Secret.from_name()`, or `Volume.lookup()` to reach into a restricted Environment.

This prevents privilege escalation from a less trusted Environment into a more sensitive one.

### Cross-Environment behavior for app and task identities

Access checks are evaluated against the **target** Environment. That means workloads running inside a restricted Environment can still access objects in an **unrestricted** Environment, but workloads running outside a restricted Environment cannot reach into it.

| Source Environment | Target Environment | Cross-Environment access |
| ------------------ | ------------------ | ------------------------ |
| Unrestricted       | Unrestricted       | Allowed                  |
| Unrestricted       | Restricted         | Denied                   |
| Restricted         | Unrestricted       | Allowed                  |
| Restricted         | Restricted         | Denied                   |

Same-Environment access is unaffected by these cross-Environment rules.

### Example: inbound vs. outbound access

Suppose you have two Environments:

* `prod` — restricted
* `test` — unrestricted

A task running in `test` cannot look up secrets, volumes, or Apps in `prod`.

A task running in `prod` can still access objects in `test`, because `test` is not restricted.

If both `prod` and `test` are restricted, then tasks in one cannot access objects in the other.

## Protecting production secrets with restricted Environments

A common RBAC setup is to place production secrets in a restricted production Environment and grant **Contributor** access only to the human users and service users that should be allowed to deploy or manage production.

| Scenario                                                                       | Result  |
| ------------------------------------------------------------------------------ | ------- |
| Developer in `dev` tries to edit a secret in restricted `prod`                 | Denied  |
| CI service user with Contributor access to restricted `prod` deploys to `prod` | Allowed |
| Task running in `prod` reads a secret in `prod`                                | Allowed |
| Task running in `prod` accesses objects in unrestricted `test`                 | Allowed |

This setup lets you keep development and testing more open while protecting production resources, including secrets, from accidental or unauthorized access.

## Common access patterns

| Pattern                                                                   | Allowed?                               | Notes                                                                                       |
| ------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------- |
| Workspace Member views logs in a restricted Environment                   | Yes                                    | Workspace Members have Viewer access by default in restricted Environments                  |
| Workspace Member deploys to a restricted Environment                      | No                                     | Contributor access is required to deploy or modify resources                                |
| Workspace Owner or Manager deploys to a restricted Environment            | Yes                                    | Owners and Managers automatically have Contributor access                                   |
| Service user deploys to a restricted Environment                          | Yes, if explicitly granted Contributor | Service users have Viewer access by default in restricted Environments                      |
| Task running in `dev` reads a secret in restricted `prod`                 | No                                     | Cross-Environment access into a restricted Environment is denied                            |
| Task running in restricted `prod` accesses objects in unrestricted `test` | Yes                                    | Cross-Environment access is allowed when the target Environment is unrestricted             |
| User views dashboards or app details in a restricted Environment          | Yes                                    | Viewer access includes read-only views such as dashboards, logs, metrics, and configuration |
| Task accesses resources in its own Environment                            | Yes                                    | Same-Environment access is unaffected by cross-Environment restrictions                     |
| Scoped proxy token used on a Web Function in an associated Environment    | Yes                                    | Token must be explicitly associated with the target Environment                             |
| Scoped proxy token used on a Web Function in a non-associated Environment | No                                     | Token is not valid for Environments it has not been associated with                         |

## FAQ

**Can I make Environments completely private?**

No. Workspace Members will always have at least **Viewer** access to restricted Environments. Fully private Environments are planned for the future.

**How do service tokens work with restricted Environments?**

Service tokens authenticate service users. In restricted Environments, service users have **Viewer** access by default unless you explicitly assign them **Contributor** access. This allows automated systems and CI/CD pipelines to deploy and manage production without granting broad Workspace permissions.

**Can I use `modal.App.lookup()` across different restricted Environments?**

No. Apps cannot look up, read from, or write to objects in a different restricted Environment.

**Can code running in a restricted Environment access other Environments?**

Yes, but only when the target Environment is not restricted. A restricted Environment blocks access **into** it from other Environments.
