# modal.Environment


```python
class Environment(modal.object.Object)
```


## hydrate

```python
hydrate(self, client=None)
```
Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

*Added in v0.72.39*: This method replaces the deprecated `.resolve()` method.

## name

```python
name(self)
```


## objects


```python
objects: EnvironmentManager
```

Namespace with methods for managing Environment objects.


### objects.create

```python
create(self, name, *, restricted=False, client=None)
```
Create a new Environment.

**Examples:**

```python notest
modal.Environment.objects.create("my-environment")
```

### objects.list

```python
list(self, *, client=None)
```
Return a list of hydrated Environment objects.

**Examples:**

```python notest
environments = modal.Environment.objects.list()
print([e.name for e in environments])
```

### objects.delete

```python
delete(self, name, *, client=None)
```
Delete a named Environment.

Warning: This is irreversible and will transitively delete all objects in the Environment.

**Examples:**

```python notest
modal.Environment.objects.delete("my-environment")
```

## members


```python
members: EnvironmentMembersManager
```

Namespace with methods for managing the membership of a restricted Environment.

See https://modal.com/docs/guide/rbac for more information on restricted Environments.


### members.list

```python
list(self)
```
Return the members of a restricted Environment with their roles.

**Examples:**

```python notest
members = modal.Environment.from_name("my-restricted-env").members.list()
print(members)
# {
#     "users": {"alice": "contributor", "bob": "viewer"},
#     "service_users": {"alice-bot": "contributor"},
# }
```

### members.update

```python
update(self, *, users=None, service_users=None)
```
Add or modify roles for members of a restricted Environment.

Each user or service user will be added to the Environment if not currently a member;
if already a member, the user or service user's role will be updated.

**Examples:**

```python notest
env = modal.Environment.from_name("my-restricted-env")
env.members.update(
    users={"alice": "contributor", "bob": "viewer"},
    service_users={"alice-bot": "contributor"},
)
```

### members.remove

```python
remove(self, *, users=None, service_users=None)
```
Remove members from a restricted Environment.

**Examples:**

```python notest
env = modal.Environment.from_name("my-restricted-env")
env.members.remove(
    users=["alice"],
    service_users=["alice-bot"],
)
```

## from_context

```python
from_context(*, client=None)
```
Look up an Environment object using the current context.

This method returns the Environment that is defined by the local configuration
(i.e., your active profile or the `MODAL_ENVIRONMENT` environment variable), or
it fetches the default environment from the server when not defined locally.
If called inside a Modal container, it will return the Environment that container
is associated with.

## from_name

```python
from_name(name, *, create_if_missing=False, client=None)
```
Look up an Environment object using its name.
