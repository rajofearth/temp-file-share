# modal.Workspace


```python
class Workspace(modal.object.Object)
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


## members


```python
members: WorkspaceMembersManager
```

Namespace with methods for managing the membership of a Workspace.


### members.list

```python
list(self)
```
Return the members of the Workspace.

**Examples:**

```python notest
members = modal.Workspace.from_context().members.list()
print([m.name for m in members])
```

## from_context

```python
from_context(*, client=None)
```
Look up the Workspace associated with the current context.

This returns the Workspace that the active Modal credentials authenticate against
(i.e., your active profile or the `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment
variables). If called inside a Modal container, it returns the Workspace that the
container is running in.
