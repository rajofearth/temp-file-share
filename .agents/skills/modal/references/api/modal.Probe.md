# modal.Probe


```python
class Probe(object)
```

Probe configuration for the Sandbox Readiness Probe.

**Usage**

```python notest
# Wait until a file exists.
readiness_probe = modal.Probe.with_exec(
    "sh", "-c", "test -f /tmp/ready",
)

# Wait until a TCP port is accepting connections.
readiness_probe = modal.Probe.with_tcp(8080)

app = modal.App.lookup('sandbox-readiness-probe', create_if_missing=True)
sandbox = modal.Sandbox.create(
    "python3", "-m", "http.server", "8080",
    readiness_probe=readiness_probe,
    app=app,
)
sandbox.wait_until_ready()
```

```python
__init__(self, tcp_port=None, exec_argv=None, interval_ms=100)
```


## with_tcp

```python
with_tcp(cls, port, *, interval_ms=100)
```


## with_exec

```python
with_exec(cls, *argv, interval_ms=100)
```

