# modal.call_graph

## modal.call_graph.InputInfo


```python
class InputInfo(object)
```

Simple data structure storing information about a function input.

```python
__init__(self, input_id, function_call_id, task_id, status, function_name,
    module_name, children)
```

## modal.call_graph.InputStatus


```python
class InputStatus(enum.IntEnum)
```

Enum representing status of a function input.

The possible values are:

* `PENDING`
* `SUCCESS`
* `FAILURE`
* `INIT_FAILURE`
* `TERMINATED`
* `TIMEOUT`
