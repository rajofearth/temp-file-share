# modal.billing

## modal.billing.WorkspaceBillingReportItem

```python
class WorkspaceBillingReportItem(TypedDict):
    object_id: str
    description: str
    environment_name: str
    interval_start: datetime.datetime
    cost: decimal.Decimal
    tags: dict[str, str]
```

## modal.billing.workspace_billing_report

```python
workspace_billing_report(*, start, end=None, resolution="d", tag_names=None,
    client=None)
```
Generate a tabular report of workspace usage by object and time.

The result will be a list of dictionaries for each interval (determined by `resolution`)
between the `start` and `end` limits. The dictionary represents a single Modal object
that billing can be attributed to (e.g., an App) along with metadata (including user-defined
tags) for identifying that object.

The `start` and `end` parameters are required to either have a UTC timezone or to be
timezone-naive (which will be interpreted as UTC times). The timestamps in the result will
be in UTC. Cost will be reported for full intervals, even if the provided `start` or `end`
parameters are partial: `start` will be rounded to the beginning of its interval, while
partial `end` intervals will be excluded.

Additional user-provided metadata can be included in the report if the objects have tags
and `tag_names` (i.e., keys) are specified in the request. Alternatively, pass `tag_names=["*"]`
to include all tags in the report. Note that tags will be attributed to the entire interval even
if they were added or removed at some point within it. If the tag name was not in use during an
interval, it will be absent from the tags dictionary in that output row.

In most cases, billing data will be available in the database that this API queries within
minutes, although there may be collection delays. If completeness is important for your use
case, we recommend leaving a buffer after the end of the query interval.

It's also possible to generate reports using the
[`modal billing report`](https://modal.com/docs/reference/cli/billing) CLI command. The CLI
has a few convenience features for generating reports across relative time ranges.
