# Incident Runbook

This runbook covers production incidents affecting Hive, Warren, or robots on customer floors. When
in doubt, page first and investigate second.

## Severity levels

| Level | Definition                                                         | First response |
|-------|--------------------------------------------------------------------|----------------|
| Sev1  | Fleet-wide stall, data loss, or any injury or near miss            | 5 minutes      |
| Sev2  | One customer site degraded, or API error rate above 5%            | 15 minutes     |
| Sev3  | Minor issue with a workaround                                      | Next business day |

Safety trumps uptime. Any event involving a person, smoke, or fire is automatically Sev1.

## Paging and escalation

Alerts page the PagerDuty schedule `hive-primary`. If the primary does not acknowledge within 10
minutes, PagerDuty escalates to `hive-secondary`, and after another 10 minutes to the engineering
manager on duty. Acknowledge a page within 5 minutes even if you only say "looking".

## Roles

- **Incident Commander (IC)**: runs the response, makes decisions, owns the timeline. The IC does not
  debug.
- **Ops Lead**: hands-on debugging and mitigation.
- **Comms Lead**: updates the status page and talks to customer success.

Open a Slack channel named `#inc-<date>-<slug>` (for example `#inc-2025-03-14-allocator`) and start
the Zoom room called "Bridge". Post a one-line summary every 15 minutes.

## Common incidents

### Fleet stall: the task allocator is stuck

Symptoms: robots idle in aisles, tasks pile up, the metric `hive_alloc_queue_depth` stays above 500.

1. Confirm in Grafana that the allocator pods are running but not draining the queue.
2. Restart the allocator: `kubectl rollout restart deploy/hive-allocator -n hive`.
3. Expect recovery within 90 seconds. The `lock_epoch` increases by one; that is normal.
4. Never restart the allocator while a `lock_epoch` change is already in progress.

If the queue is still above 500 after five minutes, escalate to Sev1 and page the database owner.

### Map desync

Symptoms: a robot reports a map version older than Warren's version and refuses new tasks.

- Resync one robot: `trundlectl map resync --robot <id>`.
- If more than 5% of the robots at a site are desynced, freeze publishing with
  `trundlectl map freeze --site <site>` and page the Warren owner before doing anything else.

### Battery thermal event

A battery above 55 degrees Celsius makes the robot park itself automatically. If there is smoke or a
battery smell: evacuate the aisle, press the nearest e-stop, call the site safety lead, and declare a
Sev1. Do not approach the robot to move it. Use the site's fire extinguisher only if trained.

### API 5xx spike

1. Check the gateway dashboard for the failing route.
2. Check PostgreSQL connections; the pool maxes out at 200. A sudden climb means a slow query or a
   leaked connection.
3. If a deploy happened in the last hour, roll it back first and investigate afterwards.

## Communication templates

For Sev1, update the public status page every 30 minutes until resolved. Use this shape:
"Investigating: we are aware of <symptom> affecting <scope>. Next update at <time>."

## After the incident

A blameless postmortem is due within 5 business days for every Sev1 and Sev2. Use the template in the
`postmortems/` folder: summary, impact, timeline, root cause, what went well, what went badly, and
action items. Action items are tracked in Linear with the label `postmortem`, and each has one owner
and a due date.
