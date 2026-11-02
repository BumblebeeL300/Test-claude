# Trundle Platform Architecture

This document describes how the Trundle warehouse robots and the Hive fleet software fit together at
Mossgrove Robotics. It is the reference for anyone touching runtime code, firmware, or deployment.

## System overview

A Trundle is an autonomous mobile robot (AMR) that carries totes between pick stations and storage
aisles. Each robot runs four on-board subsystems and talks to the cloud-side fleet manager, Hive.

| Component  | Where it runs        | Responsibility                                   |
|------------|----------------------|--------------------------------------------------|
| Whisker    | On robot (Jetson)    | Sensor fusion: lidar, depth cameras, IMU, wheels |
| Wayfinder  | On robot (Jetson)    | Local path planning and obstacle avoidance       |
| Tendon     | On robot (MCU)       | Motor control firmware, 1 kHz control loop       |
| Burrow     | On robot (SQLite)    | Local copy of the site map and recent task state |
| Hive       | Cloud (Kubernetes)   | Task allocation, fleet supervision, public API   |
| Warren     | Cloud (Kubernetes)   | Map server; publishes map versions to robots     |

## Sensing and planning

Whisker fuses two 2D lidars (front and rear), four depth cameras, an IMU and wheel encoders. The
fused obstacle grid is published at 50 Hz. Wayfinder consumes that grid and replans at 20 Hz using a
lattice planner on a 5 cm grid. The maximum speed of a Trundle is 1.5 m/s in open aisles and 0.6 m/s
inside a pick zone.

When Wayfinder cannot find any collision-free path for 3 consecutive planning cycles it enters
recovery mode: the robot stops, rotates in place to re-scan, and then asks Hive for a new route. If
recovery fails twice the robot parks itself and raises a `robot.fault` event with the code `NOPATH`.

### Latency budget

The end-to-end budget from a sensor reading to a motor command is 80 ms. It is split as follows:

- Lidar acquisition and driver: 25 ms
- Whisker fusion: 15 ms
- Wayfinder planning: 30 ms
- Tendon command handling: 10 ms

Any change that adds latency to one stage must find the same saving elsewhere. Budget regressions
are caught by the nightly `latency-gate` job in CI.

## Fleet coordination

Hive allocates tasks with an auction: every 250 ms each idle robot bids on open tasks using travel
time and battery level, and the allocator awards the lowest-cost bids. One Hive site shard supports
up to 400 robots. Hive guarantees that two robots never hold the same aisle lock at once; the lock
carries an epoch number (`lock_epoch`) that increases whenever the allocator restarts.

### Messaging

Robots and Hive exchange messages over NATS JetStream. Important subjects:

- `hive.task.assign` and `hive.task.cancel` (Hive to robots)
- `robot.<id>.telemetry` (robot to Hive, 5 Hz)
- `robot.<id>.heartbeat` (robot to Hive, every 2 seconds)

A robot is considered lost after 3 missed heartbeats, which is 6 seconds. When that happens Hive
releases the robot's aisle locks, marks its tasks as `orphaned`, and re-auctions them to the rest of
the fleet. The lost robot's totes are collected manually by the floor team.

## Safety architecture

Software is never the only line of defense. Every Trundle has a hardware bumper circuit, called the
Bumper, wired directly to the motor drivers. Pressing the bumper or the red e-stop button cuts motor
power without involving any software. The safe-stop distance at full speed is 0.6 m. Firmware changes
to Tendon require two approvals and a bench test on the Pebble rig before merging.

## Data and storage

- Hive state lives in PostgreSQL 15 with a connection pool of at most 200 connections.
- Redis holds ephemeral locks and rate-limit counters.
- Telemetry is kept for 30 days in hot storage and for 1 year in cold object storage (S3).
- Burrow on each robot keeps the current map and the last 500 task events so a robot can finish its
  current job even if the network drops.

Map versions flow from Warren to robots every 30 seconds. A robot that reports an older map version
than Warren is called desynced; see the incident runbook for how to resync it.

## Deployment

Hive and Warren run on Kubernetes in AWS `eu-west-1` (primary) and `us-east-2` (secondary). Releases
use a canary: 5% of sites receive the new version for 30 minutes, and the rollout continues
automatically only if error rate and task completion time stay within 5% of baseline. Firmware is
rolled out separately, one site at a time, outside of shift changes.
