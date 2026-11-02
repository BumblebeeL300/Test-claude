# Mossgrove Glossary

Short definitions of the words you will hear in stand-ups, runbooks and customer calls. Terms are
listed alphabetically.

## A to D

**Aisle lock**: an exclusive claim on one aisle segment. Hive grants it to a single robot at a time
so two robots never meet head-on in a narrow aisle. Each lock carries a `lock_epoch`.

**AMR**: autonomous mobile robot. A vehicle that navigates by sensing its surroundings instead of
following magnets or tape. The Trundle is an AMR.

**Auction allocator**: the part of Hive that gives tasks to robots. Idle robots bid using travel time
and battery level every 250 ms.

**Bumper**: the hardware safety circuit on every Trundle. It cuts motor power directly, without any
software, when the physical bumper or the e-stop is pressed. Also called the hardware e-stop.

**Burrow**: the on-robot SQLite database that stores the current map and the last 500 task events.

**Dock**: a charging station. A Trundle returns to the dock when its battery falls below 20%.

## E to P

**Epoch**: see `lock_epoch`. The counter increases every time the allocator restarts, which lets
robots discard stale lock grants.

**Geofence**: a polygon on the map that robots may not enter, for example a no-go zone around a
mezzanine edge.

**Hive**: the cloud fleet manager: task allocation, fleet supervision and the public REST API.

**IC**: incident commander. The person who runs an incident response; the IC coordinates and does not
debug.

**Orphaned task**: a task whose robot was declared lost. Hive re-auctions orphaned tasks to the rest
of the fleet.

**Pebble Sim**: the simulator used for local development. It models a 12-aisle warehouse with up to
20 robots. The Pebble rig is the separate physical bench used to test Tendon firmware.

## S to Z

**Sev1, Sev2, Sev3**: incident severity levels. Sev1 is the most serious and needs a response within
5 minutes.

**Stale map**: a robot's map version is older than the version published by Warren. See desync in the
incident runbook.

**Takt**: the target time between completed picks at a station, used by customers to size their fleet.

**Tendon**: the firmware that drives the motors with a 1 kHz control loop.

**Tote**: the standard bin a Trundle carries. A tote holds up to 25 kg.

**Trundle**: Mossgrove's warehouse AMR, 0.8 m long and 0.55 m wide.

**Warren**: the cloud map server that publishes new map versions every 30 seconds.

**Wayfinder**: the on-robot path planner. It replans at 20 Hz and enters recovery mode when no path
is found.

**Whisker**: the on-robot sensor fusion stack that merges lidar, depth cameras, IMU and wheel
odometry into one obstacle grid at 50 Hz.

**Zone**: a named area of the floor, such as a pick zone or a charging zone. Speed limits and
geofences are attached to zones.
