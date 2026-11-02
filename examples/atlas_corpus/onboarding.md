# Engineering Onboarding Guide

Welcome to Mossgrove Robotics. This guide walks you through your first weeks. Your onboarding buddy
(assigned before you start) is the best person to ask about anything that is not written down here.

## Before day one

Your manager files an IT ticket at least five working days before your start date. You will receive
a laptop (MacBook Pro 14" or a ThinkPad P1, your choice) and a hardware security key.

## Day one

1. Pick up your laptop and security key from the office manager.
2. Sign in to Google Workspace and set up the security key as your second factor.
3. Accept the invitations to the GitHub organization `mossgrove-robotics`, 1Password, Linear, and
   Slack.
4. Meet your buddy for the repository tour at 14:00.

PagerDuty access is added later, only when you start shadowing on-call (see below).

## Development environment

Clone the monorepo `trundle` and run `make bootstrap`. It installs the toolchain, pre-commit hooks,
and the Python and Rust dependencies. Then start the simulator with `./scripts/sim up`. The simulator
is called Pebble Sim; it runs a virtual 12-aisle warehouse with up to 20 simulated robots on a
laptop. If `make bootstrap` fails on an Apple Silicon machine, re-run it with `ARCH=arm64`.

## Floor safety training

Nobody touches a physical robot before completing Floor Safety 101. The 90-minute course covers
lock-out/tag-out, lithium battery handling, ESD precautions, and the location of every e-stop. You
must also be accompanied by a certified colleague for your first three visits to a customer site.

## Your first week

- Day 1-2: environment setup, repository tour, read the architecture document.
- Day 3: pick a ticket labelled `good-first-issue` and open a draft pull request.
- Day 4: pair-programming session with your buddy on a simulator scenario.
- Day 5: team retrospective; you present what surprised you about the codebase.

Your first pull request should be merged within your first 10 working days. Do not worry about size;
a documentation fix counts.

## Code review norms

- One approval is enough for application code; firmware (Tendon) and anything in `safety/` needs two
  approvals, one of them from a member of the safety group.
- Reviewers respond within one business day. If you have not heard back, ping the reviewer on Slack,
  then your buddy.
- Keep pull requests under roughly 400 changed lines where possible; split refactors from behavior
  changes.
- Every bug fix needs a regression test.

## On-call

On-call exists to keep customer warehouses moving. Before you join the primary rotation you must:

1. Finish your first 30 days at the company.
2. Complete two shadow rotations (two weeks in total) where you are paged alongside the primary but
   are not responsible.
3. Run one incident drill with the incident commander on duty.

A rotation lasts one week and hands over on Monday at 10:00 CET. Primary on-call receives a stipend
of 250 EUR per rotation, plus time off in lieu if a page wakes you between 22:00 and 06:00.

## 30/60/90 goals

- 30 days: shipped at least three pull requests, completed Floor Safety 101, shadowing on-call.
- 60 days: own a small component or feature end to end, lead one design review.
- 90 days: join the primary rotation and mentor the next new hire.

## Benefits and practicalities

- Learning budget: 1,500 EUR per year for courses, books and conferences.
- Paid time off: 28 days per year plus public holidays at your home office.
- Work from home: up to three days per week, except during your first two weeks.
- Expenses go through the Expensify workflow; anything over 200 EUR needs manager approval first.
