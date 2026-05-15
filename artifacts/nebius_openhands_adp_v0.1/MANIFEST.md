# Nebius OpenHands ADP Canonical v0.1

## Source dataset

- Source: nebius/SWE-rebench-openhands-trajectories
- Source trajectories: 67,074
- Source format: OpenHands message trajectories
- Source license: CC-BY-4.0

## Conversion

- Converter: scripts/openhands_raw_to_adp.py
- Conversion type: canonical OpenHands → ADP
- OpenHands tool calls are preserved as ADP ApiAction.
- System/user/tool messages are preserved as ADP TextObservation.
- Assistant-only messages are preserved as ADP MessageAction.
- Top-level metadata is preserved in Trajectory.details as strings.

## Validation

- ADP rows: 67,074
- Validation errors: 0
- Repositories: 1,823
- Resolved: 32,161
- Unresolved: 34,913

## ADP class counts

- text_observation: 4,384,161
- api_action: 4,316,760
- message_action: 206

## API function counts

- execute_bash: 2,197,279
- str_replace_editor: 1,821,337
- think: 180,365
- finish: 60,749
- task_tracker: 57,030

## Known limitations

- Tool observation `tool_call_id` is not represented as a first-class TextObservation field.
- Assistant-side tool call IDs are preserved in ApiAction.kwargs under `_openhands_tool_call_id`.
- Tool-output pairing is recoverable by trajectory order.
