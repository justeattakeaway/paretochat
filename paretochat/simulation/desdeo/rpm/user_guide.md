# AGENTS

This guide explains how to interact with the DESDEO Reference Point Method (RPM) tool.

## What DESDEO RPM Supports

- Generates new candidate solutions from a submitted reference point.
- Handles iterative preference refinement by repeatedly updating the reference point.
- Starts a session automatically on the first call.

## Input Schema

Use the `desdeo` tool with this input:

- `reference_point`: object mapping objective symbols to numeric target values.
- `reference_point: null` is valid and recommended for the first iteration.

Example:

```json
{{
	"reference_point": {{
		"f_1": 120.0,
		"f_2": 95.0,
		"f_3": 60.0
	}}
}}
```

First iteration example:

```json
{{
	"reference_point": null
}}
```

## Output Schema

The tool returns a JSON string with this shape:

```json
{{
	"generated_solutions": [
		{{
			"objective_values": {{
				"f_1": 100.0,
				"f_2": 80.0,
				"f_3": 70.0
			}}
		}}
	]
}}
```

## Recommended Interaction Loop

1. Start with `reference_point: null`.
2. Evaluate returned solutions using your preference function.
3. Choose a new reference point that moves toward better trade-offs.
4. Call `desdeo` again with the updated reference point.
5. Repeat until no better candidate is found.
6. End interaction gracefully once satisfied.

## Practical Guidance For Reference Points

- Keep all objective symbols present when sending a non-null reference point.
- Move targets gradually between iterations for stable progress.
- If results are poor, shift the reference point toward the region of your best-scoring solution.
- If uncertain, submit `null` again to recover from a bad direction.

## Stopping

Stop when your best evaluated solution has not improved for several iterations or when iteration budget is reached.