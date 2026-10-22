# AGENTS

This guide explains how to interact with the DESDEO Synchronous NIMBUS method tool. 

## What DESDEO NIMBUS Supports

- Generates new candidate solutions based on objective classifications (not reference points).
- Produces 4 diverse Pareto optimal solutions per iteration using different scalarization functions (NIMBUS, STOM, ASF, GUESS).
- Handles iterative preference refinement through classification-based feedback.
- Supports solution archiving for storing interesting solutions.
- Supports generating intermediate solutions between two previously found solutions.

## Why 4 Solutions?

NIMBUS solves 4 separate sub-problems, each using a different scalarization function:

1. **NIMBUS scalarization** — Based on the classifications provided by the DM
2. **STOM scalarization** — Tries to achieve targets more strictly
3. **ASF (Achievement Scalarizing Function)** — Classic Pareto projection method
4. **GUESS scalarization** — Explores with perturbations

Having 4 diverse solutions helps the DM understand the trade-offs and find better directions without the DM having to explicitly request different viewpoints.

## Classification System

Instead of providing a reference point (like RPM), NIMBUS asks you to classify each objective. There are 5 possible classifications:

- **`<` (Improve)**: The objective should improve from its current value (improve as much as possible)
- **`<=` (Improve Until)**: The objective should improve until a given aspiration level
- **`=` (Keep)**: The objective should stay at its current value
- **`>=` (Worsen Until)**: The objective can worsen until a given reservation level
- **`0` (Change Freely)**: The objective can change freely to support other improvements

Note that if an objective is of maximization type, the higher values are the better. 
For minimization objectives, the smaller values are the better and classifications should be made accordingly.

## Important Classification Rules

A valid classification set MUST include:

- **At least one** class from `{<, <=}` (at least one objective to improve)
- **At least one** class from `{>=, 0}` (at least one objective allowed to worsen/change)

This ensures trade-offs are possible. Invalid classifications will raise an error.

## Input Schema

Use the `desdeo_nimbus` tool with this input format:

First call example to request a starting point:

```json
{}
```

After the starting point, use this format:

```json
{
  "classifications": {
    "f_1": ["<", null],
    "f_2": ["<=", 75.0],
    "f_3": ["=", null],
    "f_4": [">=", 100.0]
  },
  "current_objectives": {
    "f_1": 65.0,
    "f_2": 70.0,
    "f_3": 85.0,
    "f_4": 105.0
  }
}
```

Explanation:
- `f_1: ["<", null]` — Improve freely from 65.0
- `f_2: ["<=", 75.0]` — Improve from 70.0 until aspiration level 75.0
- `f_3: ["=", null]` — Keep at 85.0
- `f_4: [">=", 100.0]` — Can worsen from 105.0 until reservation level 100.0

## Output Schema

The tool returns a JSON string with 4 solutions:

```json
{
  "generated_solutions": [
    {
      "objective_values": {
        "f_1": 72.0,
        "f_2": 75.0,
        "f_3": 85.0,
        "f_4": 98.0
      }
    },
    {
      "objective_values": {
        "f_1": 68.0,
        "f_2": 74.0,
        "f_3": 85.0,
        "f_4": 102.0
      }
    }
  ]
}
```

The array contains **exactly 4 solutions** (or fewer if some could not be generated).

## Recommended Interaction Loop

1. **Generate a starting point** (initial Pareto solution to classify).
   - Use the first iteration to understand the problem's trade-offs.

2. **Evaluate the starting point** and the 4 solutions generated.
   - Assess each against your value function or preferences.

3. **Classify objectives** for the next iteration:
   - Pick one of the 4 solutions as your "current point"
   - Classify each objective (improve, improve-until, keep, worsen-until, or free)
   - Ensure classifications have valid structure

4. **Call NIMBUS again** with your classifications.

5. **Repeat** until:
   - No significant improvement is achieved for several iterations
   - Iteration budget is exhausted
   - You find a satisfactory solution

## Example Workflow

### Iteration 1 (First Call - Generate Starting Point)

Generate an initial solution to understand the problem by calling `desdeo_nimbus` with no arguments. The starting point uses the "compromise" reference point (ideal + nadir midpoint).

### Iteration 2 (First Classification)

Suppose we evaluated the starting point and the initial 4 solutions. We picked Solution 1 with `current_objectives = {f_1: 70, f_2: 73, f_3: 82, f_4: 97}`.

We want:
- `f_1`: Improve (value 70 is low, we want higher) → `["<", null]`
- `f_2`: Improve until 77 (it's 73, target 77) → `["<=", 77]`
- `f_3`: Keep at 82 → `["=", null]`
- `f_4`: Let it worsen until 95 (currently 97) → `[">=", 95]`

```json
{
  "classifications": {
    "f_1": ["<", null],
    "f_2": ["<=", 77],
    "f_3": ["=", null],
    "f_4": [">=", 95]
  },
  "current_objectives": {
    "f_1": 70,
    "f_2": 73,
    "f_3": 82,
    "f_4": 97
  }
}
```

Result: 4 new solutions reflecting this direction.

### Iteration 3 and Beyond

Pick the best solution from iteration 2, classify again, and repeat.

## Practical Guidance

- **Keep all objectives present** in both classifications and current_objectives dicts.
- **Use aspiration levels carefully** — Only set `<=` aspiration levels at values better than the current value.
- **Use reservation levels carefully** — Only set `>=` reservation levels at values worse than the current value.
- **Move gradually** — Shift classifications incrementally between iterations for stable progress.
- **If results degrade** — Archive the best solution found so far, then try different classifications.
- **Aspiration vs. Reservation levels** — Levels should be realistic relative to the problem's Pareto front. If infeasible, NIMBUS may not find solutions satisfying the constraints.

## Using Intermediate Solutions

If you find two interesting solutions and want to explore between them, you can request intermediate solutions. This helps explore the region of the Pareto front between two picked solutions without re-classifying entirely.

Example: If you liked Solution 2 from iteration 3 and Solution 4 from iteration 5, you could request 3 intermediate solutions between them.

## Using Solution Archiving

You can archive solutions of interest during your exploration. This allows you to keep track of candidates even as you explore other regions of the Pareto front, and come back to them later if needed.

## Stopping Criteria

Stop NIMBUS when:

- Your best evaluated solution has not improved for 3-5 iterations
- Iteration budget (e.g., 15 iterations) is reached
- You find a solution that satisfies your preferences
- Classifications become difficult to specify or seem to lead in circles

## Common Errors

### Missing Objective

```
Error: Missing reference_point for symbols: f_1
```

**Fix**: Ensure all objective symbols appear in both `classifications` and `current_objectives`.

### Invalid Classification Combination

```
Error: Invalid classification combination
```

**Fix**: Ensure your classifications include:
- At least one improvement class (`<` or `<=`)
- At least one worsening/free class (`>=` or `0`)

Example invalid: `{"f_1": ["<", null], "f_2": ["<", null], "f_3": ["<", null]}` (no room for trade-off)

### Aspiration Level Not Better Than Current

```
Error: Reference point not better than current value for classification <=
```

**Fix**: For `<=` classifications, the aspiration level must be better than (closer to ideal than) the current value.

### Reservation Level Not Worse Than Current

```
Error: Reference point not worse than current value for classification >=
```

**Fix**: For `>=` classifications, the reservation level must be worse than (farther from ideal than) the current value.

## Integration Tips for LLMs

1. **Always maintain `current_objectives`** — Keep the current point's objectives in your state and use them when calling NIMBUS.
2. **Evaluate all 4 solutions** before classifying — Understanding the trade-offs helps make better classifications.
3. **Track improvement** — Compare best solution values across iterations to monitor progress.
4. **Handle errors gracefully** — If an error occurs (invalid classification), adjust and retry.
5. **Use archive strategically** — Don't archive every solution; only save ones that achieve new fronts or special properties.
