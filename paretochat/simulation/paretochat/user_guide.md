# AGENTS

This document is for AI agents that want to interact with ParetoChat.
It explains what ParetoChat can do, how to phrase requests, and how to provide feedback in ways that map to current behavior.

## What ParetoChat Supports

ParetoChat currently supports these command-level capabilities:

- Find a new solution based on user preferences.
- Summarize session progress and saved solutions.

It also supports feedback processing on the latest solution:

- Classifies feedback sentiment.
- Stores free-text feedback.
- Marks the last solution as saved when appropriate.

## Interaction Model

Each turn is a free-form natural language message.
ParetoChat classifies the message into one or more intent kinds:

- FEEDBACK
- COMMAND
- QUESTION

If multiple intents are present in one message, processing order is:

1. FEEDBACK first
2. COMMAND next (if present)
3. QUESTION path currently ends the flow

Implication: you can combine feedback and a next-step request in one message.

## Command Kinds

The command processor maps user requests to one of the following:

- FIND_NEW_SOLUTION
- SUMMARIZE_SESSION

Examples:

- "Find another solution that prioritizes cost over emissions." -> FIND_NEW_SOLUTION
- "Summarize what we explored and what I saved." -> SUMMARIZE_SESSION

## Expressing Preferences For New Solutions

When asking for a new solution, express your preferences as a human would — in terms of
what matters to you and how objectives relate to each other. ParetoChat will interpret and
translate your preferences into the appropriate strategy.

Good request patterns:

- "Find a solution where cost is most important; I'm willing to accept worse quality for it."
- "Cost matters much more to me than quality right now."
- "Top priority is minimizing cost. Once that's settled, I'd like quality to be as high as possible."
- "I'm aiming for something around cost 100, risk 0.1, and quality 80."
- "Keep emissions below 50, then make profit as high as possible."

## Giving Feedback

Feedback is expected to refer to the latest presented solution.
ParetoChat extracts:

- sentiment: POSITIVE, NEGATIVE, or NEUTRAL
- feedback_text: natural language rationale
- save_last_solution: whether to save the latest solution

Save behavior in current implementation:

- If user explicitly asks to save, save_last_solution is true.
- If user does not explicitly ask, positive feedback may still set save_last_solution to true.

Useful feedback patterns:

- "I like this one, save it."
- "This is better on cost but quality is too low; do not save it."
- "Good trade-off. Save this and now find one with lower emissions."

## Mixed-Intent Messages (Recommended)

You can send feedback and a next command in one turn.

Examples:

- "This is close. Save it, and find a version with slightly lower risk."
- "Not good enough on quality. Try again with higher quality priority."

These map to FEEDBACK + COMMAND and are handled in that order.

## Duplicate Solution Handling

During new-solution search, if a duplicate solution is found, ParetoChat retries with adjusted internal guidance.
Current maximum retry attempts for duplicate avoidance is 3.

## Session Summary Capability

When asked to summarize, ParetoChat can report:

- all explored trials
- user-saved solutions
- the latest solution
- feedback history and explored trade-offs

Good summary prompts:

- "Summarize this session."
- "What have I saved so far, and how do they differ?"

## Stopping

Terminate your interaction with ParetoChat gracefully when you are done.

## Practical Loop For External AI Agents

1. Request a new solution with explicit objective preferences.
2. Evaluate returned trade-offs.
3. Provide feedback on the latest solution, optionally save it.
4. In same message, request the next solution direction.
5. Periodically ask for session summary to compare saved options.
6. Exit gracefully when the process is finished.

## Limitations To Account For

- Question-only inputs currently end the graph flow; full standalone Q and A handling is not implemented yet.
- Command space is currently limited to FIND_NEW_SOLUTION and SUMMARIZE_SESSION.
- Feedback attaches to the latest solution in session state.

## Quick Start Prompt Sequence

Use a sequence like this when orchestrating ParetoChat:

1. "Find a new solution that prioritizes cost over delivery time."
2. "This is good, save it, and now find one that improves delivery time while keeping cost close."
3. "Do not save this one. Find an option with lower risk."
4. "Summarize all explored trials and compare my saved solutions."
5. "Thanks for the support, stop and exit."
