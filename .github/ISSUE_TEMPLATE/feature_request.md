---
name: Feature request
about: Propose a new interpretability tool, model feature, or improvement
title: "[FEAT] "
labels: enhancement
assignees: RithvikReddy0-0
---

## Problem / motivation

What problem does this feature solve? Why does it matter for interpretability research or education?

## Proposed solution

Describe the feature concretely. What would the API look like?

```python
# Example of the proposed API
from kamui.mechinterp import YourNewTool

tool = YourNewTool(model, tokenizer)
result = tool.run(token_ids)
result.plot()
```

## Alternatives considered

What other approaches did you consider? Why is this the best one?

## Where it fits in the architecture

Which module would this live in? Does it depend on any new hook points?

## Research reference

If this is based on a paper or existing tool, link it here.

## Willingness to implement

- [ ] I am willing to implement this and open a PR
- [ ] I am requesting this for someone else to implement
