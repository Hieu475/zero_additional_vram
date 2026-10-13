# Exploratory run; superseded

This 150-prompt run completed before a cross-check found that 18 prompts from
the prior acceptance pilot were reused. The six prior CNN/DM prompts appeared
in the cross-task holdout, and prior calibration prompts could occur in the
new train/development/in-domain splits. Its data is retained as an audit trail
only and must not be described as independent held-out validation. The
corrected run in `../expanded_heldout_rtx4050_independent/` excludes all 18
prior pilot prompts before splitting and is the result to use.
