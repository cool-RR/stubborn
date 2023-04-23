# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

transplants=(
    "policy_5 /home/ramrachum/.stubborn/chuck/2023-04-10-20-58-51-238802/tune_rollouts/glug_59bef_00000_0_identity_is_linear=True_2023-04-10_20-58-51/policy_snapshots/0050/policy_5"
    "policy_2 /home/ramrachum/.stubborn/chuck/2023-04-10-20-58-51-238802/tune_rollouts/glug_59bef_00000_0_identity_is_linear=True_2023-04-10_20-58-51/policy_snapshots/0050/policy_2"
    "policy_0 /home/ramrachum/.stubborn/chuck/2023-04-10-20-58-51-238802/tune_rollouts/glug_59bef_00000_0_identity_is_linear=True_2023-04-10_20-58-51/policy_snapshots/0050/policy_0"
    "policy_5 /home/ramrachum/.stubborn/chuck/2023-04-10-22-17-44-204676/tune_rollouts/glug_60893_00000_0_2023-04-10_22-17-47/policy_snapshots/0050/policy_5"
    "policy_2 /home/ramrachum/.stubborn/chuck/2023-04-10-22-17-44-204676/tune_rollouts/glug_60893_00000_0_2023-04-10_22-17-47/policy_snapshots/0050/policy_2"
    "policy_0 /home/ramrachum/.stubborn/chuck/2023-04-10-22-17-44-204676/tune_rollouts/glug_60893_00000_0_2023-04-10_22-17-47/policy_snapshots/0050/policy_0"
)

for transplant in "${transplants[@]}"
do
    python -m stubborn chuck run --use-tune --n-tune-samples 10 --n-generations 100 --transplant-policy ${(s: :)transplant}
done
