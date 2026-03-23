# Running the EKF Test Suite

## From MATLAB GUI

Open MATLAB, set the working directory to the project root
(`C:\Users\hsken\c_dev\sensfus_state_estimation`), then run:

```matlab
addpath('src')
run_ekf_test
```

Plots are saved as `<tag>_position.jpg` and `<tag>_heading.jpg` in the project root.
Figure windows are suppressed by default — open the `.jpg` files to review results.

---

## From the command line (batch / headless)

Run from anywhere — the command sets the working directory internally:

```bat
"C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch "cd('C:/Users/hsken/c_dev/sensfus_state_estimation'); addpath('src'); run_ekf_test"
```

stdout prints the MSE summary table. Plots are saved to the project root as above.

---

## Output files

| File | Contents |
|------|----------|
| `task1_1_position.jpg` … `task2_3_position.jpg` | XY trajectory: GT (blue) vs estimate (red) |
| `task1_1_heading.jpg`  … `task2_3_heading.jpg`  | Yaw over time: GT (blue) vs estimate (red) |

---

## Editing the EKF

All tuning parameters are in the `ONE-TIME INITIALISATION` block at the top of
`src/myEKF_ca.m` (lines ~54–111). After editing, the batch command above will pick
up the changes automatically — no recompilation needed.

See `src/ekf_design_notes.md` for a full record of what has been tried and why.
