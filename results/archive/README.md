Superseded runs kept for the record; nothing here is read by the tool.
- area_macc.csv, timing_macc.csv: before -noalumacc (Yosys fused the multipliers and tree into one $macc).
- area_genlib.csv: -noalumacc but before -constr (ABC mapped against unit gate delays, so -D had no effect).
- area_t1-35500_t2-24850.csv, timing_t1-35500_t2-24850.csv: first T1/T2 runs before the 3% guard band on T1 and before T2 was dropped (D7).
