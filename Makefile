# FormatScope: thin targets so a fresh clone reproduces every result (build-plan.md S2).
# EDA tools: `source ~/tools/oss-cad-suite/environment` first; OpenSTA via $FORMATSCOPE_STA if not on PATH.

PYTHON ?= python
LIB    ?= sky130hd
FMT    ?=
TARGET ?= unc
ALIGN_W ?= 24
EPOCHS ?= 60

FMT_FLAG := $(if $(FMT),--fmt $(FMT),)

.PHONY: env train quant test test-ci synth synth-sweep synth-perturb synth-hier sta plot table demo smoke freeze clean-eda

env:
	$(PYTHON) -m pip install -e ".[train,test]"

train:
	EPOCHS=$(EPOCHS) $(PYTHON) models/train.py

quant:
	$(PYTHON) quant/eval.py

test:
	$(PYTHON) -m pytest quant/tests tool/tests synth/tests sta/tests $(wildcard tb) -q

test-ci:
	$(PYTHON) -m pytest quant/tests tool/tests synth/tests sta/tests $(wildcard tb) -q -m "not slow"

smoke:
	$(PYTHON) synth/run_synth.py --smoke --lib $(LIB)
	$(PYTHON) sta/run_sta.py --smoke --lib $(LIB)

synth:
	$(PYTHON) synth/run_synth.py --lib $(LIB) $(FMT_FLAG) --target $(TARGET) --align-w $(ALIGN_W)

synth-sweep:
	$(PYTHON) synth/run_synth.py --lib $(LIB) --fmt fp8e4m3 --fmt mxint8 --fmt mxfp4 --target $(TARGET) --align-w 32

synth-perturb:
	$(PYTHON) synth/run_synth.py --lib $(LIB) $(FMT_FLAG) --target $(TARGET) --align-w $(ALIGN_W) --perturb

synth-hier:
	$(PYTHON) synth/run_synth.py --lib $(LIB) $(FMT_FLAG) --target unc --align-w $(ALIGN_W) --hier

sta:
	$(PYTHON) sta/run_sta.py

plot:
	formatscope plot --lib $(LIB)
	formatscope table --png --lib $(LIB)

table:
	formatscope table --lib $(LIB)

demo:
	formatscope demo

# Copy the final netlists and reports out of the gitignored working dirs for commit (S9.4).
freeze:
	mkdir -p results/netlists results/reports
	cp synth/out/sky130hd_*_netlist.v results/netlists/
	cp synth/out/sky130hd_*_stat.txt results/reports/
	cp sta/out/sky130hd_*_sta.txt results/reports/
	-cp synth/out/asap7_*_netlist.v results/netlists/
	-cp synth/out/asap7_*_stat.txt sta/out/asap7_*_sta.txt results/reports/

clean-eda:
	rm -rf synth/out sta/out sim_build
