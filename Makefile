.PHONY: help train generate experiment plot test clean

CKPT ?= runs/gan/checkpoints/best_fid.pt
N_PER_CLASS ?= 1300
SEED ?= 42

help:
	@echo "Usage:"
	@echo "  make train         Train the cWGAN-GP (Step 1)"
	@echo "  make generate      Generate synthetic dataset from best checkpoint (Step 2)"
	@echo "  make experiment    Run full 15-experiment grid (Step 3)"
	@echo "  make plot          Generate accuracy / time / F1 plots (Step 4)"
	@echo "  make test          Run model smoke tests"
	@echo "  make clean         Remove all generated outputs (runs/, data_synth/)"
	@echo ""
	@echo "Override variables, e.g.:"
	@echo "  make generate CKPT=runs/gan/checkpoints/ckpt_epoch0050.pt N_PER_CLASS=800"

train:
	python train_gan.py

generate:
	python scripts/generate_synth.py \
		--ckpt $(CKPT) \
		--n_per_class $(N_PER_CLASS) \
		--seed $(SEED)

experiment:
	python scripts/run_experiments.py

plot:
	python scripts/plot_results.py

test:
	pytest tests/ -v

clean:
	rm -rf runs/ data_synth/
