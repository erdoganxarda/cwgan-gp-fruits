# Contributing

Contributions, bug reports, and suggestions are welcome.

## Reporting Issues

Open a GitHub issue with:
- A clear description of the problem
- Steps to reproduce it
- Your OS, Python version, and PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- Relevant error output or log snippets

## Making Changes

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep diffs focused — one logical change per PR.
3. Run the smoke tests before submitting:
   ```bash
   pytest tests/ -v
   ```
4. Open a pull request with a short description of what changed and why.

## Code Style

- Follow the existing style (PEP 8, no external formatters enforced).
- Keep functions small and self-documenting.
- Add or update docstrings if you change public-facing behaviour.
- Do not commit model checkpoints, datasets, or generated images.

## Experiment Results

If you run experiments and get results worth sharing, add them to the **Results** section of the README or open an issue with the numbers. Results from different hardware or dataset splits are especially interesting.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
