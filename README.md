# Visual Coreset

Novel coreset filter method using visual multidimensional geometry of computationally processing Parallel Coordinates data visualizations.

Coresets are tested by training an SVM linear classifier on the coreset and evaluating on the remaining known data throwing out any train samples.

| Dataset     | Train Cases | Validation Cases | Accuracy   |
|-------------|-------------|------------------|------------|
| Fisher Iris | 23          | 127              | 96.06%     |
| WBC9        | 54          | 645              | 97.36%     |

## Coreset Programs

The main program currently is `exhaustive_filter.py`, however, additional algorithms are being tested and compared with this approach. The folder `./coresets` and sub-directories are for storing extracted train/eval splits found with this program.

| Program | Description |
|---------|-------------|
| `minmax_filter.py` | Identifies boundary cases based on min-max values across dimensions. |
| `exhaustive_filter.py` | Initial exhaustive algorithm, best results, slow but parallelizable. |

## Academic References

[1] <https://cs.stanford.edu/people/jure/pubs/craig-icml20.pdf>

[2] <https://dl.acm.org/doi/pdf/10.1145/3580305.3599326>

## License

This project is licensed for both personal and commerical use under the MIT license, see [LICENSE](LICENSE) for full details.
