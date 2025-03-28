# Visual Coreset

Novel coreset filter method using visual multidimensional geometry.

Coresets are tested by training an SVM linear classifier on the coreset and evaluating on the remaining known data throwing out any train samples.

23 cases extracted from Fisher Iris yielding 96.06% accuracy.

58 cases extracted from WBC9 (former non-index wrapped version) yielding 98.20% accuracy.

## References

[1] <https://cs.stanford.edu/people/jure/pubs/craig-icml20.pdf>

[2] <https://dl.acm.org/doi/pdf/10.1145/3580305.3599326>
