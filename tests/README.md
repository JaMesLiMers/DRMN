# Tests

Run `python -m unittest discover -s tests -v` from the repository root. All 26 tests passed in the documented CPU environment.

Coverage includes checkpoint corruption, sampling and gradients, BCE/Dice backward computation, padding, phrase aggregation, archived AR calculations, strict checkpoint loading, prediction round trips, PNG annotation conversion, image/text forward execution, pretrained parameter mapping, epoch/resume control, structural variants, and analysis exports.

Model tests use random parameters and synthetic samples without real model parameter updates. The training lifecycle test mocks `optimizer.step`. Paper metrics still require intact trained weights and real data.
