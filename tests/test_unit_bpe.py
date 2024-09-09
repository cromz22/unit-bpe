from unit_bpe import fit_py, encode_py, decode_py, fit_concurrent_py, encode_concurrent_py, decode_concurrent_py

def test_core_py():
    units = [0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]
    _encoded_units, merges = fit_py(units, 10)

    units_to_encode = [0, 1, 0, 1, 2, 3, 4, 5]
    encoded = encode_py(units_to_encode, merges)
    decoded = decode_py(encoded, merges)

    assert units_to_encode == decoded

def test_concurrent_py():
    units_list = [
        [0, 1, 0, 1, 2, 0, 1, 2, 3],
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]
    ]
    _encoded_units, merges = fit_concurrent_py(units_list, 10)

    units_list_to_encode = [[0, 1, 0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2, 3]]
    encoded = encode_concurrent_py(units_list_to_encode, merges)
    decoded = decode_concurrent_py(encoded, merges)

    assert units_list_to_encode == decoded
