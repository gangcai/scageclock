

def pytest_addoption(parser):
    parser.addoption("--test_data_dir", action="store", required=True, help="Path to root path of the .h5ad files (testing datasets)")
    parser.addoption("--meta_file", action="store", required=True, help="Path to meta file")
    parser.addoption("--k_fold_data_dir", action="store", required=True, help="Path to root path of the .h5ad files (k-fold mode datasets)")
    parser.addoption("--out_root_dir", action="store", required=True, help="Path to the output")

