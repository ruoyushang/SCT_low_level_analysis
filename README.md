save_training_matrix.py is where you construct the big matrix of 1-D image vectors.
build_eigenvectors.py is where you decompose the big matrix and obtain the eigenvectors. You also build the lookup table. here.
monotel_analysis.py is where you reconstruct an image's energy and source location using the eigenvectors and lookup tables.
