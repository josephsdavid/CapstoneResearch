let
  pkgs = import <stable> {};

  umap = pkgs.callPackage ./umap.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    fetchPypi = pkgs.python37.pkgs.fetchPypi;
    pythonSource = pkgs.python37Packages;
  };
  dtw = pkgs.callPackage ./dtw.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    fetchPypi = pkgs.python37.pkgs.fetchPypi;
    pythonSource = pkgs.python37Packages;
  };

  n2d = pkgs.callPackage ./n2d.nix {
    buildPythonPackage = pkgs.python37.pkgs.buildPythonPackage;
    pythonSource = pkgs.python37Packages;
    fetchPypi = pkgs.python37.pkgs.fetchPypi;
    umapVar = umap;
  };

in
  pkgs.mkShell {
    name = "Capstone";
    buildInputs = with pkgs; [
      openblas
      armadillo
      python37
      n2d
      python37Packages.numpy
      python37Packages.scikitlearn
      python37Packages.numba
      zip
      python37Packages.scipy
      python37Packages.pip
      python37Packages.pandas
      python37Packages.seaborn
      python37Packages.h5py
#      python37Packages.pytorch
#      python37Packages.torchvision
      python37Packages.tensorflowWithCuda
      python37Packages.tensorflow-tensorboard
      python37Packages.tensorflow-probability
      umap
      python37Packages.pillow
      python37Packages.matplotlib
      python37Packages.Keras
      python37Packages.virtualenv
      python37Packages.twine
      python37Packages.wheel
      python37Packages.hdbscan
      python37Packages.sphinx
      python37Packages.recommonmark
      python37Packages.sphinx_rtd_theme
      python37Packages.ipython
      dtw
    ];
    shellHook = ''
      '';

  }
