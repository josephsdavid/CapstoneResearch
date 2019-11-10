{lib, buildPythonPackage, pythonSource, umapVar, fetchPypi}:

    buildPythonPackage rec {
      pname = "n2d";
      version = "0.1.2";
    

  src = fetchPypi {
    inherit pname version;
    sha256 ="0ajn372z2q5kql91cylfclr5z8m09g3sibnhmbjggmdhspzwiglp";

  };

      buildInputs = [
        pythonSource.h5py
        pythonSource.Keras
        pythonSource.tensorflow
        pythonSource.scipy
        pythonSource.numpy
        pythonSource.pandas
        pythonSource.seaborn
        pythonSource.matplotlib
        pythonSource.scikitlearn
        pythonSource.numba
        umapVar


        

      ];

    }
