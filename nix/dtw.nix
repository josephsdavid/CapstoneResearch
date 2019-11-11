{lib, buildPythonPackage, fetchPypi, pythonSource}:


buildPythonPackage rec {
  pname = "dtw-python";
  version = "1.0.3" ;

  src = fetchPypi {
    inherit pname version;
    sha256 ="1pm9lfbhalnxariid6lklf5myjb03wxj2s0n2ag1l3b91mi2r76f";
  };

  #doCheck = false;
  buildInputs = [
    pythonSource.scipy 
    pythonSource.numpy
    pythonSource.cython
      ] ;
  propogatedBuildInputs = with pythonSource; [scipy numpy cython] ;
}

