{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    python312Packages.pip
    python312Packages.virtualenv
  ];

  # tokenizers (transitive dep of litellm) links against libstdc++
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];

  shellHook = ''
    if [ ! -d .venv ]; then
      python -m venv .venv
      .venv/bin/pip install -e ".[dev]"
    fi
    source .venv/bin/activate
  '';
}
