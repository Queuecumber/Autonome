{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    uv
  ];

  # tokenizers (transitive dep of litellm) links against libstdc++
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];

  shellHook = ''
    uv sync --extra dev --quiet 2>/dev/null
    source .venv/bin/activate
  '';
}
