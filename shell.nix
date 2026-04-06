{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    uv
  ];

  shellHook = ''
    uv sync --quiet 2>/dev/null
    source .venv/bin/activate
  '';
}
