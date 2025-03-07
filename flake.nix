{
  description = "A very basic flake";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
  inputs.utils.url = "github:numtide/flake-utils";
  inputs.pre-commit-hooks.url = "github:cachix/pre-commit-hooks.nix";
  inputs.pre-commit-hooks.inputs.nixpkgs.follows = "nixpkgs";

  nixConfig = {
    substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs = { self, nixpkgs, utils, pre-commit-hooks }@flakeArgs:
    utils.lib.eachDefaultSystem (system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ ];
          config.allowUnfree = true;
        };
      in {
        # default environment to build cuda code
        # https://nixos.wiki/wiki/CUDA => making a nix-shell
        devShells.default = pkgs.mkShellNoCC {
          buildInputs = with pkgs; [
            gcc11
          ] ++ (with pkgs.cudaPackages; [
            cudatoolkit
            cuda_nvcc
            cuda_cudart
            cuda_nvprof
          ]);

          CUDA_PATH = pkgs.cudatoolkit;
          # https://github.com/NixOS/nixpkgs/issues/342553
          LD_LIBRARY_PATH = "/run/opengl-driver/lib:/run/opengl-driver-32/lib:${pkgs.cudaPackages.cuda_nvprof.lib}/lib";
        };
      });
}
