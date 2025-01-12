{
  description = "A very basic flake";

  # inputs.nixpkgs.url = "github:nixos/nixpkgs/a3f9ad65a0bf298ed5847629a57808b97e6e8077";
  # inputs.nixpkgs.url = "github:nixos/nixpkgs/master";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
  inputs.utils.url = "github:numtide/flake-utils";
  inputs.pre-commit-hooks.url = "github:cachix/pre-commit-hooks.nix";
  inputs.pre-commit-hooks.inputs.nixpkgs.follows = "nixpkgs";

  # nixConfig = {
  #   substituters = [
  #     "https://cuda-maintainers.cachix.org"
  #   ];
  #   trusted-public-keys = [
  #     "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
  #   ];
  # };

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
        ]);

        CUDA_PATH = pkgs.cudatoolkit;
        LD_LIBRARY_PATH = "/run/opengl-driver/lib:/run/opengl-driver-32/lib";
      };
    });
}
