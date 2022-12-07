let
  pkgs = import <nixpkgs> {};
in
   pkgs.mkShell {
    buildInputs = with pkgs; [
      rustfmt
      clippy
      llvmPackages_latest.llvm
      llvmPackages_latest.bintools
      llvmPackages_latest.lld
      rust-analyzer
    ];

    nativeBuildInputs = with pkgs; [
      pkg-config
      cargo
      rustc
      gcc
    ];

    LD_LIBRARY_PATH = builtins.concatStringsSep ":" (with pkgs; (builtins.map (a: ''${a}/lib:'') [
      libxkbcommon
      libGL
      xorg.libX11
      xorg.libXi
      xorg.libXcursor
      xorg.libXrandr
      mesa
      alsa-lib
      wayland
      vulkan-tools
    ]));
    
    RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
  }
