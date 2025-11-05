{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      utils,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        run-model = pkgs.writeShellScriptBin "run_model" ''
          set -exo pipefail
          if ! pgrep ollama; then
              ${pkgs.ollama}/bin/ollama serve &> /dev/null &
              ollama_pid=$!
          fi

          sleep 0.5s

          ${pkgs.ollama}/bin/ollama run qwen2.5-coder:0.5b-base

          if [[ -n "$ollama_pid" ]]; then
              kill "$ollama_pid"
          fi
        '';
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            nodejs_22
            ollama
            run-model
          ];
        };
      }
    );
}
