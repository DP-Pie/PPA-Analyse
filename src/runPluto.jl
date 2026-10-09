using Pkg
Pkg.activate(@__DIR__)  # gemeinsame Projekt-Umgebung in src/ nutzen (Project.toml/Manifest.toml), kein precompiling von pluto beim Notebookstart

using Pluto

# Pluto starten
Pluto.run()
