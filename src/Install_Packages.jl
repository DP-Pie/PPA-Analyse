using Pkg

REQUIRED_PACKAGES = [
    "NativeFileDialog",
    "RobustModels",
    "LinRegOutliers",
    "DataFrames",
    "CSV",
    "GLMakie",
    "JLD2",
    "CairoMakie",
    "Statistics",
    "GLM",
    "StatsModels",
    "StatsAPI",
	"Documenter",
	"Pluto",
    "DataStructures",
    "BenchmarkTools"
]

# Installiere fehlende Pakete
for package in REQUIRED_PACKAGES
    Base.find_package(package) === nothing && Pkg.add(package)
end

