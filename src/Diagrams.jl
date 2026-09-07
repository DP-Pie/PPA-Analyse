# //////////////////////////////////////////
# Überblick der Einträge je Teil
# //////////////////////////////////////////

# Sortiere die Teile nach der Anzahl der Einträge (n) in absteigender Reihenfolge
sortierte_teile = sort(
    collect(values(Teiledict)),
    by = teil -> teil.n,
    rev = true
)

anzahlen = [teil.n for teil in sortierte_teile]
x = 1:length(anzahlen)

function show_einträge()
    fig = Figure(size = (1000, 600))
    y_max = maximum(anzahlen) +10
    ax = Axis(
        fig[1, 1],
        xlabel = "Teile, absteigend sortiert",
        ylabel = "Anzahl der Einträge",
        title = "Eintragsverteilung der Teile",

        limits = (-10, 2600, 0, y_max),
        xticks = 0:100:2600,
        yticks = 0:10:y_max,

        backgroundcolor = :white
    )

    barplot!(
        ax,
        x,
        anzahlen;
        colorrange = (1, max(length(x), 2)),
        strokewidth = 0
    )

    gesamt = sum(anzahlen)
    maximalwert = maximum(anzahlen)
    anzahl_teile = length(anzahlen)
    prozent_bis_30 = 100 * count(anzahlen .<= 30) / anzahl_teile
    prozent_bis_20 = 100 * count(anzahlen .<= 20) / anzahl_teile
    prozent_bis_10 = 100 * count(anzahlen .<= 10) / anzahl_teile

    text!(
        ax,
        2550,
        y_max * 0.95,
        text = """
    Rückmeldungen gesamt: $gesamt
    EInträge je Teil: $anzahl_teile
    Maximalwert: $maximalwert

    Teile ≤ 30 Einträge: $(round(prozent_bis_30, digits=1)) %
    Teile ≤ 20 Einträge: $(round(prozent_bis_20, digits=1)) %
    Teile ≤ 10 Einträge: $(round(prozent_bis_10, digits=1)) %
    """,
        align = (:right, :top),
        fontsize = 16,
        color = :black,
        space = :data
    )
    fig
end 