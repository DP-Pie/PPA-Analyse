# //////////////////////////////////////////
# Überblick der Einträge je Teil
# //////////////////////////////////////////

# Sortiere die Teile nach der Anzahl der Einträge (n) in absteigender Reihenfolge
function show_einträge(dict::Dict{String, Teil})
    sortierte_teile = sort(
    collect(values(dict)),
    by = teil -> teil.n,
    rev = true
    )

    anzahlen = [teil.n for teil in sortierte_teile]
    x = 1:length(anzahlen)

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

# //////////////////////////////////////////
# Zeigt die Regressionsdaten
# //////////////////////////////////////////

function Print_Regression(teil::Teil, ModellNr::Int64 = -1)
    # Check: Regressionsmodelle vorhanden?
    if isempty(teil.RegressionsModell)
        throw(ArgumentError("Das Teil: $(teil.name) hat keine Regressionsdaten hinterlegt."))
    end

    Modell = liste_regressionsmodelle(teil; returnText = false, returnVec = true)

    # Prüfe ModellNr und zeige verfügbare Modelle bei Fehlern
    if ModellNr <= 0 || ModellNr > length(Modell)
        @warn "Ungültige Modellnummer: $ModellNr. Verfügbare Modelle: $Modell"
        liste_regressionsmodelle(teil; returnText = true)
        return nothing
    end

    MyReg = teil.RegressionsModell[Modell[ModellNr]]

    fig = Figure()
    ax = Axis(fig[1:2, 2:3], yaxisposition = :left, ylabel = "Zeit [s]", xlabel = "Bauteilanzahl x", 
              xgridcolor = :deepskyblue, ygridcolor = :deepskyblue)  # Hauptplot
    ax2 = Axis(fig[1:2, 2:3], yaxisposition = :right, ylabel = "Zeit [min]")  # Zweite Achse (Minuten)
    bx = Axis(fig[3, 2:3])    # Density für X-Werte
    cx = Axis(fig[1:2, 1])    # Density für Y-Werte
    dx = fig[3, 1]      # InfoText  

    Label(fig[0,:], "Regression $(MyReg.name) von $(teil.name)", fontsize = 30)

    # Verlinke nur die x-Achsen von ax und bx
    linkxaxes!(ax, bx, ax2)
    linkyaxes!(ax, cx)
    bx.yreversed = true       # Umgekehrte Density für X
    cx.xreversed = true       # X-Achse von cx: Max → 0

    # Berechne y-Limits für beide Achsen
    y_min = minimum([MyReg.points.y; MyReg.points.y_hat]) * 0.9
    y_max = maximum([MyReg.points.y; MyReg.points.y_hat]) * 1.1
    ylims!(ax, y_min, y_max)
    ylims!(ax2, y_min / 60, y_max / 60)  # Minuten = Sekunden / 60

    # Zeichne Scatter + Regressionslinie (über ALLE Punkte)
    scatter!(ax, MyReg.points.x, MyReg.points.y)
    lines!(ax, MyReg.points.x, MyReg.points.y_hat, color = :tomato3)

    # Densities
    density!(bx, MyReg.points.x)                                      # Vertikal (X-Werte)
    density!(cx, MyReg.points.y, direction = :y)                      # Horizontal (Y-Werte)


    InfoTXT = if hasproperty(MyReg.stats, :m)
        # Altes Schema (Theil-Sen): Steigung und Achsenabschnitt als Spalten
        m =  truncate_float(MyReg.stats.m[1]; digits=3)
        n = truncate_float(MyReg.stats.n[1]; digits=3)
        "Regression: $(MyReg.name)\nSteigung: $(m)\nAchsenabschnitt: $(n)"
    else
        # Long-Format (MM): eine Zeile je Term/Koeffizient
        koeffizienten = join(
            ["$(row.Term) = $(truncate_float(row.coef; digits=3))" for row in eachrow(MyReg.stats)],
            "\n")
        "Regression: $(MyReg.name)\n$(koeffizienten)"
    end
    #label!(dx, InfoTXT, justification = :right, color = :tomato3)
    Label(dx, InfoTXT, justification = :right, color = :tomato3)

    return fig
end

function Show_3D_Längen(df::DataFrame; Caption::String="3D Plot mit Längen")
    # Spaltenprüfung
    benötigte_spalten = ["rMenge", "Länge", "tges"]
    if !issubset(benötigte_spalten, names(df))
        fehlend = setdiff(benötigte_spalten, names(df))
        error("Fehlende Spalten im DataFrame: $fehlend")
    end

    fig = Figure(size = (1000, 900))

    # 3D-Plot
    ax = Axis3(fig[1:2, 1:2],
        aspect = (1, 1, 1),
        xlabel = "Anzahl",
        ylabel = "Länge [mm]",
        zlabel = "tges [s]"
    )

    # 2D-Plots
    bx = Axis(fig[1, 3], xlabel = "Anzahl", ylabel = "Länge [mm]", title = "Anzahl-Länge", height = 200)
    cx = Axis(fig[2, 3], xlabel = "Länge [mm]", ylabel = "tges [s]", title = "Länge-tges", height = 200)
    dx = Axis(fig[3, 3], xlabel = "Anzahl", ylabel = "tges [s]", title = "Anzahl-tges", height = 200)

    # Titel
    Label(fig[0,:], Caption, fontsize = 30)

    # Daten plotten
    scatter!(ax, df.rMenge, df.Länge, df.tges, markersize = 5)
    scatter!(bx, df.rMenge, df.Länge, markersize = 5)
    scatter!(cx, df.Länge, df.tges, markersize = 5)
    scatter!(dx, df.rMenge, df.tges, markersize = 5)

    # --- SCHIEBEREGLER MIT LABELS (kompakt) ---
    # Azimuth + Label (vertikal gestapelt)
    azimuth_label = Label(fig, "Azimuth [°]", width = 100)
    azimuth_slider = Slider(fig, range = 0:0.5:360, startvalue = 45)

    # Elevation + Label
    elevation_label = Label(fig, "Elevation [°]", width = 100)
    elevation_slider = Slider(fig, range = 0:0.5:90, startvalue = 30)

    # Zoom + Label
    zoom_label = Label(fig, "Zoom", width = 100)
    zoom_slider = Slider(fig, range = 0.5:0.1:3, startvalue = 1.0)

    # Kombination: Label + Slider vertikal, dann alle horizontal
    slider_grid = grid!(vcat(
        hcat(azimuth_label, azimuth_slider),
        hcat(elevation_label, elevation_slider),
        hcat(zoom_label, zoom_slider)
    ), tellheight = false)

    # Platzierung im Layout (unter den Plots)
    fig[3,1:2] = slider_grid

    # Verbindung mit der Kamera
    on(azimuth_slider.value) do val
        ax.azimuth[] = val
    end
    on(elevation_slider.value) do val
        ax.elevation[] = val
    end
    on(zoom_slider.value) do val
        ax.perspective[] = val
    end

    return fig
end