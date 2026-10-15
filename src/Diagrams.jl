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

"""
    Plot_Gruppe(gruppe; X, Y, ...) -> Figure

Scatterplot je Gruppe mit Toggle je Gruppe zum Ein-/Ausblenden.
X und Y müssen numerische Spalten sein.
"""
function Plot_Gruppe(gruppe::GroupedDataFrame{DataFrame},
        X::Symbol = :rMenge,
        Y::Symbol = :tges;
        Gruppenname::Symbol = :Gruppe,
        xlab::String = "rMenge",
        ylab::String = "tges",
        caption::String = "Gruppenplot",
        maxrow::Int64 = 10,
        x_min = nothing,
        x_max = nothing)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = xlab, ylabel = ylab, title = caption)

    # Seitenpanel für die Toggles
    panel = GridLayout(fig[1, 2])

    zeile = 1
    spalte = 0
    for g in gruppe
        name = String(g[1, Gruppenname])
        x = g[!, X]
        y = g[!, Y]

        # Frühzeitiger, verständlicher Fehler statt Makie-Conversion-Wall
        if !(eltype(x) <: Real) || !(eltype(y) <: Real)
            error("Spalten ($X, $Y) müssen numerisch sein, gefunden: " *
                  "$(eltype(x)), $(eltype(y))")
        end

        tg = Toggle(fig, active = true)
        lb = Label(fig, name)

        panel[zeile, 1 + spalte] = lb
        panel[zeile, 2 + spalte] = tg

        scatter!(ax, x, y, visible = tg.active)

        zeile += 1
        if zeile > maxrow
            spalte += 2   # Toggle plus Text = 2er Schritt
            zeile = 1
        end
    end

    panel.tellheight = false   # Panel nicht auf Toggle-Höhe zusammenschrumpfen lassen

    # Achsengrenzen nur setzen, wenn mindestens eins übergeben wurde
    if !isnothing(x_min) || !isnothing(x_max)
        xlims!(ax, x_min, x_max)
    end

    return fig
end

function Plot_Gruppe(df::DataFrame, X::Symbol = :rMenge, Y::Symbol = :tges; kwargs...)
    return Plot_Gruppe(groupby(df, :Gruppe), X, Y; kwargs...)
end

function Plot_Gruppe(teiledict::Dict{String,Teil},
        X::Symbol = :rMenge,
        Y::Symbol = :tges;
        Regression::String = "Theil-Sen",
        xlab::String = "rMenge",
        ylab::String = "tges",
        caption::String = "Gruppenplot",
        maxrow::Int64 = 10,
        x_min = nothing,
        x_max = nothing)

        fig = Figure()
        ax = Axis(fig[1, 1], xlabel = xlab, ylabel = ylab, title = caption)

        # Seitenpanel für die Toggles
        panel = GridLayout(fig[1, 2])

        zeile = 1
        spalte = 0

        for (key,val) in teiledict 
            name = val.name
            x = val.df[!, X]
            y = val.df[!, Y]

            # Frühzeitiger, verständlicher Fehler statt Makie-Conversion-Wall
            if !(eltype(x) <: Real) || !(eltype(y) <: Real)
                error("Spalten ($X, $Y) müssen numerisch sein, gefunden: " *
                    "$(eltype(x)), $(eltype(y))")
            end

            tg = Toggle(fig, active = true)
            lb = Label(fig, name)

            panel[zeile, 1 + spalte] = lb
            panel[zeile, 2 + spalte] = tg

            scatter!(ax, x, y, visible = tg.active)

            zeile += 1
            if zeile > maxrow
                spalte += 2   # Toggle plus Text = 2er Schritt
                zeile = 1
            end

            if haskey(val.RegressionsModell, Regression)
                reg = val.RegressionsModell[Regression]
                if hasproperty(reg.stats, :m)
                    m = reg.stats.m[1]          # Steigung
                    n = reg.stats.n[1]          # Achsenabschnitt
                    ablines!(ax, n, m; linestyle = :dash, visible = tg.active)
                end
                if hasproperty(reg.Diagnostik, :Q1)
                    Intercept_Q2_5 = reg.Diagnostik.Q2_5[1]
                    Intercept_Q1 = reg.Diagnostik.Q1[1]
                    Intercept_Q3 = reg.Diagnostik.Q3[1]
                    Intercept_Q97_5 = reg.Diagnostik.Q97_5[1]
                    Slope_Q2_5 = reg.Diagnostik.Q2_5[3]
                    Slope_Q1 = reg.Diagnostik.Q1[3]
                    Slope_Q3 = reg.Diagnostik.Q3[3]
                    Slope_Q97_5 = reg.Diagnostik.Q97_5[3]
                    ablines!(ax, Intercept_Q2_5, Slope_Q2_5; linestyle = :dash, visible = tg.active, color = :firebrick1)
                    ablines!(ax, Intercept_Q1, Slope_Q1; linestyle = :dash, visible = tg.active, color = :coral)
                    ablines!(ax, Intercept_Q3, Slope_Q3; linestyle = :dash, visible = tg.active, color = :coral)
                    ablines!(ax, Intercept_Q97_5, Slope_Q97_5; linestyle = :dash, visible = tg.active, color = :firebrick1)
                end
            else
                @warn "Teil $(val.name): kein Regressionsmodell \"$Regression\" hinterlegt"
            end
        end        

        panel.tellheight = false   # Panel nicht auf Toggle-Höhe zusammenschrumpfen lassen

        # Achsengrenzen nur setzen, wenn mindestens eins übergeben wurde
        if !isnothing(x_min) || !isnothing(x_max)
            xlims!(ax, x_min, x_max)
        end

        return fig
end

"""
 # Plot_Bayes_Gruppen(daten, X, Y, gruppe, bayes; ...) -> Figure

 Gruppenplot für das hierarchische Bayes-Modell (Weg 2, `robust_bayes_lmm`):
 je Gruppe die Punktwolke aus den Daten plus die zugehörige partiell gepoolte
 Bayes-Gerade. Ein Toggle je Gruppe blendet Punkte und Gerade gemeinsam
 ein/aus.

 ## Argumente

 - `daten`: DataFrame mit den Messpunkten (z.B. AOI_fit_df), benötigt die
   Spalten `X`, `Y` und `gruppe`.
 - `X`, `Y`, `gruppe`: Spaltennamen von Prädiktor, Zielgröße und Gruppierung.
 - `bayes` (Keyword, Pflicht): das Rückgabe-NamedTuple von `robust_bayes_lmm` (genutzt wird
   `bayes.gruppe` mit den Spalten `:Gruppe`, `:a`, `:m`, optional `:a_lo`,
   `:a_hi`, `:m_lo`, `:m_hi`); alternativ kann direkt dieses DataFrame
   übergeben werden.
 - `band = true`: zeichnet zusätzlich die Intervall-Ränder der Koeffizienten
   als graue Begleitgeraden (a_lo/m_lo und a_hi/m_hi). Das ist die intuitive
   Spanne der Randkombinationen, kein simultanes Kredibilitätsband.
 - `maxrow, x_min, x_max, xlab, ylab, caption`: wie in `Plot_Gruppe`.

 ## Hinweis

 Gruppen in `daten` ohne Fit (z.B. "Sonstiges", das für das Modell
 herausgefiltert wurde) werden als reine Punktwolke mit eigenem Toggle
 gezeigt; eine Sammelwarnung listet sie auf.

 ## Beispiel
 ```julia
 fig = Plot_Bayes_Gruppen(AOI_fit_df; bayes = AOI_bayes)
 fig = Plot_Bayes_Gruppen(AOI_fit_df, :rMenge, :Zeit, :Gruppe; bayes = AOI_bayes)
 ```
"""
function Plot_Bayes_Gruppen(daten::AbstractDataFrame,
        X::Symbol = :rMenge,
        Y::Symbol = :Zeit,
        gruppe::Symbol = :Gruppe;
        bayes,
        xlab::String = "rMenge",
        ylab::String = "Zeit",
        caption::String = "Bayes-Regression je Gruppe (partiell gepoolt)",
        maxrow::Int64 = 10,
        x_min = nothing,
        x_max = nothing,
        band::Bool = true)

    bayes_gruppe = bayes isa AbstractDataFrame ? bayes : bayes.gruppe

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = xlab, ylabel = ylab, title = caption)

    # Seitenpanel für die Toggles
    panel = GridLayout(fig[1, 2])

    zeile = 1
    spalte = 0

    for r in eachrow(bayes_gruppe)
        name = r.Gruppe
        sub = subset(daten, gruppe => ByRow(==(name)))
        x = sub[!, X]
        y = sub[!, Y]

        # Frühzeitiger, verständlicher Fehler statt Makie-Conversion-Wall
        if !(eltype(x) <: Real) || !(eltype(y) <: Real)
            error("Gruppe $(name): Spalten ($X, $Y) müssen numerisch sein, " *
                  "gefunden: $(eltype(x)), $(eltype(y))")
        end

        tg = Toggle(fig, active = true)
        lb = Label(fig, name)

        panel[zeile, 1 + spalte] = lb
        panel[zeile, 2 + spalte] = tg

        scatter!(ax, x, y, visible = tg.active)

        # Partiell gepoolte Bayes-Gerade der Gruppe
        ablines!(ax, r.a, r.m, linestyle = :dash, color = :crimson,
                 visible = tg.active)

        # Intervall-Ränder der Koeffizienten als Begleitgeraden
        if band && hasproperty(bayes_gruppe, :a_lo) &&
           hasproperty(bayes_gruppe, :a_hi) &&
           hasproperty(bayes_gruppe, :m_lo) &&
           hasproperty(bayes_gruppe, :m_hi)

            ablines!(ax, r.a_lo, r.m_lo, color = (:gray, 0.5), linewidth = 1,
                     visible = tg.active)
            ablines!(ax, r.a_hi, r.m_hi, color = (:gray, 0.5), linewidth = 1,
                     visible = tg.active)
        end

        zeile += 1
        if zeile > maxrow
            spalte += 2   # Toggle plus Text = 2er Schritt
            zeile = 1
        end
    end

    # Daten-Gruppen ohne Fit: nur Punktwolke mit eigenem Toggle
    ohne_fit = setdiff(unique(String.(daten[!, gruppe])), String.(bayes_gruppe.Gruppe))

    for name in ohne_fit
        sub = subset(daten, gruppe => ByRow(==(name)))
        x = sub[!, X]
        y = sub[!, Y]

        (eltype(x) <: Real && eltype(y) <: Real) || continue   # unbrauchbare Spalten überspringen

        tg = Toggle(fig, active = true)
        lb = Label(fig, name)

        panel[zeile, 1 + spalte] = lb
        panel[zeile, 2 + spalte] = tg

        scatter!(ax, x, y, visible = tg.active)

        zeile += 1
        if zeile > maxrow
            spalte += 2
            zeile = 1
        end
    end

    isempty(ohne_fit) || @warn "Keine Bayes-Gerade für: $(join(ohne_fit, ", ")) " *
                               "(Gruppe(n) nicht im Modell, nur Punktwolke gezeigt)"

    panel.tellheight = false   # Panel nicht auf Toggle-Höhe zusammenschrumpfen lassen

    # Achsengrenzen nur setzen, wenn mindestens eins übergeben wurde
    if !isnothing(x_min) || !isnothing(x_max)
        xlims!(ax, x_min, x_max)
    end

    return fig
end

"""
 # Plot_Bayes_Residuen(residuen; ν = 30, caption = "...") -> Figure

 Residuen-Diagnostik des hierarchischen Bayes-Modells in vier Panels:

 1. Residuum vs. Vorhersage — ein Trichter (Streuung wächst mit der
    Vorhersage) zeigt Heteroskedastizität, das Hauptverdachtsfeld, wenn ν am
    Prior-Anschlag klebt.
 2. Residuum vs. Prädiktor (x) — systematische Bögen deuten auf nichtlineare
    Mengenabhängigkeit, die eine Gerade je Gruppe nicht fangen kann.
 3. QQ-Plot der standardisierten Residuen gegen die t(ν)-Verteilung mit dem
    geschätzten ν — Punkte auf der Diagonalen bestätigen die Fehlerannahme.
 4. Beobachtet vs. Vorhersage — Diagonale als Gesamtfit-Check.

 ## Argumente

 - `residuen`: Rückgabe von `bayes_residuen` (Hierarchische_Regression.jl)
 - `ν`: t-Freiheitsgrade aus `bayes.ν` (Fallback 30 ≈ Normalverteilung)

 ## Beispiel
 ```julia
 fig = Plot_Bayes_Residuen(AOI_residuen; ν = AOI_bayes.ν)
 ```
"""
function Plot_Bayes_Residuen(residuen::AbstractDataFrame;
        ν::Real = 30.0,
        caption::String = "Residuen-Diagnostik des Bayes-Modells")

    ν = isfinite(ν) ? ν : 30.0

    fig = Figure(size = (1100, 800))
    Label(fig[0, :], caption, fontsize = 18)

    # 1. Residuum vs. Vorhersage (Trichter-Check)
    ax1 = Axis(fig[1, 1], xlabel = "Vorhersage [s]", ylabel = "Residuum [s]",
               title = "Residuum vs. Vorhersage")
    scatter!(ax1, residuen.vorhersage, residuen.residuum,
             color = (:steelblue, 0.6), markersize = 6)
    hlines!(ax1, 0, color = :red, linestyle = :dash)

    # 2. Residuum vs. Prädiktor
    ax2 = Axis(fig[1, 2], xlabel = "Prädiktor (x)", ylabel = "Residuum [s]",
               title = "Residuum vs. Prädiktor")
    scatter!(ax2, residuen.x, residuen.residuum,
             color = (:steelblue, 0.6), markersize = 6)
    hlines!(ax2, 0, color = :red, linestyle = :dash)

    # 3. QQ-Plot gegen t(ν)
    ax3 = Axis(fig[2, 1],
               xlabel = "Theoretische t-Quantile (ν = $(round(ν; digits = 2)))",
               ylabel = "Standardisierte Residuen",
               title = "QQ-Plot gegen t(ν)")
    r_sort = sort(residuen.resid_std)
    n = length(r_sort)
    theo = [quantile(TDist(ν), (i - 0.5) / n) for i in 1:n]
    scatter!(ax3, theo, r_sort, color = (:steelblue, 0.6), markersize = 6)
    lim = maximum(abs, vcat(theo, r_sort)) * 1.05
    limits!(ax3, -lim, lim, -lim, lim)
    ablines!(ax3, 0, 1, color = :red, linestyle = :dash)

    # 4. Beobachtet vs. Vorhersage
    ax4 = Axis(fig[2, 2], xlabel = "Vorhersage [s]", ylabel = "Beobachtet [s]",
               title = "Beobachtet vs. Vorhersage")
    scatter!(ax4, residuen.vorhersage, residuen.y,
             color = (:steelblue, 0.6), markersize = 6)
    ablines!(ax4, 0, 1, color = :red, linestyle = :dash)

    return fig
end

function Plot_Teil(teil::Teil,Regression::String= "Theil-Sen")
    fig = Figure(size = (1600,1000))
    Label(fig[0,1:4], teil.name, fontsize = 20, tellwidth = false)
    ax = Axis(fig[1:6,1:4],title = "Messpunkte und Regressionen", xlabel = "Rückmeldemenge", ylabel ="Gesamtzeit [s]")

    ok_df = filter(:Ausreißer => a -> a == "OK", teil.df)
    maybe_df = filter(:Ausreißer => a -> a == "Verdacht", teil.df)
    nope_df = filter(:Ausreißer => a -> a == "Ausreißer", teil.df)
    scatter!(ax,nope_df.rMenge, nope_df.tges,marker = :xcross, markersize = 15, color = :red3, label = "Ausreißer ($(nrow(nope_df)))")
    scatter!(ax,maybe_df.rMenge, maybe_df.tges,marker = '?' , markersize = 15, color = :orange, label = "Verdacht ($(nrow(maybe_df)))")
    scatter!(ax,ok_df.rMenge, ok_df.tges, color = :royalblue1, label = "OK ($(nrow(ok_df)))")

    reg = teil.RegressionsModell
    if haskey(reg, Regression)
        reg = teil.RegressionsModell[Regression]
        m = reg.stats.m[1]          # Steigung
        n = reg.stats.n[1]          # Achsenabschnitt
        ablines!(ax, n, m; linestyle = :dash, label = Regression, color = :purple)

        if hasproperty(reg.Diagnostik, :Q1)
            Intercept_Q2_5 = reg.Diagnostik.Q2_5[1]
            Intercept_Q1 = reg.Diagnostik.Q1[1]
            Intercept_Q3 = reg.Diagnostik.Q3[1]
            Intercept_Q97_5 = reg.Diagnostik.Q97_5[1]
            Slope_Q2_5 = reg.Diagnostik.Q2_5[3]
            Slope_Q1 = reg.Diagnostik.Q1[3]
            Slope_Q3 = reg.Diagnostik.Q3[3]
            Slope_Q97_5 = reg.Diagnostik.Q97_5[3]
            ablines!(ax, Intercept_Q2_5, Slope_Q2_5; linestyle = :dash, color = :red3,label= "2,5%")
            ablines!(ax, Intercept_Q1, Slope_Q1; linestyle = :dash, color = :firebrick1, label= "25,0%")
            ablines!(ax, Intercept_Q3, Slope_Q3; linestyle = :dash, color = :firebrick1, label= "75,0%")
            ablines!(ax, Intercept_Q97_5, Slope_Q97_5; linestyle = :dash, color = :red3,label= "97,5%")
        end

        # Residuen über Regressionsvorhersage
        bx = Axis(fig[7:8,1:3],title = "Residuen-Plot", xlabel = L"Gesamtzeit_{Regression}", ylabel = "Residuen" )
        scatter!(bx, reg.points.y_hat,reg.points.residuals)
        hlines!(bx,0.0,color=:red3)
        # Residuendichte
        cx = Axis(fig[7:10,4:5],title = "Residuen-Dichte", ylabel = "Residuen")
        density!(cx,reg.points.residuals, direction = :y )
        hlines!(cx,0.0 ,color=:red3)
        hidexdecorations!(cx)
        #linkyaxes!(bx, cx)

        # Residuen über RMNr
        foo(x) = n + m * x
        RMNr_residuen = teil.df.tges .- foo.(teil.df.rMenge)
        dx = Axis(fig[9:10,1:3],title = "Residuen vs. RMNr", xlabel = "RMNr", ylabel = "Residuen")
        scatter!(dx, teil.df.RMNr, RMNr_residuen)
        hlines!(dx,0.0,color=:red3)

        # QQ-Plot (robuste Referenzgerade y = median + MAD·z, da Makie kein
        # qqline-Attribut kennt — Entsprechung zu StatsPlots' qqline = :fitrobust.
        # MAD über robuste_skala_mad (Statistics bringt kein mad, das liegt in StatsBase))
        ex = Axis(fig[1:6,6:7], title = "QQ-Plot")
        qqnorm!(ex, reg.points.residuals)
        ablines!(ex, median(reg.points.residuals), robuste_skala_mad(reg.points.residuals);
                 color = :red3, linestyle = :dash)

        # Einflussdiagnostik: RStudent vs. Hebelwert mit Cook's-Distanz-Konturen.
        # Läuft über :TheilSen auf der YJ-Skala (FORMELN[:yeojohnson]), denn
        # genau dort wurden die Stufe-1-Flags berechnet (PrepareData!, Pass 1
        # in theil_sen_regression_yj!) — Panel und Einstufung sind so
        # konsistent. Hebelwerte sind reine Design-Größen; nur rstudent/cooks
        # hängen vom Schätzer ab. Hinweis: :yj_tges in teil.df trägt das finale
        # λ̂ aus Pass 2, die Flags wurden mit dem Pass-1-λ̂ berechnet — bei
        # abweichendem λ̂ verschieben sich die Punkte minimal.
        diag = dropmissing(loo_mad_diagnostics(teil, :TheilSen, FORMELN[:yeojohnson];
                                               auswahl = (:rstudent, :hebel)),
                           [:rstudent_mad, :hebel_loo])

        p = 2   # Parameter: Intercept + Steigung (bei :ptfe wäre p = 3)
        fx = Axis(fig[7:10, 6:7], title = "RStudent vs. Hebelwert (YJ-Skala)",
                  xlabel = "Hebelwert (LOO)", ylabel = "RStudent (MAD)")

        # Y-Limits: größter |RStudent|, auf runde Schritte hochgerechnet und
        # symmetrisch um 0 (passend zu den symmetrischen Cook's-Konturen)
        r_max = maximum(abs, diag.rstudent_mad)
        schritt = r_max <= 3 ? 0.5 : r_max <= 10 ? 1.0 : 5.0
        r_limit = ceil(r_max / schritt) * schritt
        ylims!(fx, -r_limit, r_limit)

        # Cook's-Distanz-Konturen: t = ±√(c·p/h)
        x_max = 1.15 * maximum(diag.hebel_loo)
        xs = collect(range(1e-4, x_max; length = 400))
        grenzwerte = [(4.0 / nrow(teil.df), L"\frac{4}{n}"), (0.5, L"0.5"), (1.0, L"1.0")]
        for (c, lab) in grenzwerte
            lines!(fx, xs,  sqrt.(c .* p ./ xs); color = :tomato3, linestyle = :dot)
            lines!(fx, xs, -sqrt.(c .* p ./ xs); color = :tomato3, linestyle = :dot)
            text!(fx, Point2f(x_max,  sqrt(c * p / x_max)); text = lab,
                  color = :tomato3, fontsize = 14, font = :bold)
            text!(fx, Point2f(x_max, -sqrt(c * p / x_max)); text = lab,
                  color = :tomato3, fontsize = 14, font = :bold)
        end
        # Einfärben nach Einstufung. Achtung: Vector{String} versteht Makie
        # als Farbnamen, nicht als Kategorien — daher explizite Zuordnung
        # (gleiche Farben wie im Messpunkte-Panel)
        einstufung_farben = Dict("OK" => :royalblue1, "Verdacht" => :orange, "Ausreißer" => :red3)
        scatter!(fx, diag.hebel_loo, diag.rstudent_mad;
                 color = [get(einstufung_farben, a, :gray) for a in teil.df.Ausreißer[diag.row]])
    else
        @warn "Teil $(teil.name): kein Regressionsmodell \"$Regression\" hinterlegt"    
    end
    Legend(fig[1:4, 5], ax , framevisible = false)

    return fig
end